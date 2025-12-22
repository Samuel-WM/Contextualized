#!/usr/bin/env python3
"""
bench_scale_contextualized_regression.py

Synthetic scaling benchmark for Contextualized regression workflow.

Modes:
  - run   : run a single config (supports torch distributed launch)
  - sweep : run CPU + GPU(1..K) sequentially (spawns torch distributed runs) and plot

Examples (Lambda 4x GPU single node):
  # Full sweep: CPU + 1/2/3/4 GPU and plots
  python bench_scale_contextualized_regression.py sweep \
    --include_cpu \
    --max_gpus 4 \
    --n 200000 \
    --c_dim 16 --x_dim 64 --y_dim 8 \
    --epochs 5 \
    --train_batch_size 2048 --val_batch_size 2048 --test_batch_size 4096 \
    --num_workers 4 \
    --out_dir ./scale_runs/run1

  # Single run on 1 GPU (no torchrun needed for 1 device)
  python bench_scale_contextualized_regression.py run --accelerator gpu --devices 1 --out_dir ./one_gpu

  # Single run on 4 GPUs using torch distributed launcher
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    bench_scale_contextualized_regression.py run --accelerator gpu --devices 4 --out_dir ./four_gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# -------------------------
# Utilities
# -------------------------

def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _maybe_git_commit() -> Optional[str]:
    try:
        if not shutil.which("git"):
            return None
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        if re.fullmatch(r"[0-9a-f]{40}", out):
            return out
    except Exception:
        pass
    return None


def _rank_world() -> Tuple[int, int, int]:
    """(rank, world_size, local_rank) for torchrun-style environments."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _cuda_sync_if_available() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Synthetic data generator
# -------------------------

def make_synth_contextual_regression(
    n: int,
    c_dim: int,
    x_dim: int,
    y_dim: int,
    noise: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Create synthetic data where coefficients beta(C) vary with context C:
      beta_flat = C @ W^T + b
      beta      = reshape(beta_flat, y_dim, x_dim)
      mu        = C @ V
      y         = sum_j beta[..., j]*x_j + mu + eps
    """
    rng = np.random.default_rng(seed)

    C = rng.normal(size=(n, c_dim)).astype(np.float32)
    X = rng.normal(size=(n, x_dim)).astype(np.float32)

    # Context -> beta mapping
    W = (rng.normal(size=(y_dim * x_dim, c_dim)).astype(np.float32) / np.sqrt(c_dim)).astype(np.float32)
    b = (0.1 * rng.normal(size=(y_dim * x_dim,))).astype(np.float32)

    beta_flat = C @ W.T + b[None, :]             # (n, y_dim*x_dim)
    beta = beta_flat.reshape(n, y_dim, x_dim)    # (n, y_dim, x_dim)

    # Context -> intercept
    V = (0.1 * rng.normal(size=(c_dim, y_dim))).astype(np.float32)
    mu = (C @ V).astype(np.float32)              # (n, y_dim)

    y = (beta * X[:, None, :]).sum(axis=-1) + mu
    y = y + noise * rng.normal(size=(n, y_dim)).astype(np.float32)
    Y = y.astype(np.float32)

    truth = {"beta": beta, "mu": mu}
    return C, X, Y, truth


def make_splits(n: int, val_frac: float, test_frac: float, seed: int) -> Dict[str, np.ndarray]:
    assert 0.0 <= val_frac < 1.0
    assert 0.0 <= test_frac < 1.0
    assert val_frac + test_frac < 1.0

    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


# -------------------------
# Result schema
# -------------------------

@dataclass
class BenchResult:
    tag: str
    accelerator: str
    devices: int
    backend: str
    n: int
    n_train: int
    n_val: int
    n_test: int
    c_dim: int
    x_dim: int
    y_dim: int
    epochs: int
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    num_workers: int
    seed: int

    fit_time_s: float
    predict_time_s: float
    total_time_s: float

    train_throughput_sps: float  # samples/sec (global, unique samples per epoch)
    predict_throughput_sps: float

    test_mse: float

    hostname: str
    python: str
    platform: str
    torch: Optional[str]
    lightning: Optional[str]
    git_commit: Optional[str]


# -------------------------
# Core runner
# -------------------------

def run_one(args: argparse.Namespace) -> Optional[BenchResult]:
    rank, world, local_rank = _rank_world()

    # Make output directory (all ranks see it; only rank0 writes result).
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    # Deferred imports so CPU-only environments don't choke on CUDA imports early.
    try:
        import torch
        import pytorch_lightning as pl
    except Exception as e:
        if rank == 0:
            raise RuntimeError(
                "Failed to import torch / pytorch_lightning. Ensure your env has them installed."
            ) from e
        return None

    # Set device for GPU
    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            if rank == 0:
                raise RuntimeError("Requested accelerator=gpu but torch.cuda.is_available() is False.")
            return None
        torch.cuda.set_device(local_rank)

    # Determinism (best-effort)
    _set_seed(args.seed)
    try:
        pl.seed_everything(args.seed, workers=True)
    except Exception:
        pass

    # Synthesize data (replicated across ranks; ok for benchmarking)
    C, X, Y, _truth = make_synth_contextual_regression(
        n=args.n,
        c_dim=args.c_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        noise=args.noise,
        seed=args.seed,
    )

    splits = make_splits(args.n, args.val_frac, args.test_frac, args.seed)
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    # Build model using your regression workflow
    # We prefer the easy wrapper if available, as it exercises your end-to-end stack.
    try:
        from contextualized.easy import ContextualizedRegressor
    except Exception as e:
        if rank == 0:
            raise RuntimeError(
                "Could not import contextualized.easy.ContextualizedRegressor. "
                "Verify your package is importable from this environment."
            ) from e
        return None

    # Strategy configuration (DDP for multi-GPU)
    strategy_obj = None
    if args.devices > 1:
        # Use Lightning DDPStrategy explicitly to control backend (nccl/gloo)
        try:
            from pytorch_lightning.strategies import DDPStrategy
            strategy_obj = DDPStrategy(
                process_group_backend=args.backend,
                find_unused_parameters=False,
            )
        except Exception:
            strategy_obj = "ddp"  # fallback: let Lightning decide

    # Create the regressor (robust to wrapper signature differences)
    # We attempt common kwargs; if wrapper rejects, we show a clear error on rank0.
    model_kwargs: Dict[str, Any] = dict(
        num_archetypes=args.num_archetypes,
        encoder_type=args.encoder_type,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # data / loader knobs (if wrapper exposes them)
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_frac,
        # trainer knobs
        accelerator=("gpu" if args.accelerator == "gpu" else "cpu"),
        devices=args.devices,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    # Some wrappers may not accept the above keys; strip unsupported keys dynamically.
    def instantiate_contextualized_regressor() -> Any:
        import inspect
        sig = inspect.signature(ContextualizedRegressor.__init__)
        accepted = set(sig.parameters.keys())
        # Always remove 'self'
        accepted.discard("self")
        filt = {k: v for k, v in model_kwargs.items() if k in accepted}
        # Strategy: some wrappers accept "strategy" directly
        if "strategy" in accepted and strategy_obj is not None:
            filt["strategy"] = strategy_obj
        return ContextualizedRegressor(**filt)

    try:
        reg = instantiate_contextualized_regressor()
    except TypeError as e:
        if rank == 0:
            raise RuntimeError(
                "Failed to instantiate ContextualizedRegressor with inferred kwargs.\n"
                "This usually means the wrapper signature differs from what this benchmark expects.\n"
                "Action: open contextualized/easy/wrappers.py and confirm which Trainer/loader args are supported.\n"
                f"Original error: {e}"
            )
        return None

    # Fit timing
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    # Prefer explicit indices to exercise your stable-index paths if supported
    fit_kwargs: Dict[str, Any] = dict(C=C, X=X, Y=Y)
    try:
        import inspect
        fit_sig = inspect.signature(reg.fit)
        if "train_idx" in fit_sig.parameters:
            fit_kwargs["train_idx"] = train_idx
        if "val_idx" in fit_sig.parameters:
            fit_kwargs["val_idx"] = val_idx
        if "test_idx" in fit_sig.parameters:
            fit_kwargs["test_idx"] = test_idx
    except Exception:
        pass

    reg.fit(**fit_kwargs)

    _cuda_sync_if_available()
    fit_time = time.perf_counter() - t0

    # Predict timing (prefer predict_idx if supported)
    _cuda_sync_if_available()
    t1 = time.perf_counter()

    yhat = None
    pred_kwargs_full: Dict[str, Any] = dict(C=C, X=X)
    try:
        import inspect
        pred_sig = inspect.signature(reg.predict)
        if "predict_idx" in pred_sig.parameters:
            pred_kwargs_full["predict_idx"] = test_idx
            yhat = reg.predict(**pred_kwargs_full)
        else:
            # fallback: feed the subset directly
            yhat = reg.predict(C[test_idx], X[test_idx])
    except Exception:
        # fallback: feed the subset directly
        yhat = reg.predict(C[test_idx], X[test_idx])

    _cuda_sync_if_available()
    pred_time = time.perf_counter() - t1

    # Only rank0 should compute/report metrics if wrapper returns None on non-rank0
    if yhat is None:
        return None

    # Convert prediction to numpy
    if hasattr(yhat, "detach"):
        yhat_np = yhat.detach().cpu().numpy()
    else:
        yhat_np = np.asarray(yhat)

    y_true = Y[test_idx]
    test_mse = float(np.mean((yhat_np - y_true) ** 2))

    total_time = fit_time + pred_time

    # Throughput (global unique samples per epoch)
    n_train = int(train_idx.shape[0])
    n_val = int(val_idx.shape[0])
    n_test = int(test_idx.shape[0])

    train_throughput = (n_train * args.epochs) / max(fit_time, 1e-9)
    pred_throughput = (n_test) / max(pred_time, 1e-9)

    # Version info
    torch_ver = getattr(torch, "__version__", None)
    lightning_ver = getattr(pl, "__version__", None)

    tag = args.tag or f"{args.accelerator}_{args.devices}dev"
    result = BenchResult(
        tag=tag,
        accelerator=args.accelerator,
        devices=args.devices,
        backend=args.backend,

        n=args.n,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        c_dim=args.c_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,

        fit_time_s=_safe_float(fit_time),
        predict_time_s=_safe_float(pred_time),
        total_time_s=_safe_float(total_time),

        train_throughput_sps=_safe_float(train_throughput),
        predict_throughput_sps=_safe_float(pred_throughput),

        test_mse=_safe_float(test_mse),

        hostname=platform.node(),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch=torch_ver,
        lightning=lightning_ver,
        git_commit=_maybe_git_commit(),
    )

    # Write per-run JSON on rank0 only
    if rank == 0:
        out_json = out_dir / f"result_{tag}.json"
        with out_json.open("w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"[rank0] Wrote: {out_json}")
        print(
            f"[rank0] fit={result.fit_time_s:.3f}s "
            f"pred={result.predict_time_s:.3f}s "
            f"total={result.total_time_s:.3f}s "
            f"train_thr={result.train_throughput_sps:.1f} samp/s "
            f"pred_thr={result.predict_throughput_sps:.1f} samp/s "
            f"test_mse={result.test_mse:.6f}"
        )

    return result


# -------------------------
# Sweep + plotting
# -------------------------

def _load_results(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for p in sorted(out_dir.glob("result_*.json")):
        with p.open("r") as f:
            d = json.load(f)
        results[d["tag"]] = d
    return results


def _write_csv(out_dir: Path, rows: Dict[str, Dict[str, Any]]) -> Path:
    out_csv = out_dir / "results.csv"
    keys = sorted(next(iter(rows.values())).keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for _tag, d in sorted(rows.items(), key=lambda kv: (kv[1]["accelerator"], kv[1]["devices"])):
            w.writerow({k: d.get(k, None) for k in keys})
    return out_csv


def plot_results(out_dir: Path, baseline_devices: int = 1, include_cpu_speedup: bool = True) -> None:
    # Use non-interactive backend for headless servers
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_results(out_dir)
    if not rows:
        raise RuntimeError(f"No result_*.json found in {out_dir}")

    # Split CPU vs GPU
    cpu = [d for d in rows.values() if d["accelerator"] == "cpu"]
    gpu = [d for d in rows.values() if d["accelerator"] == "gpu"]

    gpu_sorted = sorted(gpu, key=lambda d: d["devices"])
    cpu_sorted = sorted(cpu, key=lambda d: d["devices"])

    # Baseline for speedup/efficiency
    base_gpu = next((d for d in gpu_sorted if d["devices"] == baseline_devices), None)
    if base_gpu is None and gpu_sorted:
        base_gpu = gpu_sorted[0]

    # Helper series
    def series(ds, key):
        return [float(d[key]) for d in ds]

    # Plot 1: wall time (fit/predict/total) vs devices (GPU)
    if gpu_sorted:
        x = [d["devices"] for d in gpu_sorted]

        fit_t = series(gpu_sorted, "fit_time_s")
        pred_t = series(gpu_sorted, "predict_time_s")
        tot_t = series(gpu_sorted, "total_time_s")

        plt.figure(figsize=(8, 5))
        plt.plot(x, fit_t, marker="o", label="fit_time_s")
        plt.plot(x, pred_t, marker="o", label="predict_time_s")
        plt.plot(x, tot_t, marker="o", label="total_time_s")
        plt.xlabel("GPUs (devices)")
        plt.ylabel("Seconds")
        plt.title("Wall time vs GPUs")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.legend()
        p = out_dir / "wall_time_vs_gpus.png"
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # Plot 2: throughput vs devices (GPU)
    if gpu_sorted:
        x = [d["devices"] for d in gpu_sorted]
        thr = series(gpu_sorted, "train_throughput_sps")

        plt.figure(figsize=(8, 5))
        plt.plot(x, thr, marker="o")
        plt.xlabel("GPUs (devices)")
        plt.ylabel("Train throughput (samples/sec, global)")
        plt.title("Train throughput vs GPUs")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        p = out_dir / "throughput_vs_gpus.png"
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # Plot 3: speedup + efficiency vs devices (GPU)
    if gpu_sorted and base_gpu is not None:
        x = [d["devices"] for d in gpu_sorted]
        base_thr = float(base_gpu["train_throughput_sps"])
        speedup = [float(d["train_throughput_sps"]) / max(base_thr, 1e-9) for d in gpu_sorted]
        efficiency = [s / max(dev, 1e-9) for s, dev in zip(speedup, x)]

        plt.figure(figsize=(8, 5))
        plt.plot(x, speedup, marker="o", label=f"Speedup vs {base_gpu['devices']} GPU")
        plt.plot(x, x, linestyle="--", label="Ideal linear speedup")
        plt.xlabel("GPUs (devices)")
        plt.ylabel("Speedup")
        plt.title("Speedup vs GPUs")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.legend()
        p = out_dir / "speedup_vs_gpus.png"
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(x, efficiency, marker="o")
        plt.xlabel("GPUs (devices)")
        plt.ylabel("Scaling efficiency (speedup / GPUs)")
        plt.title("Scaling efficiency vs GPUs")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        p = out_dir / "efficiency_vs_gpus.png"
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # Optional: CPU vs best GPU throughput comparison
    if include_cpu_speedup and cpu_sorted and gpu_sorted:
        cpu_thr = float(cpu_sorted[0]["train_throughput_sps"])
        best_gpu = max(gpu_sorted, key=lambda d: float(d["train_throughput_sps"]))
        best_thr = float(best_gpu["train_throughput_sps"])
        ratio = best_thr / max(cpu_thr, 1e-9)

        plt.figure(figsize=(8, 5))
        labels = ["CPU (1)", f"GPU ({best_gpu['devices']})"]
        vals = [cpu_thr, best_thr]
        plt.bar(labels, vals)
        plt.ylabel("Train throughput (samples/sec, global)")
        plt.title(f"CPU vs best GPU throughput (GPU/CPU = {ratio:.2f}x)")
        plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        p = out_dir / "cpu_vs_best_gpu_throughput.png"
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # Write CSV for convenience
    out_csv = _write_csv(out_dir, rows)
    print(f"Wrote plots + {out_csv}")


def sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    # Write run config
    run_cfg = vars(args).copy()
    run_cfg["timestamp"] = _now_ts()
    run_cfg["git_commit"] = _maybe_git_commit()
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_cfg, f, indent=2)

    # Build base run args (forwarded to run subcommand)
    base_run = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--n", str(args.n),
        "--c_dim", str(args.c_dim),
        "--x_dim", str(args.x_dim),
        "--y_dim", str(args.y_dim),
        "--noise", str(args.noise),
        "--val_frac", str(args.val_frac),
        "--test_frac", str(args.test_frac),
        "--epochs", str(args.epochs),
        "--train_batch_size", str(args.train_batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--test_batch_size", str(args.test_batch_size),
        "--num_workers", str(args.num_workers),
        "--learning_rate", str(args.learning_rate),
        "--num_archetypes", str(args.num_archetypes),
        "--encoder_type", str(args.encoder_type),
        "--backend", str(args.backend),
        "--seed", str(args.seed),
        "--out_dir", str(out_dir),
    ]
    if args.deterministic:
        base_run.append("--deterministic")

    # 1) CPU (single proc)
    if args.include_cpu:
        cmd = base_run + ["--accelerator", "cpu", "--devices", "1", "--tag", "cpu_1dev"]
        print("\n=== Running CPU (1 device) ===")
        subprocess.run(cmd, check=True)

    # 2) GPU sweeps
    for k in range(1, args.max_gpus + 1):
        tag = f"gpu_{k}dev"
        print(f"\n=== Running GPU ({k} device{'s' if k > 1 else ''}) ===")
        if k == 1 and not args.force_torchrun_for_1gpu:
            # Single process GPU
            cmd = base_run + ["--accelerator", "gpu", "--devices", "1", "--tag", tag]
            subprocess.run(cmd, check=True)
        else:
            # Multi-process launch (also works for 1 GPU if forced)
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={k}",
                str(Path(__file__).resolve()),
                "run",
                "--accelerator", "gpu",
                "--devices", str(k),
                "--tag", tag,
                "--out_dir", str(out_dir),
                "--n", str(args.n),
                "--c_dim", str(args.c_dim),
                "--x_dim", str(args.x_dim),
                "--y_dim", str(args.y_dim),
                "--noise", str(args.noise),
                "--val_frac", str(args.val_frac),
                "--test_frac", str(args.test_frac),
                "--epochs", str(args.epochs),
                "--train_batch_size", str(args.train_batch_size),
                "--val_batch_size", str(args.val_batch_size),
                "--test_batch_size", str(args.test_batch_size),
                "--num_workers", str(args.num_workers),
                "--learning_rate", str(args.learning_rate),
                "--num_archetypes", str(args.num_archetypes),
                "--encoder_type", str(args.encoder_type),
                "--backend", str(args.backend),
                "--seed", str(args.seed),
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            subprocess.run(cmd, check=True)

    # Plot at end
    plot_results(out_dir, baseline_devices=args.speedup_baseline_gpu)


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contextualized regression scaling benchmark (synthetic).")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out_dir", type=str, required=True, help="Output directory for results and plots.")
    common.add_argument("--tag", type=str, default="", help="Tag for this run; used in filename result_<tag>.json")

    common.add_argument("--n", type=int, default=200_000)
    common.add_argument("--c_dim", type=int, default=16)
    common.add_argument("--x_dim", type=int, default=64)
    common.add_argument("--y_dim", type=int, default=8)
    common.add_argument("--noise", type=float, default=0.5)

    common.add_argument("--val_frac", type=float, default=0.2)
    common.add_argument("--test_frac", type=float, default=0.1)

    common.add_argument("--epochs", type=int, default=5)
    common.add_argument("--train_batch_size", type=int, default=2048)
    common.add_argument("--val_batch_size", type=int, default=2048)
    common.add_argument("--test_batch_size", type=int, default=4096)
    common.add_argument("--num_workers", type=int, default=4)

    common.add_argument("--learning_rate", type=float, default=1e-3)
    common.add_argument("--num_archetypes", type=int, default=0)
    common.add_argument("--encoder_type", type=str, default="mlp")

    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])

    common.add_argument("--deterministic", action="store_true", help="Best-effort deterministic training.")

    # run
    pr = sub.add_parser("run", parents=[common], help="Run one benchmark configuration.")
    pr.add_argument("--accelerator", type=str, required=True, choices=["cpu", "gpu"])
    pr.add_argument("--devices", type=int, required=True)

    # sweep
    ps = sub.add_parser("sweep", parents=[common], help="Run CPU + GPU(1..K) sweep and plot.")
    ps.add_argument("--include_cpu", action="store_true", help="Include a CPU baseline run.")
    ps.add_argument("--max_gpus", type=int, default=4, help="Max GPUs to sweep up to (inclusive).")
    ps.add_argument("--force_torchrun_for_1gpu", action="store_true",
                   help="Also launch 1-GPU via torch.distributed.run for consistency.")
    ps.add_argument("--speedup_baseline_gpu", type=int, default=1,
                   help="Baseline GPU count for speedup/efficiency plots (default: 1).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        # In DDP, only rank0 writes; non-rank0 returns None
        run_one(args)

    elif args.cmd == "sweep":
        sweep(args)

    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
