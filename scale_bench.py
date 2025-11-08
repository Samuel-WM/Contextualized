#!/usr/bin/env python3
# scale_bench.py
"""
Scalability benchmark for Contextualized-ML (single-node, Lambda GPU instance).

Runs 4 configs with a FIXED GLOBAL BATCH SIZE:
  - 1 CPU
  - 1 GPU
  - 2 GPUs (DDP)
  - 4 GPUs (DDP)

Outputs:
  - results/bench_results.csv
  - results/scaling_samples_per_sec.png
  - results/scaling_wallclock.png
  - results/scaling_epoch_time.png
  - results/scaling_convergence_time.png

Requirements:
  - Your package importable on the instance.
  - PyTorch + Lightning working w/ CUDA.
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, subprocess, shutil, uuid, signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch

# ---- Your package imports (as provided) ----
from contextualized.regression.trainers import RegressionTrainer, make_trainer_with_env
from contextualized.regression.models import ContextualizedRegression   # adjust path if different
from contextualized.regression.datamodules import ContextualizedRegressionDataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# =========================
# Synthetic data generator
# =========================
def make_synth(n=200_000, c_dim=16, x_dim=64, y_dim=1, noise=0.10, seed=123):
    """
    Multivariate regression with context-conditioned parameters:
      Y = g( beta(C)*X + mu(C) ) + noise
    g = identity (MSE)
    """
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, c_dim)).astype(np.float32)
    X = rng.normal(size=(n, x_dim)).astype(np.float32)

    # Context-conditioned weights: low-rank projection from C -> (y_dim, x_dim) and mu
    Wc = rng.normal(scale=0.5, size=(c_dim, y_dim * x_dim)).astype(np.float32)
    Wm = rng.normal(scale=0.5, size=(c_dim, y_dim)).astype(np.float32)

    beta_flat = C @ Wc                       # (n, y_dim*x_dim)
    beta = beta_flat.reshape(n, y_dim, x_dim)
    mu   = (C @ Wm).reshape(n, y_dim, 1)     # (n, y_dim, 1)

    # Broadcast X to (n, y_dim, x_dim) for multivariate form
    Xb = np.expand_dims(X, 1).repeat(y_dim, axis=1)
    y_true = (beta * Xb).sum(axis=-1, keepdims=True) + mu   # (n, y_dim, 1)

    Y = y_true + noise * rng.normal(size=y_true.shape).astype(np.float32)
    Y = Y.squeeze(-1)  # (n, y_dim)
    return C, X, Y


# =========================
# Metrics callback
# =========================
class MetricsCallback(Callback):
    """
    Collect:
      - wall-clock time
      - per-epoch time (avg)
      - total epochs
      - samples/sec (global)
      - convergence time & steps (val_loss <= target; or train_loss if no val)
      - max memory (GPU or CPU)
    Only rank 0 logs final metrics in DDP.
    """
    def __init__(self, global_batch_size: int, train_size: int, target_loss: float, use_val: bool):
        super().__init__()
        self.global_batch = global_batch_size
        self.train_size   = train_size
        self.target_loss  = target_loss
        self.use_val      = use_val

        self.t0 = None
        self.epoch_starts = []
        self.epoch_durations = []
        self.total_epochs = 0

        self.converged = False
        self.convergence_epoch = None
        self.convergence_time_s = None
        self.gradient_steps = None

        self.max_gpu_mem = 0
        self.max_cpu_rss = 0  # placeholder (psutil optional)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.t0 = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_starts.append(time.perf_counter())

    def on_train_epoch_end(self, trainer, pl_module):
        t1 = time.perf_counter()
        if self.epoch_starts:
            self.epoch_durations.append(t1 - self.epoch_starts[-1])
        self.total_epochs += 1

        # Track GPU memory (max) on rank 0 if CUDA
        if torch.cuda.is_available() and torch.cuda.current_device() == 0:
            self.max_gpu_mem = max(self.max_gpu_mem, torch.cuda.max_memory_reserved(0))

        # Convergence check at end of epoch (val preferred)
        if not self.converged:
            metrics = trainer.callback_metrics
            key = "val_loss" if self.use_val and ("val_loss" in metrics) else "train_loss"
            loss_val = float(metrics.get(key, float("inf")))
            if loss_val <= self.target_loss:
                self.converged = True
                self.convergence_epoch = self.total_epochs
                self.convergence_time_s = time.perf_counter() - self.t0
                # gradient steps up to and including this epoch
                steps_per_epoch = math.ceil(self.train_size / self.global_batch)
                self.gradient_steps = steps_per_epoch * self.convergence_epoch

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass

    def finalize(self, trainer: pl.Trainer) -> Dict[str, Any]:
        wall = time.perf_counter() - self.t0 if self.t0 else None
        avg_epoch = (sum(self.epoch_durations) / len(self.epoch_durations)) if self.epoch_durations else None
        total_samples = self.train_size * (self.total_epochs if self.total_epochs else 0)
        sps = (total_samples / wall) if wall and wall > 0 else None

        # Convert bytes -> GiB for readability
        mem_gib = (self.max_gpu_mem / (1024**3)) if self.max_gpu_mem else 0.0

        return dict(
            wall_clock_s=wall,
            epoch_time_s=avg_epoch,
            total_epochs=self.total_epochs,
            samples_per_sec=sps,
            convergence_time_s=(self.convergence_time_s if self.converged else None),
            gradient_steps=(self.gradient_steps if self.converged else None),
            max_memory_gib=mem_gib,
        )


# =========================
# Runner: one configuration
# =========================
@dataclass
class RunConfig:
    label: str                  # e.g., "cpu-1", "gpu-1", "gpu-2", "gpu-4"
    accelerator: str            # "cpu" or "gpu"
    devices: int                # 1,2,4
    strategy: str               # "auto" or "ddp"
    global_batch: int

@dataclass
class RunResult:
    hardware: str
    wall_clock_s: Optional[float]
    epoch_time_s: Optional[float]
    total_epochs: int
    samples_per_sec: Optional[float]
    convergence_time_s: Optional[float]
    gradient_steps: Optional[int]
    max_memory_gib: Optional[float]

def per_device_batch(global_batch: int, world_size: int) -> int:
    if world_size < 1:
        world_size = 1
    b = max(1, global_batch // world_size)
    if b * world_size != global_batch:
        print(f"[warn] global_batch={global_batch} not divisible by world_size={world_size}; "
              f"using per_device_batch={b} (effective global={b*world_size}).")
    return b

def run_single_config(args, cfg: RunConfig) -> RunResult:
    # Synthesize data
    C, X, Y = make_synth(
        n=args.n, c_dim=args.c_dim, x_dim=args.x_dim, y_dim=args.y_dim,
        noise=args.noise, seed=args.seed
    )

    # Split indices (simple holdout)
    n = C.shape[0]
    n_val = int(n * args.val_split)
    permutation = np.random.default_rng(args.seed).permutation(n)
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]

    world_size = cfg.devices if cfg.accelerator == "gpu" and cfg.strategy == "ddp" else 1
    eff_per_device = per_device_batch(cfg.global_batch, world_size)

    # DataModule (map-style; Lightning will shard w/ DistributedSampler)
    dm = ContextualizedRegressionDataModule(
        C=C, X=X, Y=Y,
        task_type="singletask_multivariate",
        train_idx=train_idx, val_idx=val_idx, test_idx=None,
        predict_idx=val_idx,
        train_batch_size=eff_per_device,
        val_batch_size=eff_per_device,
        test_batch_size=eff_per_device,
        predict_batch_size=eff_per_device,
        num_workers=args.num_workers,
        pin_memory=(cfg.accelerator == "gpu"),
        persistent_workers=bool(args.num_workers > 0),
        drop_last=False,
        shuffle_train=True,
        shuffle_eval=False,
        dtype=torch.float,
    )
    dm.prepare_data(); dm.setup()

    # Model
    model = ContextualizedRegression(
        context_dim=args.c_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        num_archetypes=args.archetypes,
        encoder_type="mlp",
        encoder_kwargs=dict(width=args.width, layers=args.layers, link_fn="identity"),
        learning_rate=args.lr,
        metamodel_type="subtype",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    )

    # Metrics
    use_val = args.val_split > 0.0
    mcb = MetricsCallback(
        global_batch_size=(eff_per_device * world_size),
        train_size=len(train_idx),
        target_loss=args.target_loss,
        use_val=use_val,
    )

    # Trainer (via your factory)
    trainer = make_trainer_with_env(
        RegressionTrainer,
        max_epochs=args.max_epochs,
        enable_progress_bar=False,
        logger=False,
        accelerator=cfg.accelerator,
        devices=(cfg.devices if cfg.accelerator == "gpu" else 1),
        strategy=cfg.strategy,         # "ddp" or "auto"
        precision=32,
        callbacks=[mcb],
        # sanity & val
        num_sanity_val_steps=0,
        limit_val_batches=(1.0 if use_val else 0.0),
    )

    # Fit
    if use_val and dm.val_dataloader() is not None:
        trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    else:
        trainer.fit(model, train_dataloaders=dm.train_dataloader())

    # Finalize metrics (rank-0 only meaningful; in non-ddp it's fine)
    metrics = mcb.finalize(trainer)

    return RunResult(
        hardware=cfg.label,
        wall_clock_s=metrics["wall_clock_s"],
        epoch_time_s=metrics["epoch_time_s"],
        total_epochs=metrics["total_epochs"],
        samples_per_sec=metrics["samples_per_sec"],
        convergence_time_s=metrics["convergence_time_s"],
        gradient_steps=(int(metrics["gradient_steps"]) if metrics["gradient_steps"] is not None else None),
        max_memory_gib=metrics["max_memory_gib"],
    )


# =========================
# Sweep driver (single node)
# =========================
def run_sweep(args):
    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)
    table_csv = results_dir / "bench_results.csv"

    # Default sweep: 1 CPU, 1 GPU, 2 GPU, 4 GPU (skip GPU configs if no CUDA)
    cuda_ok = torch.cuda.is_available()

    sweep: List[RunConfig] = [
        RunConfig("cpu-1", "cpu", 1, "auto", args.global_batch),
    ]
    if cuda_ok:
        # respect available device count
        ndev = torch.cuda.device_count()
        if ndev >= 1: sweep.append(RunConfig("gpu-1", "gpu", 1, "auto", args.global_batch))
        if ndev >= 2: sweep.append(RunConfig("gpu-2", "gpu", 2, "ddp", args.global_batch))
        if ndev >= 4: sweep.append(RunConfig("gpu-4", "gpu", 4, "ddp", args.global_batch))

    rows: List[RunResult] = []
    for cfg in sweep:
        print(f"\n=== Running config: {cfg.label}  (accelerator={cfg.accelerator}, devices={cfg.devices}, strategy={cfg.strategy}) ===")
        rr = run_single_config(args, cfg)
        rows.append(rr)
        print(f" -> Done {cfg.label}: wall={rr.wall_clock_s:.2f}s, sps={rr.samples_per_sec:.1f}, epochs={rr.total_epochs}")

    # Write CSV
    import csv
    with table_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hardware config", "wall-clock (s)", "epoch time (s)", "total epochs",
                    "samples/sec", "convergence time (s)", "gradient steps", "max memory (GiB)"])
        for r in rows:
            w.writerow([
                r.hardware,
                f"{r.wall_clock_s:.6f}" if r.wall_clock_s is not None else "",
                f"{r.epoch_time_s:.6f}" if r.epoch_time_s is not None else "",
                r.total_epochs,
                f"{r.samples_per_sec:.2f}" if r.samples_per_sec is not None else "",
                f"{r.convergence_time_s:.6f}" if r.convergence_time_s is not None else "",
                r.gradient_steps if r.gradient_steps is not None else "",
                f"{r.max_memory_gib:.3f}" if r.max_memory_gib is not None else "",
            ])
    print(f"\nSaved table -> {table_csv}")

    # Plots vs #GPUs (CPU shown at 0 GPUs on x-axis)
    def hw_to_ngpu(lbl: str) -> int:
        if lbl.startswith("cpu"): return 0
        return int(lbl.split("-")[1])

    rows_sorted = sorted(rows, key=lambda r: hw_to_ngpu(r.hardware))
    x = [hw_to_ngpu(r.hardware) for r in rows_sorted]

    def plot_metric(name: str, vals: List[Optional[float]], fname: str, ylab: str):
        xs, ys = [], []
        for xi, v in zip(x, vals):
            if v is not None:
                xs.append(xi); ys.append(v)
        if not xs:
            return
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("# GPUs (CPU plotted as 0)")
        plt.ylabel(ylab)
        plt.title(f"{name} vs #GPUs")
        plt.grid(True)
        outp = results_dir / fname
        plt.savefig(outp, bbox_inches="tight")
        print(f"Saved plot -> {outp}")

    plot_metric("Throughput (samples/sec)",
                [r.samples_per_sec for r in rows_sorted],
                "scaling_samples_per_sec.png", "samples/sec")

    plot_metric("Wall-clock",
                [r.wall_clock_s for r in rows_sorted],
                "scaling_wallclock.png", "seconds")

    plot_metric("Epoch time",
                [r.epoch_time_s for r in rows_sorted],
                "scaling_epoch_time.png", "seconds/epoch")

    plot_metric("Convergence time",
                [r.convergence_time_s for r in rows_sorted],
                "scaling_convergence_time.png", "seconds")


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Contextualized-ML scalability benchmark")
    # Data
    p.add_argument("--n", type=int, default=200_000, help="number of samples (synthetic)")
    p.add_argument("--c-dim", type=int, default=16)
    p.add_argument("--x-dim", type=int, default=64)
    p.add_argument("--y-dim", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=123)

    # Training
    p.add_argument("--global-batch", type=int, default=4096, help="fixed global batch across configs")
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--archetypes", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--target-loss", type=float, default=0.02,
                   help="convergence threshold on (val_loss or train_loss)")

    # I/O
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--mode", choices=["sweep", "single"], default="sweep",
                   help="sweep = run CPU+GPU configs; single = run one config given below")
    # For --mode single (debugging)
    p.add_argument("--single-accel", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--single-devices", type=int, default=1)
    p.add_argument("--single-strategy", choices=["auto", "ddp"], default="auto")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "sweep":
        run_sweep(args)
    else:
        label = f"{args.single_accel}-{args.single_devices}"
        cfg = RunConfig(label, args.single_accel, args.single_devices, args.single_strategy, args.global_batch)
        rr = run_single_config(args, cfg)
        print(json.dumps(asdict(rr), indent=2))


if __name__ == "__main__":
    main()
