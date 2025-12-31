#!/usr/bin/env python3
"""
scale_bench_networks.py

A torchrun-friendly DDP scaling benchmark for Contextualized *Networks* lightning modules
(e.g., ContextualizedCorrelation, ContextualizedMarkovGraph, NOTMAD).

Design goals (to reveal true scaling):
  - Fixed number of optimizer steps (not epochs) so each run does identical work.
  - Optional GPU-resident synthetic dataset to remove CPU dataloading/transfer bottlenecks.
  - Measures only the *steady-state* region (warmup steps excluded).
  - Uses Lightning DDP under torchrun correctly (devices=1 per process).
  - No validation, no logging, no checkpoints.

------------------------------------------------------------
Quick start (single node, 1..4 GPUs)
------------------------------------------------------------

# 0) NICs (optional)
ls -1 /sys/class/net
ip -o link show | awk -F': ' '{print NR-1": "$2}'

# 1) Minimal, safe NCCL/torch env (no hard-coded eth0):
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$(ls /sys/class/net | grep -E '^(ens|enp|eno|eth|bond|ib)' | head -n1)
[ -z "$NCCL_SOCKET_IFNAME" ] && export NCCL_SOCKET_IFNAME="^lo,docker0"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2) Kill any stragglers (optional)
pkill -f scale_bench_networks.py || true
pkill -f torchrun || true

# 3) Runs (IMPORTANT: --batch-size is PER GPU)

# Correlation networks
torchrun --standalone --nproc_per_node=1 scale_bench_networks.py \
  --network correlation \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 \
  --encoder-type mlp --width 1024 --layers 4 \
  --num-archetypes 8 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/corr_gpu1

torchrun --standalone --nproc_per_node=2 scale_bench_networks.py \
  --network correlation \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 \
  --encoder-type mlp --width 1024 --layers 4 \
  --num-archetypes 8 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/corr_gpu2

# Markov networks (precision matrices)
torchrun --standalone --nproc_per_node=4 scale_bench_networks.py \
  --network markov \
  --steps 400 --warmup-steps 50 \
  --batch-size 1024 --precision bf16 \
  --context-dim 16 --x-dim 256 \
  --encoder-type mlp --width 512 --layers 3 \
  --num-archetypes 8 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/markov_gpu4

Notes:
  - If scaling is poor with --data-device=cuda (or auto on GPU), the bottleneck is
    likely *real* (NCCL/topology/comm, too-small batch, CPU freq limits, etc.).
  - Multi-node: remove --standalone and use --nnodes/--node_rank with a shared rdzv endpoint.
"""

import os
import time
import json
import math
import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy

# ---- your package pieces ----
from contextualized.regression.datamodules import ContextualizedRegressionDataModule
from contextualized.regression.lightning_modules import (
    ContextualizedCorrelation,
    ContextualizedMarkovGraph,
)
from contextualized.dags.lightning_modules import NOTMAD


# ---------------- launcher/cluster helpers ----------------
def under_torchrun() -> bool:
    e = os.environ
    return ("LOCAL_RANK" in e) or ("RANK" in e) or ("WORLD_SIZE" in e)


def world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def global_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0


def is_global_zero() -> bool:
    return global_rank() == 0


# ---------------- env + perf ----------------
def set_env_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        try:
            ifaces = [
                d
                for d in os.listdir("/sys/class/net")
                if os.path.isdir(f"/sys/class/net/{d}")
            ]
            cand = next((i for i in ifaces if i not in ("lo", "docker0")), None)
            os.environ["NCCL_SOCKET_IFNAME"] = cand or "^lo,docker0"
        except Exception:
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"

    # TF32 / matmul speedups (safe for throughput benchmarking)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    if under_torchrun() and torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank())
        except Exception:
            pass

    if is_global_zero():
        keys = [
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        ]
        print("DDP/NCCL env:", {k: os.environ.get(k) for k in keys})
        if torch.cuda.is_available():
            print(
                "CUDA:",
                {
                    "torch": torch.__version__,
                    "lightning": pl.__version__,
                    "gpus_visible": torch.cuda.device_count(),
                },
            )


def map_precision(p: str):
    p = (p or "").lower()
    if p in ("bf16", "bfloat16", "bf16-mixed"):
        return "bf16-mixed"
    if p in ("fp16", "16", "16-mixed"):
        return "16-mixed"
    return 32


# ---------------- timing ----------------
class SteadyStateStepTimer(Callback):
    """
    Times optimizer steps in a steady-state window:
      - ignore first warmup_steps
      - measure next measure_steps
    Assumes accumulate_grad_batches == 1.
    """

    def __init__(self, warmup_steps: int, measure_steps: int):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self._seen_steps = 0
        self.step_times = []
        self._step_start_t = None

    @staticmethod
    def _sync_if_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        s = self._seen_steps
        if self.warmup_steps <= s < (self.warmup_steps + self.measure_steps):
            self._sync_if_cuda()
            self._step_start_t = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = self._seen_steps
        if self.warmup_steps <= s < (self.warmup_steps + self.measure_steps):
            self._sync_if_cuda()
            dt = time.time() - (self._step_start_t or time.time())
            self.step_times.append(dt)
        self._seen_steps += 1

    def measured_wall_time(self) -> float:
        return float(sum(self.step_times))


def dist_max(value: float) -> float:
    """
    Returns max(value across ranks) if distributed is initialized; else returns value.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(
                [value], device="cuda" if torch.cuda.is_available() else "cpu"
            )
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return float(t.item())
    except Exception:
        pass
    return float(value)


# ---------------- synthetic data ----------------
def make_synthetic_tensors(
    n: int,
    c_dim: int,
    x_dim: int,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Builds a fixed synthetic buffer (not timed). Shapes:
      C: (n, c_dim)
      X: (n, x_dim)
      Y: (n, x_dim)   # for networks we follow the wrapper convention (univariate task uses y_dim=x_dim)
    """
    g = torch.Generator(device=device)
    # Per-rank seed to avoid identical data, while keeping identical shapes across ranks.
    g.manual_seed(int(seed) + 1000 * global_rank())

    C = torch.randn((n, c_dim), generator=g, device=device, dtype=torch.float32)
    X = torch.randn((n, x_dim), generator=g, device=device, dtype=torch.float32)
    Y = torch.randn((n, x_dim), generator=g, device=device, dtype=torch.float32)
    return {"C": C, "X": X, "Y": Y}


# ---------------- model/datamodule/trainer ----------------
def build_model(args):
    """
    Robustly instantiate the selected network LightningModule.

    We pass a *superset* of kwargs and filter by the model's __init__ signature to
    remain compatible with small constructor differences across implementations.
    """
    import inspect

    if args.network == "correlation":
        model_cls = ContextualizedCorrelation
    elif args.network == "markov":
        model_cls = ContextualizedMarkovGraph
    elif args.network == "bayesian":
        model_cls = NOTMAD
    else:
        raise ValueError(f"Unknown --network {args.network}")

    encoder_kwargs = {"width": args.width, "layers": args.layers, "link_fn": "identity"}

    # Common superset
    kw = dict(
        context_dim=args.context_dim,
        x_dim=args.x_dim,
        y_dim=args.x_dim,  # networks wrapper convention
        univariate=True,
        num_archetypes=args.num_archetypes,
        encoder_type=args.encoder_type,
        encoder_kwargs=encoder_kwargs,
        learning_rate=args.lr,
        link_fn="identity",
        fit_intercept=True,
        loss_fn="mse",
        model_regularizer="none",
    )

    # NOTMAD-specific defaults (safe baseline; tune as needed)
    if args.network == "bayesian":
        kw.update(
            archetype_loss_params=dict(
                l1=0.0,
                dag=dict(loss_type="notears", params=dict(alpha=1.0, rho=1.0, s=1.0, tol=1e-8)),
                init_mat=None,
                num_factors=0,
                factor_mat_l1=0.0,
                num_archetypes=max(1, int(args.num_archetypes)),
            ),
            sample_specific_loss_params=dict(
                l1=0.0,
                dag=dict(loss_type="notears", params=dict(alpha=1.0, rho=1.0, s=1.0, tol=1e-8)),
            ),
            opt_params=dict(
                learning_rate=args.lr,
                step=50,
            ),
        )

    sig = inspect.signature(model_cls.__init__)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return model_cls(**kw)

    filtered = {k: v for k, v in kw.items() if k in sig.parameters}
    # Basic required-arg check (only for explicit signatures)
    required = [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in filtered]
    if missing:
        raise TypeError(
            f"{model_cls.__name__}.__init__ missing required args {missing}. "
            f"Accepted params in script: {sorted(filtered.keys())}. "
            f"Signature: {sig}"
        )
    return model_cls(**filtered)


def build_dm(args, C, X, Y) -> ContextualizedRegressionDataModule:
    """
    Uses the same DataModule family as the wrapper (consistent batch structure).
    IMPORTANT: If data lives on CUDA, we force num_workers=0.
    """
    n = int(C.shape[0])
    n_train = max(1, int(0.98 * n))
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n, dtype=np.int64)

    task_type = args.task_type
    if task_type is None:
        # Networks wrappers use the univariate convention.
        task_type = "singletask_univariate"

    dm = ContextualizedRegressionDataModule(
        C=C,
        X=X,
        Y=Y,
        task_type=task_type,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=None,
        predict_idx=None,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        predict_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.num_workers > 0),
        drop_last=True,
        shuffle_train=False,
        shuffle_eval=False,
        dtype=torch.float,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def build_trainer(args, timer: SteadyStateStepTimer) -> pl.Trainer:
    if torch.cuda.is_available():
        accelerator = "gpu"
        # Under torchrun: each process uses exactly 1 device
        devices = 1 if under_torchrun() else min(args.devices, torch.cuda.device_count())
        strategy = (
            DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                timeout=timedelta(seconds=args.ddp_timeout),
            )
            if (under_torchrun() or devices > 1)
            else "auto"
        )
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    max_steps = int(args.warmup_steps + args.steps)

    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=map_precision(args.precision),
        max_steps=max_steps,
        max_epochs=10_000,  # irrelevant when max_steps is set
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        callbacks=[timer],
        inference_mode=False,
        detect_anomaly=False,
        accumulate_grad_batches=1,
        limit_val_batches=0,          # no validation
        use_distributed_sampler=False # IMPORTANT: our synthetic buffer is already identical-sized per rank
    )


# ---------------- benchmark runner ----------------
@dataclass
class Result:
    network: str
    world_size: int
    batch_size_per_gpu: int
    global_batch_size: int
    warmup_steps: int
    measured_steps: int
    measured_wall_s: float
    throughput_samples_per_s: float
    per_gpu_throughput_samples_per_s: float
    avg_step_s: float
    p95_step_s: float
    data_device: str


def run_bench(args) -> Result:
    ws = world_size() if under_torchrun() else int(args.devices)

    # Resolve data device
    if args.data_device == "cpu":
        dev = torch.device("cpu")
    elif args.data_device == "cuda":
        dev = torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    else:  # auto
        dev = torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")

    # Dataloader workers cannot safely handle CUDA tensors
    if dev.type == "cuda" and args.num_workers != 0:
        if is_global_zero():
            print("NOTE: forcing --num-workers=0 because data-device is CUDA.")
        args.num_workers = 0

    # Build fixed synthetic buffer (not timed)
    n = int(args.batch_size * args.buffer_batches)
    tensors = make_synthetic_tensors(
        n=n,
        c_dim=args.context_dim,
        x_dim=args.x_dim,
        device=dev,
        seed=args.seed,
    )

    dm = build_dm(args, tensors["C"], tensors["X"], tensors["Y"])
    model = build_model(args)

    timer = SteadyStateStepTimer(args.warmup_steps, args.steps)
    trainer = build_trainer(args, timer)

    if is_global_zero():
        print(
            "\nConfig:",
            json.dumps(
                {
                    "network": args.network,
                    "torchrun": under_torchrun(),
                    "world_size": ws,
                    "local_rank": local_rank(),
                    "batch_size_per_gpu": args.batch_size,
                    "global_batch_size": args.batch_size * ws,
                    "steps_measured": args.steps,
                    "steps_warmup": args.warmup_steps,
                    "buffer_samples_per_rank": n,
                    "data_device": str(dev),
                    "precision": map_precision(args.precision),
                    "task_type": args.task_type or "singletask_univariate",
                },
                indent=2,
            ),
        )

    trainer.fit(model, train_dataloaders=dm.train_dataloader())

    measured_wall = timer.measured_wall_time()
    measured_wall = dist_max(measured_wall)  # slowest rank dictates

    measured_steps = int(args.steps)
    global_batch = int(args.batch_size * ws)
    samples_total = global_batch * measured_steps
    throughput = samples_total / max(measured_wall, 1e-12)
    per_gpu = throughput / max(ws, 1)

    step_times = timer.step_times[:] if timer.step_times else [float("nan")]
    avg_step = float(np.mean(step_times))
    p95_step = float(np.percentile(step_times, 95)) if len(step_times) > 1 else float("nan")

    return Result(
        network=str(args.network),
        world_size=int(ws),
        batch_size_per_gpu=int(args.batch_size),
        global_batch_size=int(global_batch),
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(measured_steps),
        measured_wall_s=float(measured_wall),
        throughput_samples_per_s=float(throughput),
        per_gpu_throughput_samples_per_s=float(per_gpu),
        avg_step_s=float(avg_step),
        p95_step_s=float(p95_step),
        data_device=str(dev),
    )


def save_result(outdir: str, res: Result) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "result.json")
    with open(path, "w") as f:
        json.dump(res.__dict__, f, indent=2)
    return path


# ---------------- main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", type=str, choices=["correlation", "markov", "bayesian"], default="correlation")

    ap.add_argument("--steps", type=int, default=400, help="Measured optimizer steps")
    ap.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps excluded from timing")

    ap.add_argument("--batch-size", type=int, default=2048, help="Per-GPU batch size")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--precision", type=str, default="bf16")

    ap.add_argument("--context-dim", type=int, default=16)
    ap.add_argument("--x-dim", type=int, default=512)

    ap.add_argument("--encoder-type", type=str, default="mlp")
    ap.add_argument("--num-archetypes", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--buffer-batches", type=int, default=32,
                    help="Per-rank buffer size = batch_size * buffer_batches")
    ap.add_argument("--data-device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    ap.add_argument("--task-type", type=str, default=None,
                    help="Override task_type if needed (default: singletask_univariate)")

    ap.add_argument("--devices", type=int, default=1, help="Only used when NOT under torchrun")
    ap.add_argument("--ddp-timeout", type=int, default=180)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="bench_out")
    return ap.parse_args()


def main():
    set_env_defaults()
    args = parse_args()

    if args.data_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # prevent accidental CUDA use

    res = run_bench(args)

    if is_global_zero():
        path = save_result(args.outdir, res)
        print("\nResult:", json.dumps(res.__dict__, indent=2))
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
