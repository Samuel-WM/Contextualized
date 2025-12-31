#!/usr/bin/env python3
"""
scale_bench.py

A single-node, torchrun-friendly DDP scaling benchmark for ContextualizedRegression.

Design goals (to reveal true scaling):
  - Fixed number of optimizer steps (not epochs) so each run does identical work.
  - Optional GPU-resident synthetic dataset to remove CPU dataloading/transfer bottlenecks.
  - Measures only the *steady-state* region (warmup steps excluded).
  - Uses Lightning DDP under torchrun correctly (devices=1 per process).

------------------------------------------------------------
Quick start (single node, 1..4 GPUs)
------------------------------------------------------------

# 0) See NICs (optional)
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

# CUDA allocator tweak (fine to keep)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2) Kill any stragglers (optional)
pkill -f scale_bench.py || true
pkill -f torchrun || true

# 3) Runs (IMPORTANT: --batch-size is PER GPU)
# Suggested defaults: steps=400 warmup=50 (steady state measured steps=400)

torchrun --standalone --nproc_per_node=1 scale_bench.py \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 --y-dim 64 \
  --width 1024 --layers 4 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/gpu1

torchrun --standalone --nproc_per_node=2 scale_bench.py \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 --y-dim 64 \
  --width 1024 --layers 4 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/gpu2

torchrun --standalone --nproc_per_node=3 scale_bench.py \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 --y-dim 64 \
  --width 1024 --layers 4 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/gpu3

torchrun --standalone --nproc_per_node=4 scale_bench.py \
  --steps 400 --warmup-steps 50 \
  --batch-size 2048 --precision bf16 \
  --context-dim 16 --x-dim 512 --y-dim 64 \
  --width 1024 --layers 4 \
  --buffer-batches 32 --data-device auto \
  --outdir bench_out/gpu4

Notes:
  - If scaling is still poor with this benchmark, it is very likely a *real* bottleneck
    (GPU interconnect/topology, NCCL config, too-small batch, CPU frequency limits, etc.),
    not a dataloader artifact.
"""

import os
import time
import json
import math
import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy

# ---- your package pieces ----
from contextualized.regression import ContextualizedRegression
from contextualized.regression.datamodules import ContextualizedRegressionDataModule


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

    # TF32 / matmul speedups (safe for benchmarking throughput)
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
        # Ensures each rank uses its intended GPU even if something upstream is odd.
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
            t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu")
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
    y_dim: int,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Generates a fixed buffer of synthetic data.

    IMPORTANT: This runs once before timing begins. Keep n reasonable.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 1000 * global_rank())

    C = torch.randn((n, c_dim), generator=g, device=device, dtype=torch.float32)
    X = torch.randn((n, x_dim), generator=g, device=device, dtype=torch.float32)
    Y = torch.randn((n, y_dim), generator=g, device=device, dtype=torch.float32)
    return {"C": C, "X": X, "Y": Y}


# ---------------- model/trainer/datamodule ----------------
def build_model(args) -> ContextualizedRegression:
    # Uses your current link_fn handling (string keys are valid).
    return ContextualizedRegression(
        context_dim=args.context_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        num_archetypes=args.num_archetypes,
        encoder_type=args.encoder_type,
        encoder_kwargs={"width": args.width, "layers": args.layers, "link_fn": "identity"},
        learning_rate=args.lr,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    )


def build_dm(args, C, X, Y) -> ContextualizedRegressionDataModule:
    n = int(C.shape[0])
    # Simple split; validation never runs in this benchmark (we pass only train_dataloader).
    n_train = int(0.95 * n)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n, dtype=np.int64)

    dm = ContextualizedRegressionDataModule(
        C=C,
        X=X,
        Y=Y,
        task_type="singletask_multivariate",
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
        shuffle_train=False,  # let Lightning/DDP sampler handle partitioning; shuffle not needed for perf
        shuffle_eval=False,
        dtype=torch.float,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def build_trainer(args, timer: SteadyStateStepTimer) -> pl.Trainer:
    if torch.cuda.is_available():
        accelerator = "gpu"
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

    # We benchmark *steps*, not epochs.
    max_steps = args.warmup_steps + args.steps

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=map_precision(args.precision),
        max_steps=max_steps,
        max_epochs=10_000,  # irrelevant when max_steps is set
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        callbacks=[timer],
        inference_mode=False,
        detect_anomaly=False,
        enable_model_summary=False,
        use_distributed_sampler=True,
        accumulate_grad_batches=1,
        limit_val_batches=0,
    )
    return trainer


# ---------------- benchmark runner ----------------
@dataclass
class Result:
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


def run_bench(args) -> Result:
    ws = world_size() if under_torchrun() else int(args.devices)
    dev = torch.device("cuda", local_rank()) if (args.data_device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    # If auto: keep data on GPU when available (this removes input bottlenecks).
    if args.data_device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda", local_rank())
        else:
            dev = torch.device("cpu")

    # Dataloader workers cannot safely handle CUDA tensors.
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
        y_dim=args.y_dim,
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
                    "torchrun": under_torchrun(),
                    "world_size": ws,
                    "local_rank": local_rank(),
                    "batch_size_per_gpu": args.batch_size,
                    "global_batch_size": args.batch_size * ws,
                    "steps_measured": args.steps,
                    "steps_warmup": args.warmup_steps,
                    "buffer_samples": n,
                    "data_device": str(dev),
                    "precision": map_precision(args.precision),
                },
                indent=2,
            ),
        )

    trainer.fit(model, train_dataloaders=dm.train_dataloader())

    measured_wall = timer.measured_wall_time()
    measured_wall = dist_max(measured_wall)  # slowest rank dictates wall time

    measured_steps = int(args.steps)
    global_batch = int(args.batch_size * ws)
    samples_total = global_batch * measured_steps
    throughput = samples_total / max(measured_wall, 1e-12)
    per_gpu = throughput / max(ws, 1)

    step_times = timer.step_times[:] if timer.step_times else [float("nan")]
    avg_step = float(np.mean(step_times))
    p95_step = float(np.percentile(step_times, 95)) if len(step_times) > 1 else float("nan")

    return Result(
        world_size=ws,
        batch_size_per_gpu=int(args.batch_size),
        global_batch_size=int(global_batch),
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(measured_steps),
        measured_wall_s=float(measured_wall),
        throughput_samples_per_s=float(throughput),
        per_gpu_throughput_samples_per_s=float(per_gpu),
        avg_step_s=float(avg_step),
        p95_step_s=float(p95_step),
    )


def save_result(outdir: str, res: Result):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "result.json")
    with open(path, "w") as f:
        json.dump(res.__dict__, f, indent=2)
    return path


# ---------------- main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="Measured optimizer steps")
    ap.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps excluded from timing")

    ap.add_argument("--batch-size", type=int, default=2048, help="Per-GPU batch size")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--precision", type=str, default="bf16")

    ap.add_argument("--context-dim", type=int, default=16)
    ap.add_argument("--x-dim", type=int, default=512)
    ap.add_argument("--y-dim", type=int, default=64)

    ap.add_argument("--encoder-type", type=str, default="mlp")
    ap.add_argument("--num-archetypes", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--buffer-batches", type=int, default=32, help="Dataset buffer size = batch_size * buffer_batches")
    ap.add_argument("--data-device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--devices", type=int, default=1, help="Only used when NOT under torchrun")

    ap.add_argument("--ddp-timeout", type=int, default=180)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="bench_out")

    return ap.parse_args()


def main():
    set_env_defaults()
    args = parse_args()

    if args.data_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure no accidental CUDA use

    res = run_bench(args)

    if is_global_zero():
        path = save_result(args.outdir, res)
        print(
            "\nResult:",
            json.dumps(res.__dict__, indent=2),
        )
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
