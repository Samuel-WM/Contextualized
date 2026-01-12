#!/usr/bin/env python3
# Single-node strong-scaling benchmark runner for ContextualizedRegression using synthetic batched data.

import os
import time
import json
import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy

from contextualized.regression import ContextualizedRegression


# Torchrun helpers
def under_torchrun() -> bool:
    e = os.environ
    return ("LOCAL_RANK" in e) or ("RANK" in e) or ("WORLD_SIZE" in e)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_global_zero() -> bool:
    return global_rank() == 0


# Environment defaults
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
            ifaces = [d for d in os.listdir("/sys/class/net") if d not in ("lo", "docker0")]
            cand = next((i for i in ifaces if i.startswith(("ens", "enp", "eno", "eth", "bond", "ib"))), None)
            os.environ["NCCL_SOCKET_IFNAME"] = cand or "^lo,docker0"
        except Exception:
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"

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


# Timing callback
class SteadyStateStepTimer(Callback):
    def __init__(self, warmup_steps: int, measure_steps: int):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self._seen = 0
        self.step_times = []
        self._t0 = None

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        s = self._seen
        if self.warmup_steps <= s < self.warmup_steps + self.measure_steps:
            self._sync()
            self._t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = self._seen
        if self.warmup_steps <= s < self.warmup_steps + self.measure_steps:
            self._sync()
            self.step_times.append(time.time() - (self._t0 or time.time()))
        self._seen += 1

    def measured_wall(self) -> float:
        return float(sum(self.step_times))


def dist_max(value: float) -> float:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([value], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return float(t.item())
    except Exception:
        pass
    return float(value)


# Synthetic batched iterable
class SyntheticBatchStream(IterableDataset):
    def __init__(
        self,
        batch_size: int,
        c_dim: int,
        x_dim: int,
        y_dim: int,
        buffer_batches: int,
        buffer_mult: int,
        seed: int,
        pin: bool,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.c_dim = int(c_dim)
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)

        self.n_batches = int(buffer_batches) * int(buffer_mult)
        if self.n_batches <= 0:
            raise ValueError("buffer_batches * buffer_mult must be >= 1")

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 1000 * global_rank())

        self.C = torch.randn((self.n_batches, self.batch_size, self.c_dim), generator=g, device="cpu", dtype=torch.float32)
        self.X = torch.randn((self.n_batches, self.batch_size, self.x_dim), generator=g, device="cpu", dtype=torch.float32)
        self.Y = torch.randn((self.n_batches, self.batch_size, self.y_dim), generator=g, device="cpu", dtype=torch.float32)

        if pin and torch.cuda.is_available():
            self.C = self.C.pin_memory()
            self.X = self.X.pin_memory()
            self.Y = self.Y.pin_memory()

    def __iter__(self):
        ws = world_size()
        r = global_rank()
        k = 0
        while True:
            b = (k * ws + r) % self.n_batches
            yield {"contexts": self.C[b], "predictors": self.X[b], "outcomes": self.Y[b]}
            k += 1


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    # Accept [B, y, 1] or [B, 1, y] and squeeze the singleton dim
    if t.ndim == 3:
        if t.shape[-1] == 1:
            # Convert [B, y, 1] -> [B, y]
            t = t.squeeze(-1)
        elif t.shape[1] == 1:
            # Convert [B, 1, y] -> [B, y]
            t = t.squeeze(1)
    if t.ndim == 1:
        return t.unsqueeze(-1)
    if t.ndim == 2:
        return t
    raise RuntimeError(f"Expected 1D or 2D tensor (or squeezable 3D), got shape {tuple(t.shape)}")


def _canonicalize_y(y: torch.Tensor, B: int, y_dim: int, name: str) -> torch.Tensor:
    y = _as_2d(y)
    if y.shape == (B, y_dim):
        return y
    if y.shape == (y_dim, B):
        return y.transpose(0, 1)
    if y_dim == 1 and y.shape == (B,):
        return y.view(B, 1)
    raise RuntimeError(f"{name} has incompatible shape {tuple(y.shape)}; expected [{B},{y_dim}] or [{y_dim},{B}].")


def _extract_mu_hat(out: Any) -> torch.Tensor:
    # Prefer mu_hat as y_pred for this benchmark
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        for k in ("mu_hat", "mu", "y_pred", "y_hat", "pred"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise RuntimeError(f"Forward returned dict without mu_hat/y_hat keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        tensors = [t for t in out if torch.is_tensor(t)]
        if len(tensors) >= 2:
            return tensors[1]
        if len(tensors) == 1:
            return tensors[0]
        raise RuntimeError("Forward returned tuple/list with no tensors.")

    raise RuntimeError(f"Unsupported forward output type: {type(out)}")


# Lightning bench module
class BenchModule(pl.LightningModule):
    def __init__(self, inner: ContextualizedRegression, lr: float, y_dim: int):
        super().__init__()
        self.inner = inner
        self.lr = float(lr)
        self.y_dim = int(y_dim)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        device = self.device

        C = batch["contexts"].to(device, non_blocking=True)
        X = batch["predictors"].to(device, non_blocking=True)
        Y = batch["outcomes"].to(device, non_blocking=True)

        B = C.shape[0]
        Y_true = _canonicalize_y(Y, B, self.y_dim, "Y_true")

        # Prefer calling with dict to match internal conventions
        out = self.inner({"contexts": C, "predictors": X, "outcomes": Y_true})

        mu_hat = _extract_mu_hat(out)
        Y_pred = _canonicalize_y(mu_hat, B, self.y_dim, "Y_pred(mu_hat)")

        loss = F.mse_loss(Y_pred, Y_true)
        return loss


# Batch sizing
def resolve_batch_sizes(args, ws: int) -> Tuple[int, int]:
    if args.global_batch_size is None:
        per_gpu = int(args.batch_size)
        return per_gpu, per_gpu * ws
    gbs = int(args.global_batch_size)
    if gbs % ws != 0:
        raise ValueError(f"--global-batch-size {gbs} must be divisible by world_size {ws}")
    return gbs // ws, gbs


# Trainer
def build_trainer(args, timer: SteadyStateStepTimer) -> pl.Trainer:
    use_cuda = torch.cuda.is_available() and (args.run_device != "cpu")

    if use_cuda:
        accelerator = "gpu"

        if under_torchrun():
            devices = 1
        else:
            devices = min(int(args.devices), torch.cuda.device_count())

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

    max_steps = int(args.warmup_steps) + int(args.steps)

    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=map_precision(args.precision),
        max_steps=max_steps,
        max_epochs=10_000,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        callbacks=[timer],
        inference_mode=False,
        enable_model_summary=False,
        accumulate_grad_batches=1,
        limit_val_batches=0,
    )


# Results
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


def save_result(outdir: str, res: Result):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "result.json")
    with open(path, "w") as f:
        json.dump(res.__dict__, f, indent=2)
    return path


# Main bench
def run_bench(args) -> Result:
    ws = world_size() if under_torchrun() else int(args.devices)
    per_gpu_bs, global_bs = resolve_batch_sizes(args, ws)

    pin = args.data_device == "cpu_pinned"

    ds = SyntheticBatchStream(
        batch_size=per_gpu_bs,
        c_dim=args.context_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        buffer_batches=args.buffer_batches,
        buffer_mult=args.buffer_mult,
        seed=args.seed,
        pin=pin,
    )

    dl = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=False)

    inner = ContextualizedRegression(
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

    model = BenchModule(inner=inner, lr=args.lr, y_dim=args.y_dim)

    timer = SteadyStateStepTimer(args.warmup_steps, args.steps)
    trainer = build_trainer(args, timer)

    if is_global_zero():
        buffer_batches_total = int(args.buffer_batches) * int(args.buffer_mult)
        buffer_samples_per_rank = int(per_gpu_bs) * buffer_batches_total
        print(
            "\nConfig:",
            json.dumps(
                {
                    "torchrun": under_torchrun(),
                    "world_size": ws,
                    "local_rank": local_rank(),
                    "batch_size_per_gpu": per_gpu_bs,
                    "global_batch_size": global_bs,
                    "steps_measured": int(args.steps),
                    "steps_warmup": int(args.warmup_steps),
                    "buffer_batches_total": buffer_batches_total,
                    "buffer_samples_per_rank": buffer_samples_per_rank,
                    "buffer_samples_global_approx": buffer_samples_per_rank * int(ws),
                    "run_device": args.run_device,
                    "data_device": args.data_device,
                    "pin_memory": pin,
                    "precision": map_precision(args.precision),
                },
                indent=2,
            ),
        )

    trainer.fit(model, train_dataloaders=dl)

    measured_wall = dist_max(timer.measured_wall())

    measured_steps = int(args.steps)
    samples_total = global_bs * measured_steps
    throughput = samples_total / max(measured_wall, 1e-12)
    per_gpu_thr = throughput / max(ws, 1)

    step_times = timer.step_times[:] if timer.step_times else [float("nan")]
    avg_step = float(np.mean(step_times))
    p95_step = float(np.percentile(step_times, 95)) if len(step_times) > 1 else float("nan")

    return Result(
        world_size=int(ws),
        batch_size_per_gpu=int(per_gpu_bs),
        global_batch_size=int(global_bs),
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(measured_steps),
        measured_wall_s=float(measured_wall),
        throughput_samples_per_s=float(throughput),
        per_gpu_throughput_samples_per_s=float(per_gpu_thr),
        avg_step_s=float(avg_step),
        p95_step_s=float(p95_step),
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup-steps", type=int, default=50)

    ap.add_argument("--batch-size", type=int, default=2048, help="Per-GPU batch size (ignored if --global-batch-size set)")
    ap.add_argument("--global-batch-size", type=int, default=None, help="Fixed global batch for strong scaling")

    ap.add_argument("--precision", type=str, default="bf16")

    ap.add_argument("--context-dim", type=int, default=16)
    ap.add_argument("--x-dim", type=int, default=512)
    ap.add_argument("--y-dim", type=int, default=64)

    ap.add_argument("--encoder-type", type=str, default="mlp")
    ap.add_argument("--num-archetypes", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--buffer-batches", type=int, default=16, help="Buffer depth in batches (per rank)")
    ap.add_argument("--buffer-mult", type=int, default=4, help="Extra multiplier on buffer size (per rank)")

    ap.add_argument("--data-device", choices=["cpu", "cpu_pinned"], default="cpu_pinned")
    ap.add_argument("--run-device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--devices", type=int, default=1, help="Used only when NOT under torchrun")

    ap.add_argument("--ddp-timeout", type=int, default=180)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="bench_out")

    return ap.parse_args()


def main():
    set_env_defaults()
    args = parse_args()

    if args.run_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    res = run_bench(args)

    if is_global_zero():
        path = save_result(args.outdir, res)
        print("\nResult:", json.dumps(res.__dict__, indent=2))
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
