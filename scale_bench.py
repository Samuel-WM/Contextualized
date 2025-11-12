#!/usr/bin/env python3
import os, time, csv, argparse, math, json
from dataclasses import dataclass
from typing import List, Dict
from datetime import timedelta

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

def is_global_zero() -> bool:
    # With torchrun Lightning sets ranks; outside it, fall back to env.
    return int(os.environ.get("RANK", "0")) == 0


# ---------------- env + perf ----------------
def set_env_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Safe NCCL defaults for cloud
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # IB typically unavailable on single Lambda node

    # Pick an interface if not set
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        try:
            ifaces = [d for d in os.listdir("/sys/class/net") if os.path.isdir(f"/sys/class/net/{d}")]
            cand = next((i for i in ifaces if i not in ("lo", "docker0")), None)
            os.environ["NCCL_SOCKET_IFNAME"] = cand or "lo"
        except Exception:
            os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    # Rendezvous only matters for non-torchrun spawn
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(12355 + (os.getpid() % 20000)))

    if is_global_zero():
        keys = ["NCCL_DEBUG","NCCL_IB_DISABLE","NCCL_P2P_DISABLE","NCCL_SOCKET_IFNAME","MASTER_ADDR","MASTER_PORT"]
        print("DDP/NCCL env:", {k: os.environ.get(k) for k in keys})

    # Ampere+ matmul speedups
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def map_precision(p):
    p = (p or "").lower()
    if p in ("bf16", "bfloat16", "bf16-mixed"):
        return "bf16-mixed"
    if p in ("fp16", "16", "16-mixed"):
        return "16-mixed"
    return 32  # full precision


class EpochTimer(Callback):
    def __init__(self):
        self._epoch_start = None
        self.epoch_times = []

    @staticmethod
    def _using_cuda(trainer) -> bool:
        try:
            return trainer.accelerator is not None and "cuda" in str(trainer.accelerator).lower()
        except Exception:
            return torch.cuda.is_available()

    def on_train_epoch_start(self, trainer, pl_module):
        if self._using_cuda(trainer):
            torch.cuda.synchronize()
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._using_cuda(trainer):
            torch.cuda.synchronize()
        self.epoch_times.append(time.time() - self._epoch_start)


# ---------------- synthetic data ----------------
def make_synthetic(n, c_dim, x_dim, y_dim, seed=42):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n, c_dim)).astype(np.float32)
    X = rng.standard_normal((n, x_dim)).astype(np.float32)
    W = rng.standard_normal((y_dim, x_dim)).astype(np.float32)
    MU = rng.standard_normal((y_dim, 1)).astype(np.float32)
    Y = (X @ W.T) + MU.squeeze(-1) + 0.01 * rng.standard_normal((n, y_dim)).astype(np.float32)
    return C, X, Y


# ---------------- model/trainer builders ----------------
def build_model(c_dim, x_dim, y_dim, width, layers, lr):
    model = ContextualizedRegression(
        context_dim=c_dim,
        x_dim=x_dim,
        y_dim=y_dim,
        num_archetypes=8,
        encoder_type="mlp",
        encoder_kwargs={"width": width, "layers": layers, "link_fn": "identity"},
        learning_rate=lr,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    )
    return model


def build_dm(
    C, X, Y,
    train_batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    n = C.shape[0]
    perm = np.random.permutation(n)
    n_train = int(0.9 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    dm = ContextualizedRegressionDataModule(
        C=C, X=X, Y=Y,
        task_type="singletask_multivariate",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=None,
        predict_idx=None,
        train_batch_size=train_batch_size,
        val_batch_size=train_batch_size,
        test_batch_size=train_batch_size,
        predict_batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
        shuffle_train=True,
        shuffle_eval=False,
        dtype=torch.float,
    )
    dm.prepare_data(); dm.setup()
    return dm


def build_trainer(devices, precision, epochs, ddp_timeout_s=120, torchrun_mode=False):
    """
    devices:
      - 0 => cpu
      - 1 => single gpu
      - >1 => multi-gpu (only when NOT torchrun)
    torchrun_mode:
      - True => use DDP (1 GPU per process), devices=1
    """
    timer = EpochTimer()

    if devices == 0:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "gpu"
        if torchrun_mode and world_size() > 1:
            # true DDP under torchrun
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                timeout=timedelta(seconds=ddp_timeout_s),
            )
            devices = 1  # 1 GPU per process
        elif devices == 1:
            strategy = "auto"
        else:
            # multi-gpu without torchrun: fall back to spawn
            strategy = DDPStrategy(
                start_method="spawn",
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
                timeout=timedelta(seconds=ddp_timeout_s),
            )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        log_every_n_steps=50,
        callbacks=[timer],
        inference_mode=False,
        detect_anomaly=False,
    )
    return trainer, timer


# ---------------- benchmark runner ----------------
@dataclass
class BenchCfg:
    label: str
    devices: int      # 0=cpu, >=1 gpus


def run_once(cfg: BenchCfg, C, X, Y, args, torchrun_mode: bool) -> Dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # datamodule
    dm = build_dm(
        C, X, Y,
        train_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(cfg.devices >= 1),
    )
    # model
    model = build_model(args.context_dim, args.x_dim, args.y_dim,
                        args.width, args.layers, args.lr)

    # ---- warm-up on the SAME accelerator config ----
    tiny = max(1024, math.ceil(0.01 * C.shape[0]))
    dm_warm = build_dm(
        C[:tiny], X[:tiny], Y[:tiny],
        train_batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(cfg.devices >= 1),
    )
    warm_trainer, _ = build_trainer(
        devices=cfg.devices if not torchrun_mode else 1,
        precision=map_precision(args.precision),
        epochs=1,
        ddp_timeout_s=args.ddp_timeout,
        torchrun_mode=torchrun_mode,
    )
    warm_trainer.fit(model, train_dataloaders=dm_warm.train_dataloader())

    # ---- main timed run ----
    trainer, timer = build_trainer(
        devices=cfg.devices if not torchrun_mode else 1,
        precision=map_precision(args.precision),
        epochs=args.epochs,
        ddp_timeout_s=args.ddp_timeout,
        torchrun_mode=torchrun_mode,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    trainer.fit(model, train_dataloaders=dm.train_dataloader())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0

    # metrics (use actual train size, not full N)
    train_samples = len(dm.train_dataloader().dataset)
    samples_total = train_samples * args.epochs
    throughput = samples_total / max(wall, 1e-9)
    per_device = (throughput / (world_size() if torchrun_mode and cfg.devices >= 1 else max(cfg.devices, 1)))
    epoch_times = timer.epoch_times[:]  # seconds per epoch

    res = dict(
        label=cfg.label,
        devices=(world_size() if torchrun_mode else cfg.devices),
        wall_seconds=wall,
        samples_total=int(samples_total),
        throughput_samples_per_s=throughput,
        per_device_throughput=per_device,
        steps_per_epoch=math.ceil(train_samples / args.batch_size),
        samples_per_epoch=int(train_samples),
        epoch_times=epoch_times,
    )
    if is_global_zero():
        print(json.dumps({
            "label": res["label"],
            "devices": res["devices"],
            "wall_s": round(res["wall_seconds"], 3),
            "throughput_sps": round(res["throughput_samples_per_s"], 2),
            "per_device_sps": round(res["per_device_throughput"], 2),
            "avg_epoch_s": round(float(np.mean(res["epoch_times"])) if res["epoch_times"] else float("nan"), 3)
        }, indent=2))
    return res


def save_csv(rows: List[Dict], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "scale_results.csv")
    fields = ["label","devices","wall_seconds","samples_total",
              "throughput_samples_per_s","per_device_throughput",
              "steps_per_epoch","samples_per_epoch","epoch_times"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r2 = r.copy()
            r2["epoch_times"] = ";".join(f"{x:.6f}" for x in r["epoch_times"])
            w.writerow(r2)
    return path


def plot_curves(rows: List[Dict], outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(outdir, exist_ok=True)
    labels = [r["label"] for r in rows]
    devs = [r["devices"] for r in rows]
    thr = [r["throughput_samples_per_s"] for r in rows]
    wall = [r["wall_seconds"] for r in rows]
    avg_epoch = [np.mean(r["epoch_times"]) if r["epoch_times"] else float("nan") for r in rows]

    # Throughput
    plt.figure()
    plt.plot(devs, thr, marker="o")
    plt.xticks(devs, labels, rotation=30, ha="right")
    plt.xlabel("Devices")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Throughput vs Devices")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "throughput_vs_devices.png"))
    plt.close()

    # Wall time
    plt.figure()
    plt.plot(devs, wall, marker="o")
    plt.xticks(devs, labels, rotation=30, ha="right")
    plt.xlabel("Devices")
    plt.ylabel("Total Wall Time (s)")
    plt.title("Wall Time vs Devices")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walltime_vs_devices.png"))
    plt.close()

    # Avg epoch time
    plt.figure()
    plt.plot(devs, avg_epoch, marker="o")
    plt.xticks(devs, labels, rotation=30, ha="right")
    plt.xlabel("Devices")
    plt.ylabel("Avg Train Epoch Time (s)")
    plt.title("Epoch Time vs Devices")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "epoch_time_vs_devices.png"))
    plt.close()


# ---------------- main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)  # PER GPU
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--precision", type=str, default="bf16")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--context-dim", type=int, default=16)
    ap.add_argument("--x-dim", type=int, default=512)
    ap.add_argument("--y-dim", type=int, default=64)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="bench_out")
    ap.add_argument("--ddp-timeout", type=int, default=180)
    ap.add_argument("--max-gpus", type=int, default=4)
    return ap.parse_args()


def main():
    set_env_defaults()
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # opt for fixed shapes

    # data once
    C, X, Y = make_synthetic(args.n, args.context_dim, args.x_dim, args.y_dim)

    results = []
    torchrun_mode = under_torchrun()

    if torchrun_mode:
        # Run exactly one multi-GPU config under torchrun: devices = WORLD_SIZE
        cfg = BenchCfg(label=f"gpu-{world_size()}", devices=1)  # 1 per proc; true DDP
        if is_global_zero():
            print(f"\n=== Running {cfg.label} (torchrun, {world_size()} processes) ===")
        res = run_once(cfg, C, X, Y, args, torchrun_mode=True)
        results.append(res)
    else:
        # Standalone: run cpu + 1..k GPUs (ddp-spawn for k>1)
        gpus = torch.cuda.device_count()
        dev_list = [BenchCfg("cpu", 0)]
        for k in range(1, min(args.max_gpus, gpus) + 1):
            dev_list.append(BenchCfg(f"gpu-{k}", k))
        for cfg in dev_list:
            if is_global_zero():
                print(f"\n=== Running {cfg.label} ===")
            res = run_once(cfg, C, X, Y, args, torchrun_mode=False)
            results.append(res)

    if is_global_zero():
        csv_path = save_csv(results, args.outdir)
        plot_curves(results, args.outdir)
        print(f"\nSaved CSV → {csv_path}")
        print(f"Saved plots → {args.outdir}/throughput_vs_devices.png, "
              f"walltime_vs_devices.png, epoch_time_vs_devices.png")


if __name__ == "__main__":
    main()
