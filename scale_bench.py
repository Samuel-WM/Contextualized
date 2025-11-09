#!/usr/bin/env python3
"""
Sweep benchmark for Contextualized-ML scaling.

Runs: CPU, 1-GPU, 2-GPU, 3-GPU, 4-GPU (skips if not available).
Outputs:
  - bench_out/scale_results.csv
  - bench_out/throughput_vs_devices.png
  - bench_out/walltime_vs_devices.png
  - bench_out/epoch_time_vs_devices.png
"""

import os, time, argparse, csv, math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from contextualized.regression.datamodules import ContextualizedRegressionDataModule
from contextualized.regression import ContextualizedRegression
from contextualized.regression.trainers import RegressionTrainer, make_trainer_with_env
from pytorch_lightning.callbacks import Callback

# ----------------- utils -----------------
def env_defaults():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

def make_synth(n, c_dim, x_dim, y_dim, seed=1337):
    rng = np.random.RandomState(seed)
    C = rng.randn(n, c_dim).astype("float32")
    X = rng.randn(n, x_dim).astype("float32")
    W = rng.randn(y_dim, x_dim).astype("float32")
    b = rng.randn(y_dim, 1).astype("float32")
    Y = (X @ W.T + b.squeeze(-1) + 0.05 * rng.randn(n, y_dim)).astype("float32")
    return C, X, Y

class TimingCallback(Callback):
    """Collect per-epoch timings and global wall time."""
    def __init__(self):
        self.epoch_times = []
        self._t_epoch = None
        self._t0 = None
        self._t1 = None

    def on_fit_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_fit_end(self, trainer, pl_module):
        self._t1 = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        t = time.perf_counter()
        if self._t_epoch is not None:
            self.epoch_times.append(t - self._t_epoch)
        self._t_epoch = None

    @property
    def wall_time(self):
        if self._t0 is None or self._t1 is None:
            return None
        return self._t1 - self._t0


def run_one(cfg, data, args):
    """
    cfg: dict with keys:
      - label: str (e.g., "cpu", "gpu-1", "gpu-2", ...)
      - accelerator: "cpu" or None
      - devices: "auto" or int
      - strategy: "auto" or "ddp"
    """
    C, X, Y = data
    # datamodule (map-style -> PL autoshard)
    pin_mem = (cfg["accelerator"] != "cpu")
    dm = ContextualizedRegressionDataModule(
        C=C, X=X, Y=Y,
        task_type="singletask_multivariate",
        train_idx=None, val_idx=None, test_idx=None, predict_idx=None,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        predict_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        drop_last=True,
        shuffle_train=True,
        shuffle_eval=False,
        dtype=torch.float,
    )

    model_kwargs = dict(
        context_dim=args.context_dim,
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        num_archetypes=args.num_archetypes,
        encoder_type=args.encoder_type,
        encoder_kwargs={"width": args.width, "layers": args.layers, "link_fn": "identity"},
        learning_rate=args.lr,
        metamodel_type="subtype",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    )
    model = ContextualizedRegression(**model_kwargs)

    # precision
    prec_map = {"32":32, "64":64, "16":"16-mixed", "bf16":"bf16-mixed"}
    precision = prec_map[args.precision]

    timing_cb = TimingCallback()
    trainer = make_trainer_with_env(
        trainer_cls=RegressionTrainer,
        max_epochs=args.epochs,
        accelerator=cfg["accelerator"],
        devices=cfg["devices"],
        strategy=cfg["strategy"],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        precision=precision,
        limit_val_batches=0,
        limit_train_batches=1.0,
        callbacks=[timing_cb],
    )

    # Warmup 1 epoch for stable timings (optional)
    warm_model = ContextualizedRegression(**model_kwargs)
    trainer.fit(warm_model, train_dataloaders=dm.train_dataloader())

    # Timed run
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    trainer.fit(model, train_dataloaders=dm.train_dataloader())
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # metrics
    # Compute seen samples per epoch (drop_last=True)
    steps_per_epoch = math.ceil(len(C) / args.batch_size)
    seen_per_epoch = steps_per_epoch * args.batch_size
    total_seen = seen_per_epoch * args.epochs
    wall = timing_cb.wall_time
    throughput = total_seen / wall if wall and wall > 0 else float("nan")
    world_size = trainer.num_devices if hasattr(trainer, "num_devices") else 1

    return {
        "label": cfg["label"],
        "world_size": world_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "samples_per_epoch": seen_per_epoch,
        "wall_time_s": wall,
        "throughput_samples_per_s": throughput,
        "per_gpu_throughput": throughput / max(1, world_size),
        "epoch_times_s": timing_cb.epoch_times,
    }

def plot_one(x, y, xlabel, ylabel, outpng):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    env_defaults()

    ap = argparse.ArgumentParser()
    # data/model
    ap.add_argument("--n-samples", type=int, default=300_000)
    ap.add_argument("--context-dim", type=int, default=32)
    ap.add_argument("--x-dim", type=int, default=256)
    ap.add_argument("--y-dim", type=int, default=64)
    ap.add_argument("--encoder-type", type=str, default="mlp", choices=["mlp","ngam","linear"])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--num-archetypes", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=3)
    # dataloader
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--persistent-workers", action="store_true", default=True)
    # precision
    ap.add_argument("--precision", type=str, default="bf16", choices=["32","16","bf16","64"])
    # output
    ap.add_argument("--outdir", type=str, default="bench_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # synth data
    data = make_synth(args.n_samples, args.context_dim, args.x_dim, args.y_dim)

    # decide available gpu configs
    n_gpus = torch.cuda.device_count()
    configs = [{"label":"cpu", "accelerator":"cpu", "devices=ignored":1, "devices":1, "strategy":"auto"}]
    for k in [1,2,3,4]:
        if n_gpus >= k:
            configs.append({"label":f"gpu-{k}", "accelerator":None, "devices":k, "strategy":"ddp"})

    results = []
    for cfg in configs:
        print(f"\n=== Running {cfg['label']} ===")
        res = run_one(cfg, data, args)
        for k,v in res.items():
            if k != "epoch_times_s":
                print(f"{k}: {v}")
        results.append(res)

    # write CSV
    csv_path = outdir / "scale_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label","world_size","batch_size","num_workers","precision","epochs",
            "steps_per_epoch","samples_per_epoch","wall_time_s","throughput_samples_per_s","per_gpu_throughput","epoch_times_s"
        ])
        for r in results:
            w.writerow([
                r["label"], r["world_size"], r["batch_size"], r["num_workers"], r["precision"], r["epochs"],
                r["steps_per_epoch"], r["samples_per_epoch"], f"{r['wall_time_s']:.6f}",
                f"{r['throughput_samples_per_s']:.3f}", f"{r['per_gpu_throughput']:.3f}",
                ";".join(f"{et:.6f}" for et in r["epoch_times_s"])
            ])
    print(f"\n[Saved] {csv_path}")

    # plots
    xs = [ (0 if r["label"]=="cpu" else r["world_size"]) for r in results ]
    labels = [r["label"] for r in results]

    # Throughput vs devices (GPU counts; CPU plotted at 0)
    plot_one(xs, [r["throughput_samples_per_s"] for r in results],
             "Devices (0=CPU)", "Throughput (samples/s)", str(outdir / "throughput_vs_devices.png"))
    # Wall time vs devices
    plot_one(xs, [r["wall_time_s"] for r in results],
             "Devices (0=CPU)", "Wall time (s)", str(outdir / "walltime_vs_devices.png"))
    # Mean epoch time vs devices
    plot_one(xs, [float(np.mean(r["epoch_times_s"])) for r in results],
             "Devices (0=CPU)", "Mean epoch time (s)", str(outdir / "epoch_time_vs_devices.png"))

    print(f"[Saved] {outdir/'throughput_vs_devices.png'}")
    print(f"[Saved] {outdir/'walltime_vs_devices.png'}")
    print(f"[Saved] {outdir/'epoch_time_vs_devices.png'}")

if __name__ == "__main__":
    main()
