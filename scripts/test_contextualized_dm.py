# scripts/test_contextualized_dm.py
"""
Smoke-test your ContextualizedRegressionDataModule with synthetic data.

Examples:
  # Single-process sanity check
  python scripts/test_contextualized_dm.py --task-type singletask_multivariate --peek

  # CPU DDP on Windows (Git Bash or PowerShell)
  python scripts/test_contextualized_dm.py --task-type singletask_multivariate --devices 2 --peek
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy

# --- Make repo root importable if running from source tree ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from contextualized.regression.datamodules import ContextualizedRegressionDataModule

# ---- Candidate key names in your batch dict ----
CTX_CANDIDATES = ("contexts", "context", "ctx", "C", "c")
X_CANDIDATES   = ("predictors", "X", "features", "x", "inputs", "data")
Y_CANDIDATES   = ("outcomes", "Y", "targets", "y", "labels")


def pick_first_key(d: Dict[str, torch.Tensor], candidates) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


# ---------------------------
# Synthetic (C, X, Y)
# ---------------------------
def make_synthetic(
    n: int,
    c_dim: int,
    x_dim: int,
    y_dim: int,
    seed: int = 1234,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n, c_dim, generator=g)
    X = torch.randn(n, x_dim, generator=g)
    W = torch.randn(x_dim, y_dim, generator=g) / (x_dim ** 0.5)
    Y = X @ W + 0.05 * torch.randn(n, y_dim, generator=g)
    return C, X, Y


def make_indices(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randperm(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ---------------------------
# Tiny adaptive model
# ---------------------------
class AdaptiveTinyModel(pl.LightningModule):
    """
    - If batch has (features, targets): Linear -> MSE
    - Else if batch has "contexts": mean(contexts**2)
    - Else: mean of first float tensor
    Holds an anchor param so the optimizer is never empty.
    """
    def __init__(self, x_dim: Optional[int] = None, y_dim: Optional[int] = None, lr: float = 1e-2):
        super().__init__()
        self.lr = lr
        self.mse = nn.MSELoss()
        self._anchor = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(x_dim, y_dim) if (x_dim is not None and y_dim is not None) else None

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_key = pick_first_key(batch, X_CANDIDATES)
        y_key = pick_first_key(batch, Y_CANDIDATES)

        if x_key and y_key:
            x = batch[x_key].float()
            y = batch[y_key].float()
            if x.ndim == 3:  # (B, T, D)
                B, T, D = x.shape
                x = x.view(B * T, D)
                y = y.view(B * T, -1)
            if self.head is None:
                self.head = nn.Linear(x.shape[-1], y.shape[-1]).to(self.device)
            preds = self.head(x)
            return self.mse(preds, y)

        c_key = pick_first_key(batch, CTX_CANDIDATES)
        if c_key:
            c = batch[c_key].float()
            return (c ** 2).mean()

        for k, v in batch.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                return (v.float() ** 2).mean()

        raise RuntimeError("No usable tensor found in batch to compute a loss.")

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


# ---------------------------
# CLI / Trainer
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Test ContextualizedRegressionDataModule")
    p.add_argument("--task-type",
                   choices=[
                       "singletask_multivariate",
                       "singletask_univariate",
                       "multitask_multivariate",
                       "multitask_univariate",
                   ],
                   required=True)
    p.add_argument("--n", type=int, default=256, help="Total samples")
    p.add_argument("--c-dim", type=int, default=8, help="Context dim")
    p.add_argument("--x-dim", type=int, default=16, help="Feature dim")
    p.add_argument("--y-dim", type=int, default=4, help="Target dim")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=1)
    p.add_argument("--limit-train-batches", type=float, default=2)
    p.add_argument("--limit-val-batches", type=float, default=1)
    p.add_argument("--peek", action="store_true", help="Print first batch keys/shapes")
    return p.parse_args()


def _unset_dist_env():
    # Ensure env:// rendezvous is NOT selected
    for k in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK", "INIT_METHOD"):
        if k in os.environ:
            os.environ.pop(k)


def build_trainer(args) -> pl.Trainer:
    if args.devices > 1:
        # Force local file-store DDP init (no sockets/ports)
        init_path = Path(tempfile.gettempdir()) / f"pl_init_{os.getpid()}.pt"
        init_uri = init_path.as_uri()  # proper file:///C:/... on Windows
        strategy = DDPStrategy(
            process_group_backend="gloo",
            init_method=init_uri,
        )
    else:
        strategy = "auto"

    return pl.Trainer(
        accelerator="cpu",
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
        logger=False,
    )


def main():
    # Windows-safe start method for spawn/DDP
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    _unset_dist_env()
    args = parse_args()

    # --- synthetic data + splits ---
    C, X, Y = make_synthetic(n=args.n, c_dim=args.c_dim, x_dim=args.x_dim, y_dim=args.y_dim)
    train_idx, val_idx, test_idx = make_indices(args.n)

    # --- your datamodule ---
    dm = ContextualizedRegressionDataModule(
        C=C, X=X, Y=Y,
        task_type=args.task_type,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        predict_idx=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,           # CPU run
        persistent_workers=False,   # safe with num_workers=0
        drop_last=False,
        shuffle_train=True,
        shuffle_eval=False,
        dtype=torch.float,
    )

    # Setup and peek one batch to infer dims so model has parameters before optimizer init
    dm.setup("fit")
    sample = next(iter(dm.train_dataloader()))
    if args.peek:
        print("[peek] batch keys:", list(sample.keys()))
        for k, v in sample.items():
            if torch.is_tensor(v):
                print(f"[peek]  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        print()

    # Infer x_dim/y_dim from batch (handles (B,T,D))
    x_key = pick_first_key(sample, X_CANDIDATES)
    y_key = pick_first_key(sample, Y_CANDIDATES)
    x_dim = y_dim = None
    if x_key and y_key:
        x = sample[x_key]
        y = sample[y_key]
        x_dim = x.shape[-1]
        y_dim = y.shape[-1]

    # Build model (now has params)
    model = AdaptiveTinyModel(x_dim=x_dim, y_dim=y_dim)

    # Trainer
    trainer = build_trainer(args)
    trainer.fit(model, dm)
    print("✅ Test completed successfully.")


if __name__ == "__main__":
    main()
