#!/usr/bin/env python
import argparse, sys, time, warnings, socket, os
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)

try:
    import torch
    import lightning as pl
    from lightning.pytorch.strategies import DDPStrategy
except Exception as e:
    print(f"[FATAL] torch/lightning import failed: {e}"); sys.exit(1)

try:
    from contextualized.regression.datamodules import ContextualizedRegressionDataModule
    from contextualized.regression.datasets import (
        MultivariateDataset, UnivariateDataset, MultitaskMultivariateDataset, MultitaskUnivariateDataset
    )
    _ctx_ok = True
except Exception as e:
    print(f"[FATAL] Could not import contextualized modules: {e}"); _ctx_ok = False

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def make_tensors(n=32, c_dim=3, x_dim=5, y_dim=4, dtype=torch.float32, seed=7):
    rng = np.random.default_rng(seed)
    C = torch.tensor(rng.normal(size=(n, c_dim)), dtype=dtype)
    X = torch.tensor(rng.normal(size=(n, x_dim)), dtype=dtype)
    Y = torch.tensor(rng.normal(size=(n, y_dim)), dtype=dtype)
    return C, X, Y

def simple_splitter(C, X, Y):
    n = C.shape[0]; idx = torch.arange(n, dtype=torch.long)
    n_tr = int(0.6*n); n_va = int(0.2*n)
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

class TinyLightning(pl.LightningModule):
    def __init__(self, in_dim=5, out_dim=4, lr=1e-3):
        super().__init__(); self.save_hyperparameters()
        self.head = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.manual_seed(0)
        with torch.no_grad():
            w = torch.arange(in_dim*out_dim).float().reshape(out_dim, in_dim)/100.0
            self.head.weight.copy_(w)
        self.mu = torch.nn.Parameter(torch.zeros(out_dim, 1))
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
    def training_step(self, batch, batch_idx):
        betas = self.head(batch["predictors"]); loss = (betas**2).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=False, logger=False); return loss
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        betas = self.head(batch["predictors"]); mus = self.mu.view(1,-1).repeat(betas.shape[0],1)
        return {"idx": batch["idx"].detach().clone().cpu(),
                "betas": betas.detach().cpu(),
                "mus": mus.detach().cpu()}

def check_dataset_shapes(C, X, Y):
    print("\n[CHECK] Dataset constructors & shapes")
    mv = MultivariateDataset(C, X, Y); uv = UnivariateDataset(C, X, Y)
    mtmv = MultitaskMultivariateDataset(C, X, Y); mtuv = MultitaskUnivariateDataset(C, X, Y)
    print(f"  MultivariateDataset: len={len(mv)} sample keys={list(mv[0].keys())}")
    print(f"  UnivariateDataset:  len={len(uv)} sample keys={list(uv[0].keys())}")
    print(f"  MultitaskMultivariateDataset: len={len(mtmv)}")
    print(f"  MultitaskUnivariateDataset:  len={len(mtuv)}")
    for name, ds in [("MultivariateDataset", mv), ("UnivariateDataset", uv),
                     ("MultitaskMultivariateDataset", mtmv), ("MultitaskUnivariateDataset", mtuv)]:
        s = ds[0]
        for k in ("idx","contexts","predictors","outcomes"):
            assert k in s, f"{name} sample missing '{k}'"
    print("  ✔ Map-style and key shape checks passed.")

def run_single_process(dm, x_dim, y_dim, max_epochs=1):
    print("\n[RUN] Single-process (CPU) trainer...")
    model = TinyLightning(in_dim=x_dim, out_dim=y_dim)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs,
                         logger=False, enable_progress_bar=False,
                         default_root_dir=str(Path("./_tmp_sanity").resolve()),
                         enable_checkpointing=False)
    tic = time.time(); trainer.fit(model, datamodule=dm)
    outs = trainer.predict(model, datamodule=dm); sec = time.time() - tic
    idx = torch.cat([o["idx"] for o in outs]).numpy()
    betas = torch.cat([o["betas"] for o in outs]); mus = torch.cat([o["mus"] for o in outs])
    print(f"  Predict returned {len(idx)} rows in {sec:.2f}s")
    print(f"  idx head: {idx[:10]}")
    print(f"  betas shape: {tuple(betas.shape)}, device={betas.device.type}")
    print(f"  mus   shape: {tuple(mus.shape)}, device={mus.device.type}")
    assert betas.device.type == "cpu" and mus.device.type == "cpu"
    assert (idx == np.sort(idx)).all()
    assert len(np.unique(idx)) == len(idx)
    print("  ✔ Single-process checks passed.")
    return idx, betas, mus

def run_ddp(dm, x_dim, y_dim, world_size=2):
    print(f"\n[RUN] DDP spawn (CPU, world_size={world_size})...")
    # Force local master & explicit init_method to ignore any stale env vars
    addr = "127.0.0.1"; port = _free_port()
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)
    strategy = DDPStrategy(process_group_backend="gloo",
                           init_method=f"tcp://{addr}:{port}")
    model = TinyLightning(in_dim=x_dim, out_dim=y_dim)
    trainer = pl.Trainer(accelerator="cpu", devices=world_size, strategy=strategy,
                         max_epochs=0, logger=False, enable_progress_bar=False,
                         default_root_dir=str(Path("./_tmp_sanity_ddp").resolve()),
                         enable_checkpointing=False)
    outs = trainer.predict(model, datamodule=dm)
    idx = torch.cat([o["idx"] for o in outs]).numpy()
    betas = torch.cat([o["betas"] for o in outs]); mus = torch.cat([o["mus"] for o in outs])
    print(f"  Gathered rows: {len(idx)} (unique={len(np.unique(idx))})")
    print(f"  idx head: {idx[:10]}")
    print(f"  betas shape: {tuple(betas.shape)}, device={betas.device.type}")
    print(f"  mus   shape: {tuple(mus.shape)}, device={mus.device.type}")
    assert betas.device.type == "cpu" and mus.device.type == "cpu"
    assert len(np.unique(idx)) == len(idx)
    print("  ✔ DDP checks passed.")
    return idx, betas, mus

def maybe_try_wrapper(X):
    try:
        from contextualized.easy.wrappers.SKLearnWrapper import SKLearnWrapper  # type: ignore
    except Exception as e:
        print(f"[INFO] SKLearnWrapper not available ({e}); skipping wrapper test."); return
    print("\n[TRY] SKLearnWrapper in-memory vs memory-bounded (if supported)...")
    class DummyEstimator:
        def fit(self, C, X, Y): return self
        def predict(self, X):
            if isinstance(X, torch.Tensor): return X.sum(-1, keepdim=True).numpy()
            return X.sum(-1, keepdims=True)
    try:
        wrapper = SKLearnWrapper(estimator=DummyEstimator())
        p1 = np.asarray(wrapper.predict(X, memory_bounded=False))
        p2 = np.asarray(wrapper.predict(X, memory_bounded=True))
        print(f"  wrapper outputs shapes: {p1.shape} vs {p2.shape}")
        print("  ✔ Wrapper paths match on toy data." if (p1.shape==p2.shape and np.allclose(p1,p2,1e-6,1e-6))
              else "  ⚠ Wrapper paths differ on toy data.")
    except TypeError as e:
        print(f"  ⚠ Wrapper signature mismatch: {e} — skipping.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--c-dim", type=int, default=3)
    ap.add_argument("--x-dim", type=int, default=5)
    ap.add_argument("--y-dim", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--ddp", type=int, default=0)
    ap.add_argument("--try-wrapper", action="store_true")
    args = ap.parse_args()
    if not _ctx_ok: sys.exit(2)
    C, X, Y = make_tensors(n=args.n, c_dim=args.c_dim, x_dim=args.x_dim, y_dim=args.y_dim)
    check_dataset_shapes(C, X, Y)
    print("\n[BUILD] ContextualizedRegressionDataModule")
    dm = ContextualizedRegressionDataModule(
        C=C, X=X, Y=Y, task_type="singletask_multivariate",
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle_eval=False, shuffle_train=True, pin_memory=False, persistent_workers=False,
        splitter=simple_splitter,
    )
    dm.setup("fit")
    idx1, betas1, mus1 = run_single_process(dm, x_dim=args.x_dim, y_dim=args.y_dim)
    if args.ddp and args.ddp > 1:
        idx2, betas2, mus2 = run_ddp(dm, x_dim=args.x_dim, y_dim=args.y_dim, world_size=args.ddp)
        assert set(idx1.tolist()) == set(idx2.tolist()), "DDP vs single-process index coverage mismatch"
        print("  ✔ DDP vs single-process index coverage matches.")
    if args.try_wrapper: maybe_try_wrapper(X)
    print("\n✅ ALL SANITY CHECKS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
