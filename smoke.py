#!/usr/bin/env python3
"""
CPU-only smoke test for Contextualized (no val loop, ~2–5s on a laptop CPU).

- Forces CPU (disables CUDA)
- Generates a tiny synthetic regression set
- Fits a very small ContextualizedRegressor for 2 epochs
- Prints wall time and a few predictions
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU before importing torch/PL

import numpy as np
from contextualized.easy.ContextualizedRegressor import ContextualizedRegressor

def make_synth(n=2_000, c_dim=8, x_dim=16, y_dim=1, seed=123):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, c_dim)).astype(np.float32)
    X = rng.normal(size=(n, x_dim)).astype(np.float32)
    # context-conditioned linear truth
    W = rng.normal(size=(c_dim, x_dim, y_dim)).astype(np.float32)
    Y = (C @ W.reshape(c_dim, -1)).reshape(n, x_dim, y_dim)
    Y = (X[..., None] * Y).sum(axis=1) + 0.1 * rng.normal(size=(n, y_dim)).astype(np.float32)
    return C, X, Y

def main():
    C, X, Y = make_synth()

    # Tiny model, no validation, CPU-only trainer settings live inside .fit kwargs
    model = ContextualizedRegressor(
        encoder_type="mlp",
        width=16,
        layers=2,
        learning_rate=1e-3,
        univariate=False,   # multivariate target OK; here y_dim=1 anyway
    )

    t0 = time.time()
    model.fit(
        X, Y, C,                 # README order: (X, Y, C)
        # ----- data -----
        train_batch_size=128,
        num_workers=0,
        val_split=0.0,           # <— no val loop (avoids EarlyStopping/val_loss)
        # ----- trainer -----
        accelerator="cpu",
        devices=1,
        strategy="auto",
        max_epochs=2,
        enable_progress_bar=False,
        logger=False,
        limit_val_batches=0,
        # safety/consistency if callbacks sneak in
        es_patience=0,
    )
    dt = time.time() - t0

    # quick predict
    yhat = model.predict(C[:8], X[:8])
    print(f"\nDone. Wall time: {dt:.2f}s")
    print("Pred sample (first 5 rows):\n", np.asarray(yhat)[:5].round(3))

if __name__ == "__main__":
    main()
