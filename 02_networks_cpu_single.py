import numpy as np

from contextualized.easy import ContextualizedCorrelationNetworks  # adjust if your import path differs

def main():
    np.random.seed(0)

    n = 80
    c_dim = 3
    x_dim = 5

    C = np.random.randn(n, c_dim).astype(np.float32)
    X = np.random.randn(n, x_dim).astype(np.float32)

    net = ContextualizedCorrelationNetworks(
        metamodel_type="subtype",
        num_archetypes=4,
    )

    net.fit(
        C=C, X=X,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        max_epochs=2,
        val_split=0.2,
        num_workers=0,
        enable_progress_bar=False,
        logger=False,
    )

    rhos2 = net.predict_correlation(C, individual_preds=False, squared=True)
    rhos2 = np.asarray(rhos2)

    print("SINGLE CPU CORRELATION NETWORKS")
    print("rhos2.shape:", rhos2.shape)

    assert rhos2.shape[0] == n and rhos2.shape[1] == x_dim and rhos2.shape[2] == x_dim, \
        "Expected (n, x_dim, x_dim)"
    assert np.all(np.isfinite(rhos2)), "Correlations must be finite"

    # Symmetry sanity (should be symmetric-ish)
    sym_err = np.max(np.abs(rhos2 - np.transpose(rhos2, (0, 2, 1))))
    print("Symmetry max_err:", float(sym_err))

    print("PASS: single-process CPU networks tests")

if __name__ == "__main__":
    main()
