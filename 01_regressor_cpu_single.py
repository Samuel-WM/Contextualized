import numpy as np

from contextualized.easy import ContextualizedRegressor

def main():
    np.random.seed(0)

    n = 96
    c_dim = 4
    x_dim = 6
    y_dim = 2

    C = np.random.randn(n, c_dim).astype(np.float32)
    X = np.random.randn(n, x_dim).astype(np.float32)

    # Construct a learnable signal: Y = X @ W + noise
    W = np.array([[1.5, -0.5],
                  [0.7,  0.2],
                  [0.0,  0.0],
                  [0.3, -1.0],
                  [0.0,  0.0],
                  [0.2,  0.1]], dtype=np.float32)  # (x_dim, y_dim)

    Y = (X @ W + 0.05 * np.random.randn(n, y_dim).astype(np.float32)).astype(np.float32)

    model = ContextualizedRegressor(
        metamodel_type="subtype",
        num_archetypes=4,
        univariate=False,
    )

    # CPU-only fit
    model.fit(
        C=C, X=X, Y=Y,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        max_epochs=3,
        val_split=0.2,
        num_workers=0,
        enable_progress_bar=False,
        logger=False,
    )

    yhat = model.predict(C, X)
    betas, mus = model.predict_params(C)

    # --- shape sanity ---
    yhat_arr = np.asarray(yhat)
    betas_arr = np.asarray(betas)
    mus_arr = np.asarray(mus)

    print("SINGLE CPU REGRESSOR")
    print("yhat.shape:", yhat_arr.shape)
    print("betas.shape:", betas_arr.shape)
    print("mus.shape:", mus_arr.shape)

    # Expected conventions (based on your current implementation)
    assert yhat_arr.shape[0] == n, "yhat first dim should be n"
    assert betas_arr.shape[0] == n, "betas first dim should be n"
    assert mus_arr.shape[0] == n, "mus first dim should be n"

    # yhat is typically (n, y_dim, 1) for multivariate in your code path
    # betas is (n, y_dim, x_dim)
    assert betas_arr.shape[1] == y_dim and betas_arr.shape[2] == x_dim, "betas expected (n, y_dim, x_dim)"

    # --- quick quality check: MSE vs baseline mean predictor ---
    # squeeze last dim if present
    yhat_s = yhat_arr[..., 0] if (yhat_arr.ndim == 3 and yhat_arr.shape[-1] == 1) else yhat_arr
    y_true = Y

    mse = np.mean((yhat_s - y_true) ** 2)
    baseline = np.mean((np.mean(y_true, axis=0, keepdims=True) - y_true) ** 2)

    print("MSE:", float(mse))
    print("Baseline MSE (mean predictor):", float(baseline))
    assert np.isfinite(mse), "MSE must be finite"
    assert mse < baseline, "Model should beat baseline mean predictor on this synthetic signal"

    # --- ordering check (this is critical for your gather/sort design) ---
    perm = np.random.permutation(n)
    yhat_perm = np.asarray(model.predict(C[perm], X[perm]))
    yhat_perm_s = yhat_perm[..., 0] if (yhat_perm.ndim == 3 and yhat_perm.shape[-1] == 1) else yhat_perm

    max_err = np.max(np.abs(yhat_perm_s - yhat_s[perm]))
    print("Ordering check max_err:", float(max_err))
    assert max_err < 1e-5, "Prediction order is not stable under permutation"

    print("PASS: single-process CPU regressor tests")

if __name__ == "__main__":
    main()
