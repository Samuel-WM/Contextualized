"""
sklearn-like interface to Contextualized Classifiers.
"""

import numpy as np

from contextualized.functions import LINK_FUNCTIONS
from contextualized.easy import ContextualizedRegressor
from contextualized.regression import LOSSES


class ContextualizedClassifier(ContextualizedRegressor):
    """
    Contextualized Logistic Regression reveals context-dependent decisions and decision boundaries.
    Implemented as a ContextualizedRegressor with logistic link function and binary cross-entropy loss.
    """

    def __init__(self, **kwargs):
        kwargs["link_fn"] = LINK_FUNCTIONS["logistic"]
        kwargs["loss_fn"] = LOSSES["bceloss"]
        super().__init__(**kwargs)

    def predict(self, C, X, individual_preds=False, **kwargs):
        """Predict binary outcomes from context C and predictors X."""
        out = super().predict(C, X, individual_preds, **kwargs)
        out = np.asarray(out)
        if not individual_preds:
            if out.ndim == 3 and out.shape[-1] == 1:
                out = out[..., 0]
            return np.round(out)
        # individual_preds=True: list/array per-bootstrap -> squeeze each
        return [np.round(p[..., 0] if (p.ndim == 3 and p.shape[-1] == 1) else p) for p in out]


    def predict_proba(self, C, X, **kwargs):
        """
        Predict probabilities of outcomes from context C and predictors X.

        Returns
        -------
        np.ndarray of shape (n_samples, y_dim, 2)
        """
        probs = super().predict(C, X, **kwargs)  # (n, y_dim[, 1])
        probs = np.asarray(probs)
        if probs.ndim == 3 and probs.shape[-1] == 1:
            probs = probs[..., 0]
        p1 = probs
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=-1)