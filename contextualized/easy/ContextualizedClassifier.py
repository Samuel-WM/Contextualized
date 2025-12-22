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

    def predict(self, C, X, individual_preds: bool = False, **kwargs):
        out = super().predict(C, X, individual_preds=individual_preds, **kwargs)
        if out is None:
            return None

        out = np.asarray(out)

        if not individual_preds:
            # common binary case: (N, 1, 1) or (N, 1)
            if out.ndim == 3 and out.shape[-1] == 1:
                out = out[..., 0]
            return np.round(out)

        # individual_preds=True: list/array across bootstraps
        return [
            np.round(p[..., 0] if (p.ndim == 3 and p.shape[-1] == 1) else p)
            for p in out
        ]

    def predict_proba(self, C, X, **kwargs):
        probs = super().predict(C, X, **kwargs)
        if probs is None:
            return None

        probs = np.asarray(probs)
        if probs.ndim == 3 and probs.shape[-1] == 1:
            probs = probs[..., 0]

        p1 = probs
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=-1)
