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
        return np.round(super().predict(C, X, individual_preds, **kwargs))

    def predict_proba(self, C, X, **kwargs):
        """
        Predict probabilities of outcomes from context C and predictors X.

        Returns
        -------
        np.ndarray of shape (n_samples, y_dim, 2)
        """
        probs = super().predict(C, X, **kwargs)  # (n, y_dim[, 1])
        return np.array([1 - probs, probs]).T.swapaxes(0, 1)
