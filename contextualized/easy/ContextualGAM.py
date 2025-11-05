"""
Contextual Generalized Additive Model.
See https://www.sciencedirect.com/science/article/pii/S1532046422001022
for more details.
"""

from contextualized.easy import ContextualizedClassifier
from contextualized.easy import ContextualizedRegressor


class ContextualGAMClassifier(ContextualizedClassifier):
    """
    Contextual GAM Classifier with a Neural Additive Model ("ngam") encoder.
    Inherits the sklearn-like API from ContextualizedClassifier.
    """

    def __init__(self, **kwargs):
        # Force interpretability via NAM encoder
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)


class ContextualGAMRegressor(ContextualizedRegressor):
    """
    Contextual GAM Regressor with a Neural Additive Model ("ngam") encoder.
    Inherits the sklearn-like API from ContextualizedRegressor.
    """

    def __init__(self, **kwargs):
        # Force interpretability via NAM encoder
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)
