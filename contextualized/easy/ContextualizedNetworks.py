"""
sklearn-like interface to Contextualized Networks.
"""

from typing import List, Tuple, Union

import numpy as np

from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression.trainers import CorrelationTrainer, MarkovTrainer
from contextualized.regression.lightning_modules import (
    ContextualizedCorrelation,
    ContextualizedMarkovGraph,
)
from contextualized.dags.lightning_modules import (
    NOTMAD,
    DEFAULT_DAG_LOSS_TYPE,
    DEFAULT_DAG_LOSS_PARAMS,
)
from contextualized.dags.trainers import GraphTrainer
from contextualized.dags.graph_utils import dag_pred_np


class ContextualizedNetworks(SKLearnWrapper):
    """
    sklearn-like interface to Contextualized Networks.
    """

    def _split_train_data(
        self, C: np.ndarray, X: np.ndarray, **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Splits data into train and val sets (no Y for networks)."""
        return super()._split_train_data(C, X, Y_required=False, **kwargs)

    def predict_networks(
        self,
        C: np.ndarray,
        with_offsets: bool = False,
        individual_preds: bool = False,
        **kwargs,
    ) -> Union[
        np.ndarray,
        List[np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[List[np.ndarray], List[np.ndarray]],
    ]:
        """
        Predicts context-specific network parameters (and offsets if available).
        """
        betas, mus = self.predict_params(
            C, individual_preds=individual_preds, uses_y=False, **kwargs
        )
        return (betas, mus) if with_offsets else betas

    def predict_X(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Reconstructs X via predicted networks using the base wrapper predict().
        """
        return self.predict(C, X, individual_preds=individual_preds, **kwargs)


class ContextualizedCorrelationNetworks(ContextualizedNetworks):
    """
    Contextualized Correlation Networks reveal context-varying feature correlations.
    Uses the Contextualized Networks model.
    """

    def __init__(self, **kwargs):
        super().__init__(
            ContextualizedCorrelation, [], [], CorrelationTrainer, **kwargs
        )

    def predict_correlation(
        self, C: np.ndarray, individual_preds: bool = True, squared: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        C_scaled = self._maybe_scale_C(C)
        Y_zero = np.zeros((len(C_scaled), self.x_dim), dtype=np.float32)
        dm = self._build_datamodule(
            C=C_scaled,
            X=np.zeros((len(C_scaled), self.x_dim), dtype=np.float32),
            Y=Y_zero,
            predict_idx=np.arange(len(C_scaled)),
            data_kwargs=dict(
                batch_size=self._init_kwargs["data"].get("val_batch_size", 16),
                num_workers=self._init_kwargs["data"].get("num_workers", 0),
                pin_memory=self._init_kwargs["data"].get("pin_memory", (self.accelerator == "gpu")),
                persistent_workers=self._init_kwargs["data"].get("persistent_workers", False),
                shuffle_train=False, shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate",  # correlation uses univariate convention
        )
        rhos = np.array([
            self.trainers[i].predict_correlation(self.models[i], dm.predict_dataloader())
            for i in range(len(self.models))
        ])
        if individual_preds:
            return np.square(rhos) if squared else rhos
        mean_rhos = np.mean(rhos, axis=0)
        return np.square(mean_rhos) if squared else mean_rhos

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Measures mean-squared reconstruction errors using (betas, mus).
        """
        betas, mus = self.predict_networks(C, individual_preds=True, with_offsets=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        F = X.shape[-1]
        for i in range(F):
            for j in range(F):
                tiled_xi = np.array([X[:, i] for _ in range(len(betas))])
                tiled_xj = np.array([X[:, j] for _ in range(len(betas))])
                residuals = tiled_xi - betas[:, :, i, j] * tiled_xj - mus[:, :, i, j]
                mses += residuals**2 / (F**2)
        return mses if individual_preds else np.mean(mses, axis=0)


class ContextualizedMarkovNetworks(ContextualizedNetworks):
    """
    Contextualized Markov Networks (Gaussian precision matrices).
    """

    def __init__(self, **kwargs):
        super().__init__(ContextualizedMarkovGraph, [], [], MarkovTrainer, **kwargs)

    def predict_precisions(
        self, C: np.ndarray, individual_preds: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predicts context-specific precision matrices.
        """
        C_scaled = self._maybe_scale_C(C)
        Y_zero = np.zeros((len(C_scaled), self.x_dim), dtype=np.float32)
        dm = self._build_datamodule(
            C=C_scaled,
            X=np.zeros((len(C_scaled), self.x_dim), dtype=np.float32),
            Y=Y_zero,
            predict_idx=np.arange(len(C_scaled)),
            data_kwargs=dict(
                batch_size=self._init_kwargs["data"].get("val_batch_size", 16),
                num_workers=self._init_kwargs["data"].get("num_workers", 0),
                pin_memory=self._init_kwargs["data"].get("pin_memory", (self.accelerator == "gpu")),
                persistent_workers=self._init_kwargs["data"].get("persistent_workers", False),
                shuffle_train=False, shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate",
        )
        precisions = np.array([
            self.trainers[i].predict_precision(self.models[i], dm.predict_dataloader())
            for i in range(len(self.models))
        ])
        return precisions if individual_preds else np.mean(precisions, axis=0)

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Measures mean-squared reconstruction errors using precision-implied betas/mus.
        """
        betas, mus = self.predict_networks(C, individual_preds=True, with_offsets=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        F = X.shape[-1]
        for b in range(len(betas)):
            for i in range(F):
                preds = np.array(
                    [
                        X[j].dot(betas[b, j, i, :]) + mus[b, j, i]
                        for j in range(len(X))
                    ]
                )
                residuals = X[:, i] - preds
                mses[b, :] += residuals**2 / F
        return mses if individual_preds else np.mean(mses, axis=0)


class ContextualizedBayesianNetworks(ContextualizedNetworks):
    """
    Contextualized Bayesian Networks (NOTMAD): context-dependent DAGs.
    """

    def _parse_private_init_kwargs(self, **kwargs):
        """
        Parse NOTMAD kwargs into model init dicts.
        """
        # Encoder Parameters
        self._init_kwargs["model"]["encoder_kwargs"] = {
            "type": kwargs.pop(
                "encoder_type", self._init_kwargs["model"]["encoder_type"]
            ),
            "params": {
                "width": self.constructor_kwargs["encoder_kwargs"]["width"],
                "layers": self.constructor_kwargs["encoder_kwargs"]["layers"],
                "link_fn": self.constructor_kwargs["encoder_kwargs"]["link_fn"],
            },
        }

        # Archetype parameters
        archetype_dag_loss_type = kwargs.pop(
            "archetype_dag_loss_type", DEFAULT_DAG_LOSS_TYPE
        )
        self._init_kwargs["model"]["archetype_loss_params"] = {
            "l1": kwargs.get("archetype_l1", 0.0),
            "dag": kwargs.get(
                "archetype_dag_params",
                {
                    "loss_type": archetype_dag_loss_type,
                    "params": kwargs.get(
                        "archetype_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[archetype_dag_loss_type].copy(),
                    ),
                },
            ),
            "init_mat": kwargs.pop("init_mat", None),
            "num_factors": kwargs.pop("num_factors", 0),
            "factor_mat_l1": kwargs.pop("factor_mat_l1", 0),
            "num_archetypes": kwargs.pop("num_archetypes", 16),
        }
        if self._init_kwargs["model"]["archetype_loss_params"]["num_archetypes"] <= 0:
            print(
                "WARNING: num_archetypes is 0. NOTMAD requires archetypes. Setting num_archetypes to 16."
            )
            self._init_kwargs["model"]["archetype_loss_params"]["num_archetypes"] = 16

        # Allow convenience overrides for archetype DAG params
        for param, value in self._init_kwargs["model"]["archetype_loss_params"]["dag"][
            "params"
        ].items():
            self._init_kwargs["model"]["archetype_loss_params"]["dag"]["params"][
                param
            ] = kwargs.pop(f"archetype_{param}", value)

        # Sample-specific parameters
        sample_specific_dag_loss_type = kwargs.pop(
            "sample_specific_dag_loss_type", DEFAULT_DAG_LOSS_TYPE
        )
        self._init_kwargs["model"]["sample_specific_loss_params"] = {
            "l1": kwargs.pop("sample_specific_l1", 0.0),
            "dag": kwargs.pop(
                "sample_specific_loss_params",
                {
                    "loss_type": sample_specific_dag_loss_type,
                    "params": kwargs.pop(
                        "sample_specific_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[sample_specific_dag_loss_type].copy(),
                    ),
                },
            ),
        }
        for param, value in self._init_kwargs["model"]["sample_specific_loss_params"][
            "dag"
        ]["params"].items():
            self._init_kwargs["model"]["sample_specific_loss_params"]["dag"]["params"][
                param
            ] = kwargs.pop(f"sample_specific_{param}", value)

        # Optimization parameters
        self._init_kwargs["model"]["opt_params"] = {
            "learning_rate": kwargs.pop("learning_rate", 1e-3),
            "step": kwargs.pop("step", 50),
        }

        return [
            "archetype_dag_loss_type",
            "archetype_l1",
            "archetype_dag_params",
            "archetype_dag_loss_params",
            "archetype_dag_loss_type",
            "archetype_alpha",
            "archetype_rho",
            "archetype_s",
            "archetype_tol",
            "archetype_loss_params",
            "archetype_use_dynamic_alpha_rho",
            "init_mat",
            "num_factors",
            "factor_mat_l1",
            "sample_specific_dag_loss_type",
            "sample_specific_alpha",
            "sample_specific_rho",
            "sample_specific_s",
            "sample_specific_tol",
            "sample_specific_loss_params",
            "sample_specific_use_dynamic_alpha_rho",
        ]

    def __init__(self, **kwargs):
        super().__init__(
            NOTMAD,
            extra_model_kwargs=[
                "sample_specific_loss_params",
                "archetype_loss_params",
                "opt_params",
            ],
            extra_data_kwargs=[],
            trainer_constructor=GraphTrainer,
            remove_model_kwargs=[
                "link_fn",
                "univariate",
                "loss_fn",
                "model_regularizer",
            ],
            **kwargs,
        )

    def predict_params(
        self, C: np.ndarray, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predicts context-specific Bayesian network parameters (SEM coefficients).
        """
        # No mus for NOTMAD at present.
        return super().predict_params(C, model_includes_mus=False, **kwargs)

    def predict_networks(
        self, C: np.ndarray, project_to_dag: bool = True, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predicts context-specific Bayesian networks (optionally projected to DAG).
        """
        if kwargs.pop("with_offsets", False):
            print("No offsets can be returned by NOTMAD.")
        betas = self.predict_params(
            C, uses_y=False, project_to_dag=project_to_dag, **kwargs
        )
        return betas

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Measures mean-squared errors of DAG-based reconstruction.
        """
        betas = self.predict_networks(C, individual_preds=True, **kwargs)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for b in range(len(betas)):
            X_pred = dag_pred_np(X, betas[b])
            mses[b, :] = np.mean((X - X_pred) ** 2, axis=1)
        return mses if individual_preds else np.mean(mses, axis=0)
