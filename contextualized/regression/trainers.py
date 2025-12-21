"""
PyTorch-Lightning trainers used for Contextualized regression.
"""

from typing import Any, Tuple, List
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import LightningEnvironment
import os
from pytorch_lightning.strategies import DDPStrategy


def _stack_from_preds(preds: List[dict], key: str) -> torch.Tensor:
    """Concatenate a tensor field from the list of batch dicts returned by predict()."""
    parts = []
    for p in preds:
        val = p[key]
        # ensure tensor on cpu
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        parts.append(val.detach().cpu())
    return torch.cat(parts, dim=0)


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression lightning_modules
    and provides convenience prediction helpers that reshape
    batched outputs into expected numpy arrays without relying
    on model-private _*reshape helpers.
    """

    @torch.no_grad()
    def predict_params(self, model: pl.LightningModule, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns context-specific regression parameters.

        Returns
        -------
        (betas, mus)
            betas: (n, y_dim, x_dim)
            mus:   (n, y_dim) or (n, y_dim, 1) depending on the model
        """
        preds = super().predict(model, dataloader)  # list of batch dicts
        betas = _stack_from_preds(preds, "betas")
        mus   = _stack_from_preds(preds, "mus")
        return betas.numpy(), mus.numpy()

    @torch.no_grad()
    def predict_y(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific predictions of the response Y.

        Returns
        -------
        y_hat : (n, y_dim, 1) for multivariate, or (n, y_dim, x_dim) for univariate
        """
        preds = super().predict(model, dataloader)  # list of batch dicts

        y_parts = []
        for p in preds:
            # Required keys were added by model.predict_step(...)
            C = p["contexts"]
            X = p["predictors"]
            betas = p["betas"]
            mus = p["mus"]

            # Ensure tensors on CPU first; model will move as needed inside helpers
            if not torch.is_tensor(C):     C = torch.as_tensor(C)
            if not torch.is_tensor(X):     X = torch.as_tensor(X)
            if not torch.is_tensor(betas): betas = torch.as_tensor(betas)
            if not torch.is_tensor(mus):   mus = torch.as_tensor(mus)

            # --- shape fixes for multivariate (3D) and univariate (4D) ---
            # Multivariate convention: X (B, y, x), betas (B, y, x), mus (B, y, 1)
            # Univariate convention:   X (B, y, x, 1), betas (B, y, x, 1), mus (B, y, x, 1)

            # If X is (B, x) and betas is (B, y, x), expand X -> (B, 1, x)
            if X.dim() == 2 and betas.dim() == 3 and betas.size(-1) == X.size(-1):
                X = X.unsqueeze(1)

            # If betas is (B, y, x) but X is (B, y, x, 1), add trailing singleton to betas
            if betas.dim() == 3 and X.dim() == 4 and betas.size(-1) == X.size(-2):
                betas = betas.unsqueeze(-1)
                

            # Ensure mus trailing dim is singleton
            if mus.dim() == 2:               # (B, y)
                mus = mus.unsqueeze(-1)      # (B, y, 1)
            elif mus.dim() == 3 and X.dim() == 4 and mus.size(-1) != 1:
                mus = mus.unsqueeze(-1)      # (B, y, x, 1)
            # --- end shape fixes ---


            yhat = model._predict_y(C, X, betas, mus)  # uses model's link
            y_parts.append(yhat.detach().cpu())

        y = torch.cat(y_parts, dim=0)
        return y.numpy()



class CorrelationTrainer(RegressionTrainer):
    """
    Trains the contextualized.regression correlation lightning_modules
    and exposes a helper to compute context-specific correlation matrices.
    """

    @torch.no_grad()
    def predict_correlation(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific correlation networks containing Pearson's correlation coefficient.

        Returns
        -------
        correlations : (n, x_dim, x_dim)
        """
        # If the model already returns 'correlations' in predict_step, prefer that.
        preds = super().predict(model, dataloader)
        if "correlations" in preds[0]:
            cors = torch.cat([p["correlations"].detach().cpu() for p in preds], dim=0)
            return cors.numpy()

        # Fallback: derive from betas like before
        betas, _ = self.predict_params(model, dataloader)
        signs = np.sign(betas)
        signs[signs != np.transpose(signs, (0, 2, 1))] = 0
        correlations = signs * np.sqrt(np.abs(betas * np.transpose(betas, (0, 2, 1))))
        return correlations


class MarkovTrainer(CorrelationTrainer):
    """
    Trains the contextualized.regression markov graph lightning_modules
    and exposes a helper to compute context-specific precision matrices.
    """

    @torch.no_grad()
    def predict_precision(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific precision matrix under a Gaussian graphical model.

        Assuming all diagonal precisions are equal and constant over context,
        this is equivalent to the negative of the multivariate regression coefficient.

        Returns
        -------
        precision : (n, x_dim, x_dim)
        """
        # A trick in the markov lightning_module predict_step ensures the
        # correlation output corresponds (up to sign) to precision entries.
        return -super().predict_correlation(model, dataloader)



def choose_lightning_environment() -> LightningEnvironment:
    # If you have a custom Environment subclass, wire it here.
    # Otherwise, the default LightningEnvironment is fine.
    return LightningEnvironment()

def make_trainer_with_env(trainer_cls, **trainer_kwargs):
    """
    Factory that respects caller-provided `devices` and `strategy`.
    FIXED: Don't inject LightningEnvironment when torchrun is managing processes.
    """
    import os
    
    # Check if we're under torchrun (WORLD_SIZE > 1 means torchrun is managing)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Only inject LightningEnvironment for single-process runs
    # When torchrun is active, Lightning will auto-detect TorchElasticEnvironment
    if "plugins" not in trainer_kwargs and world_size == 1:
        env = choose_lightning_environment()
        trainer_kwargs["plugins"] = [env]

    return trainer_cls(**trainer_kwargs)