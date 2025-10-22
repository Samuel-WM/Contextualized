# contextualized/regression/datamodules.py
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as pl

from .datasets import (
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)

TensorLike = Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
IndexLike = Optional[Union[Sequence[int], np.ndarray, torch.Tensor]]

TASK_TO_DATASET = {
    "singletask_multivariate": MultivariateDataset,
    "singletask_univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}


def _to_tensor(x: TensorLike, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, copy=False)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy(copy=False)
    # x is now np.ndarray or array-like
    return torch.tensor(x, dtype=dtype)


def _maybe_index(x: torch.Tensor, idx: IndexLike) -> torch.Tensor:
    if idx is None:
        return x
    if isinstance(idx, torch.Tensor):
        return x[idx]
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return x[idx]
    # assume Sequence[int]
    return x[torch.as_tensor(idx, dtype=torch.long)]


class ContextualizedRegressionDataModule(pl.LightningDataModule):
    """
    DataModule that returns map-style datasets for contextualized regression,
    allowing Lightning's Trainer (DDP) to auto-attach DistributedSampler and shard data.

    give  ∈ {
        "singletask_multivariate",
        "singletask_univariate",
        "multitask_multivariate",
        "multitask_univariate",
    }
    """

    def __init__(
        self,
        C: TensorLike,
        X: TensorLike,
        Y: Optional[TensorLike],
        *,
        task_type: str,
        # splits: pass explicit index arrays OR a splitter callable
        train_idx: IndexLike = None,
        val_idx: IndexLike = None,
        test_idx: IndexLike = None,
        predict_idx: IndexLike = None,
        splitter: Optional[
            Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                     Tuple[IndexLike, IndexLike, IndexLike]]
        ] = None,
        # dataloader config
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        shuffle_train: bool = True,
        shuffle_eval: bool = False,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        if task_type not in TASK_TO_DATASET:
            raise ValueError(
                f"Unknown task_type={task_type!r}. "
                f"Expected one of {list(TASK_TO_DATASET)}."
            )
        self.task_type = task_type

        # raw inputs (convert in setup)
        self._C_raw = C
        self._X_raw = X
        self._Y_raw = Y

        # split config
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.predict_idx = predict_idx
        self.splitter = splitter

        # dl config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = bool(persistent_workers and num_workers > 0)
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.shuffle_eval = shuffle_eval
        self.dtype = dtype

        # will be set in setup()
        self.C: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.ds_predict = None

    # One-time downloads or heavy ops would go here; we have none.
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Convert inputs to tensors
        C = _to_tensor(self._C_raw, self.dtype)
        X = _to_tensor(self._X_raw, self.dtype)
        Y = None if self._Y_raw is None else _to_tensor(self._Y_raw, self.dtype)

        # Basic shape sanity could be added here if desired.

        # If no explicit indices were given, allow a splitter to define them.
        if self.train_idx is None and self.val_idx is None and self.test_idx is None:
            if self.splitter is not None:
                tr, va, te = self.splitter(C, X, Y)
                self.train_idx, self.val_idx, self.test_idx = tr, va, te

        # If predict_idx not given, default to test indices (or full range if all None)
        if self.predict_idx is None:
            if self.test_idx is not None:
                self.predict_idx = self.test_idx
            else:
                self.predict_idx = torch.arange(C.shape[0], dtype=torch.long)

        # Slice tensors per split (map-style datasets rely on correct len() for sharding)
        def _mk_dataset(idx: IndexLike):
            if idx is None:
                return None
            C_s = _maybe_index(C, idx)
            X_s = _maybe_index(X, idx)
            Y_s = None if (Y is None) else _maybe_index(Y, idx)
            ds_cls = TASK_TO_DATASET[self.task_type]
            # Y can be optional for some tasks; the dataset constructors you showed
            # expect Y. If a task doesn't use Y, pass a placeholder or ensure callers pass X as Y when needed.
            if Y_s is None:
                # If Y is truly not used for this task_type, construct a compatible placeholder.
                # Here we create zeros with appropriate last dim to match dataset expectations.
                # For singletask_univariate/multivariate we assume Y has shape (n, y_dim).
                # Override as needed if your upstream code guarantees a Y.
                Y_s = torch.zeros((C_s.shape[0], X_s.shape[-1]), dtype=self.dtype)
            return ds_cls(C_s, X_s, Y_s, dtype=self.dtype)

        self.ds_train = _mk_dataset(self.train_idx)
        self.ds_val = _mk_dataset(self.val_idx)
        self.ds_test = _mk_dataset(self.test_idx)
        self.ds_predict = _mk_dataset(self.predict_idx)

        # Keep tensors for potential later use
        self.C, self.X, self.Y = C, X, Y

    # ---- Dataloaders ----
    def _common_dl_kwargs(self) -> Dict:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
        }

    def train_dataloader(self) -> DataLoader:
        if self.ds_train is None:
            raise RuntimeError("train dataset is not set; provide train_idx or splitter.")
        return DataLoader(
            dataset=self.ds_train,
            shuffle=self.shuffle_train,  # True only for train
            **self._common_dl_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.ds_val is None:
            raise RuntimeError("val dataset is not set; provide val_idx or splitter.")
        return DataLoader(
            dataset=self.ds_val,
            shuffle=self.shuffle_eval,   # False by default
            **self._common_dl_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.ds_test is None:
            raise RuntimeError("test dataset is not set; provide test_idx or splitter.")
        return DataLoader(
            dataset=self.ds_test,
            shuffle=self.shuffle_eval,   # False by default
            **self._common_dl_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        if self.ds_predict is None:
            raise RuntimeError("predict dataset is not set; provide predict_idx/test_idx.")
        # IMPORTANT: keep shuffle=False for stable ordering per-rank
        return DataLoader(
            dataset=self.ds_predict,
            shuffle=False,
            **self._common_dl_kwargs(),
        )
