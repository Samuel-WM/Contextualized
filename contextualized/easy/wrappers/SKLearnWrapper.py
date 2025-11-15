# --- imports you need above the class ---
import copy
import os
from typing import *
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy  # PL v1 Strategy API

from contextualized.functions import LINK_FUNCTIONS
from contextualized.regression import REGULARIZERS, LOSSES
from contextualized.regression.datamodules import ContextualizedRegressionDataModule

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_BOOTSTRAPS = 1
DEFAULT_ES_PATIENCE = 1
DEFAULT_VAL_BATCH_SIZE = 16
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_TEST_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_ENCODER_TYPE = "mlp"
DEFAULT_ENCODER_WIDTH = 25
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_ENCODER_LINK_FN = LINK_FUNCTIONS["identity"]
DEFAULT_NORMALIZE = False


class SKLearnWrapper:
    """
    An sklearn-like wrapper for Contextualized models.

    Args:
        base_constructor (class): Base LightningModule constructor.
        extra_model_kwargs (Iterable[str]): Extra model kwargs to accept.
        extra_data_kwargs (Iterable[str]): Extra data kwargs to accept.
        trainer_constructor (class): Trainer class (usually RegressionTrainer).
        normalize (bool): If True, standardize C/X (and Y if continuous).
    """

    # -------------------- defaults --------------------
    def _set_defaults(self):
        self.default_learning_rate = DEFAULT_LEARNING_RATE
        self.default_n_bootstraps = DEFAULT_N_BOOTSTRAPS
        self.default_es_patience = DEFAULT_ES_PATIENCE
        self.default_train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
        self.default_test_batch_size = DEFAULT_TEST_BATCH_SIZE
        self.default_val_batch_size = DEFAULT_VAL_BATCH_SIZE
        self.default_val_split = DEFAULT_VAL_SPLIT
        self.default_encoder_width = DEFAULT_ENCODER_WIDTH
        self.default_encoder_layers = DEFAULT_ENCODER_LAYERS
        self.default_encoder_link_fn = DEFAULT_ENCODER_LINK_FN
        self.default_encoder_type = DEFAULT_ENCODER_TYPE
        self.default_normalize = DEFAULT_NORMALIZE

    def __init__(
        self,
        base_constructor,
        extra_model_kwargs,
        extra_data_kwargs,
        trainer_constructor,
        **kwargs,
    ):
        self._set_defaults()
        self.base_constructor = base_constructor
        self.trainer_constructor = trainer_constructor

        self.n_bootstraps = 1
        self.models = None
        self.trainers = None

        self.normalize = kwargs.pop("normalize", self.default_normalize)
        self.scalers = {"C": None, "X": None, "Y": None}
        self.context_dim = None
        self.x_dim = None
        self.y_dim = None
        self.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        # Accepted kwarg routes
        self.acceptable_kwargs = {
            "data": [
                "train_batch_size",
                "val_batch_size",
                "test_batch_size",
                "predict_batch_size",
                "C_val",
                "X_val",
                "Y_val",
                "val_split",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "drop_last",
                "shuffle_train",
                "shuffle_eval",
                "dtype",
            ],
            "model": [
                "loss_fn",
                "link_fn",
                "univariate",
                "encoder_type",
                "encoder_kwargs",
                "model_regularizer",
                "num_archetypes",
                "learning_rate",
                "context_dim",
                "x_dim",
                "y_dim",
            ],
            "trainer": [
                "max_epochs",
                "check_val_every_n_epoch",
                "val_check_interval",
                "callbacks",
                "callback_constructors",
                "accelerator",
                "devices",
                "strategy",
                "plugins",
                "logger",
                "enable_checkpointing",
                "num_sanity_val_steps",
                "default_root_dir",
                "log_every_n_steps",
                "precision",
                "enable_progress_bar",
                "limit_val_batches",
            ],
            "fit": [],
            "wrapper": [
                "n_bootstraps",
                "es_patience",
                "es_monitor",
                "es_mode",
                "es_min_delta",
                "es_verbose",
                "normalize",
            ],
        }
        self._update_acceptable_kwargs("model", extra_model_kwargs)
        self._update_acceptable_kwargs("data", extra_data_kwargs)
        self._update_acceptable_kwargs(
            "model", kwargs.pop("remove_model_kwargs", []), acceptable=False
        )
        self._update_acceptable_kwargs(
            "data", kwargs.pop("remove_data_kwargs", []), acceptable=False
        )

        # Convenience aliases handled at construction
        self.convenience_kwargs = [
            "alpha",
            "l1_ratio",
            "mu_ratio",
            "subtype_probabilities",
            "width",
            "layers",
            "encoder_link_fn",
        ]

        # Model constructor kwargs (with convenience mapping)
        self.constructor_kwargs = self._organize_constructor_kwargs(**kwargs)
        self.constructor_kwargs["encoder_kwargs"]["width"] = kwargs.pop(
            "width", self.constructor_kwargs["encoder_kwargs"]["width"]
        )
        self.constructor_kwargs["encoder_kwargs"]["layers"] = kwargs.pop(
            "layers", self.constructor_kwargs["encoder_kwargs"]["layers"]
        )
        self.constructor_kwargs["encoder_kwargs"]["link_fn"] = kwargs.pop(
            "encoder_link_fn",
            self.constructor_kwargs["encoder_kwargs"].get(
                "link_fn", self.default_encoder_link_fn
            ),
        )

        # Everything else
        self.not_constructor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self.constructor_kwargs and k not in self.convenience_kwargs
        }

        self._init_kwargs, unrecognized = self._organize_kwargs(
            **self.not_constructor_kwargs
        )
        for k, v in self.constructor_kwargs.items():
            self._init_kwargs["model"][k] = v
        for kw in unrecognized:
            print(f"Received unknown keyword argument {kw}, probably ignoring.")

    # -------------------- helpers --------------------

    def _is_gpu(self) -> bool:
        return self.accelerator in ("cuda", "gpu")

    def _update_acceptable_kwargs(self, category, new_kwargs, acceptable=True):
        if acceptable:
            self.acceptable_kwargs[category] = list(
                set(self.acceptable_kwargs[category]).union(set(new_kwargs))
            )
        else:
            self.acceptable_kwargs[category] = list(
                set(self.acceptable_kwargs[category]) - set(new_kwargs)
            )

    def _organize_kwargs(self, **kwargs):
        out = {cat: {} for cat in self.acceptable_kwargs}
        unknown = []
        for k, v in kwargs.items():
            placed = False
            for cat, allowed in self.acceptable_kwargs.items():
                if k in allowed:
                    out[cat][k] = v
                    placed = True
                    break
            if not placed:
                unknown.append(k)
        return out, unknown

    def _organize_constructor_kwargs(self, **kwargs):
        model = {}

        def maybe_add(kw, default_val):
            if kw in self.acceptable_kwargs["model"]:
                model[kw] = kwargs.get(kw, default_val)

        maybe_add("link_fn", LINK_FUNCTIONS["identity"])
        maybe_add("univariate", False)
        maybe_add("encoder_type", DEFAULT_ENCODER_TYPE)
        maybe_add("loss_fn", LOSSES["mse"])
        maybe_add(
            "encoder_kwargs",
            {
                "width": kwargs.get("encoder_width", DEFAULT_ENCODER_WIDTH),
                "layers": kwargs.get("encoder_layers", DEFAULT_ENCODER_LAYERS),
                "link_fn": kwargs.get("encoder_link_fn", DEFAULT_ENCODER_LINK_FN),
            },
        )
        if kwargs.get("subtype_probabilities", False):
            model["encoder_kwargs"]["link_fn"] = LINK_FUNCTIONS["softmax"]

        # Regularizer
        if "model_regularizer" in self.acceptable_kwargs["model"]:
            if kwargs.get("alpha", 0) > 0:
                model["model_regularizer"] = REGULARIZERS["l1_l2"](
                    kwargs["alpha"],
                    kwargs.get("l1_ratio", 1.0),
                    kwargs.get("mu_ratio", 0.5),
                )
            else:
                model["model_regularizer"] = kwargs.get(
                    "model_regularizer", REGULARIZERS["none"]
                )
        return model

    @staticmethod
    def _retarget_or_strip_early_stopping(cb, use_val: bool, train_monitor="train_loss"):
        try:
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping as _ES
        except Exception:
            return cb
        if not isinstance(cb, _ES):
            return cb
        if use_val:
            return cb
        monitor = getattr(cb, "monitor", None)
        if (monitor is None) or (isinstance(monitor, str) and monitor.startswith("val_")):
            return _ES(
                monitor=train_monitor,
                mode=getattr(cb, "mode", "min"),
                patience=getattr(cb, "patience", 1),
                verbose=getattr(cb, "verbose", False),
                min_delta=getattr(cb, "min_delta", 0.0),
            )
        return cb

    # -------------------- fit kwarg expansion --------------------
    def _organize_and_expand_fit_kwargs(self, **kwargs):
        """
        Expand/normalize kwargs for data/model/trainer/wrapper/fit, and build a clean
        configuration dict for downstream construction. Critically:
        • Merge constructor-time defaults BEFORE computing use_val.
        • Only add EarlyStopping if a val loop exists and patience > 0.
        • Retarget or strip EarlyStopping if no val loop.
        """
        organized, unrecognized = self._organize_kwargs(**kwargs)

        # -------- epochs (avoid PL default 1000) --------
        max_epochs_cli = kwargs.get("max_epochs", None)
        epochs_cli = kwargs.get("epochs", None)
        if max_epochs_cli is not None:
            organized["trainer"]["max_epochs"] = int(max_epochs_cli)
        elif epochs_cli is not None:
            organized["trainer"]["max_epochs"] = int(epochs_cli)
        else:
            organized["trainer"]["max_epochs"] = 3

        # -------- merge constructor defaults BEFORE using them --------
        for category, cat_kwargs in self._init_kwargs.items():
            for k, v in cat_kwargs.items():
                organized[category].setdefault(k, v)

        # -------- world size / validation decision --------
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        current_val_split = organized["data"].get("val_split", self.default_val_split)
        organized["data"]["val_split"] = current_val_split
        use_val = float(current_val_split) > 0.0

        # -------- trainer defaults --------
        organized["trainer"].setdefault("accelerator", self.accelerator)
        organized["trainer"].setdefault("enable_progress_bar", False)
        organized["trainer"].setdefault("logger", False)
        organized["trainer"].setdefault("enable_checkpointing", False)
        organized["trainer"].setdefault("num_sanity_val_steps", 0)
        organized["trainer"].setdefault("precision", 32)
        if not use_val:
            organized["trainer"].setdefault("limit_val_batches", 0)

        if world_size > 1:
            organized["trainer"].setdefault("devices", world_size)
            organized["trainer"].setdefault("strategy", "ddp")  # string to allow factory
        else:
            organized["trainer"]["devices"] = 1
            organized["trainer"].setdefault("strategy", "auto")
            organized["trainer"].setdefault("plugins", [LightningEnvironment()])

        # Helper to safely set defaults if the key is permitted for that category
        def maybe_add(cat, k, default):
            if k in self.acceptable_kwargs[cat]:
                organized[cat][k] = organized[cat].get(k, default)

        # -------- model defaults --------
        maybe_add("model", "learning_rate", self.default_learning_rate)
        maybe_add("model", "context_dim", self.context_dim)
        maybe_add("model", "x_dim", self.x_dim)
        maybe_add("model", "y_dim", self.y_dim)
        if organized["model"].get("num_archetypes", 1) == 0:
            organized["model"].pop("num_archetypes", None)

        # -------- data defaults (per-loader sizes) --------
        maybe_add("data", "train_batch_size", self.default_train_batch_size)
        maybe_add("data", "val_batch_size", self.default_val_batch_size)
        maybe_add("data", "test_batch_size", self.default_test_batch_size)
        maybe_add("data", "predict_batch_size", self.default_val_batch_size)
        maybe_add("data", "num_workers", 0)
        maybe_add("data", "pin_memory", self._is_gpu())
        maybe_add("data", "persistent_workers", organized["data"].get("num_workers", 0) > 0)
        maybe_add("data", "drop_last", False)
        maybe_add("data", "shuffle_train", True)
        maybe_add("data", "shuffle_eval", False)
        maybe_add("data", "dtype", torch.float)

        # -------- wrapper defaults --------
        maybe_add("wrapper", "n_bootstraps", self.default_n_bootstraps)

        # -------- EarlyStopping / Checkpoint constructors --------
        es_monitor = organized["wrapper"].get("es_monitor", "val_loss" if use_val else "train_loss")
        es_mode = organized["wrapper"].get("es_mode", "min")
        es_patience = organized["wrapper"].get("es_patience", self.default_es_patience)
        es_verbose = organized["wrapper"].get("es_verbose", False)
        es_min_delta = organized["wrapper"].get("es_min_delta", 0.0)

        cb_ctors = organized["trainer"].get("callback_constructors", [])

        # Only add EarlyStopping when there is a val loop AND patience > 0
        if use_val and (es_patience is not None and es_patience > 0):
            cb_ctors.append(
                lambda i: EarlyStopping(
                    monitor=es_monitor,
                    mode=es_mode,
                    patience=es_patience,
                    verbose=es_verbose,
                    min_delta=es_min_delta,
                )
            )

        if organized["trainer"].get("enable_checkpointing", False):
            cb_ctors.append(
                lambda i: ModelCheckpoint(
                    monitor=("val_loss" if use_val else None),
                    dirpath=f"{kwargs.get('checkpoint_path', './lightning_logs')}/boot_{i}_checkpoints",
                    filename=("{epoch}-{val_loss:.4f}" if use_val else "{epoch}"),
                )
            )
        organized["trainer"]["callback_constructors"] = cb_ctors

        # -------- unknown kw logging --------
        for kw in unrecognized:
            print(f"Received unknown keyword argument {kw}, probably ignoring.")

        # -------- sanitize any pre-specified callbacks for no-val runs --------
        cb_list = organized["trainer"].get("callbacks", [])
        cb_list = [self._retarget_or_strip_early_stopping(cb, use_val) for cb in cb_list]
        organized["trainer"]["callbacks"] = cb_list

        # Also sanitize dynamically constructed callbacks
        ctor_list = organized["trainer"].get("callback_constructors", [])
        def _wrap_ctor(ctor):
            def _wrapped(i):
                cb = ctor(i)
                return self._retarget_or_strip_early_stopping(cb, use_val)
            return _wrapped
        organized["trainer"]["callback_constructors"] = [_wrap_ctor(c) for c in ctor_list]

        return organized

    # -------------------- data module builder --------------------
    def _build_datamodule(
        self,
        C: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray],
        *,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        predict_idx=None,
        data_kwargs: Optional[dict] = None,
        task_type: str = "singletask_multivariate",
    ) -> ContextualizedRegressionDataModule:
        dk = dict(
            train_batch_size=self.default_train_batch_size,
            val_batch_size=self.default_val_batch_size,
            test_batch_size=self.default_test_batch_size,
            predict_batch_size=self.default_val_batch_size,
            num_workers=0,
            pin_memory=self._is_gpu(),
            persistent_workers=None,
            persistent_workers=False,
            drop_last=False,
            shuffle_train=True,
            shuffle_eval=False,
            dtype=torch.float,
        )
        if data_kwargs:
            dk.update(data_kwargs)

        # If not explicitly set, default to True when num_workers > 0
        if dk["persistent_workers"] is None:
            dk["persistent_workers"] = bool(dk["num_workers"] > 0)


        dm = ContextualizedRegressionDataModule(
            C=C,
            X=X,
            Y=Y,
            task_type=task_type,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            predict_idx=predict_idx,
            train_batch_size=dk["train_batch_size"],
            val_batch_size=dk["val_batch_size"],
            test_batch_size=dk["test_batch_size"],
            predict_batch_size=dk["predict_batch_size"],
            num_workers=dk["num_workers"],
            pin_memory=dk["pin_memory"],
            persistent_workers=dk["persistent_workers"],
            drop_last=dk["drop_last"],
            shuffle_train=dk["shuffle_train"],
            shuffle_eval=dk["shuffle_eval"],
            dtype=dk["dtype"],
        )
        dm.prepare_data()
        dm.setup()
        return dm

    # -------------------- split helpers --------------------
    def _split_train_data(
        self,
        C: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        *,
        Y_required: bool = True,
        val_split: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        **_,
    ):
        """
        Return (train_idx, val_idx) over rows; Lightning will attach DistributedSamplers.
        """
        if Y_required and Y is None:
            raise ValueError("Y is required but was not provided.")
        n = C.shape[0]
        vs = self.default_val_split if val_split is None else float(val_split)
        if vs <= 0.0:
            idx = np.arange(n)
            return idx, None
        tr_idx, va_idx = train_test_split(
            np.arange(n),
            test_size=vs,
            shuffle=shuffle,
            random_state=random_state,
        )
        return tr_idx, va_idx

    # -------------------- optional scaling --------------------
    def _maybe_scale_C(self, C: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["C"] is not None:
            return self.scalers["C"].transform(C)
        return C

    def _maybe_scale_X(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["X"] is not None:
            return self.scalers["X"].transform(X)
        return X

    # -------------------- public API --------------------
    def predict(self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs):
        if not hasattr(self, "models") or self.models is None:
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")

        Cq = self._maybe_scale_C(C)
        Xq = self._maybe_scale_X(X)
        Yq = np.zeros((len(Cq), self.y_dim), dtype=np.float32)

        preds = []
        for i in range(len(self.models)):
            dm = self._build_datamodule(
                C=Cq, X=Xq, Y=Yq,
                predict_idx=np.arange(len(Cq)),
                data_kwargs=dict(
                    train_batch_size=self._init_kwargs["data"].get("train_batch_size", self.default_train_batch_size),
                    val_batch_size=self._init_kwargs["data"].get("val_batch_size", self.default_val_batch_size),
                    test_batch_size=self._init_kwargs["data"].get("test_batch_size", self.default_test_batch_size),
                    predict_batch_size=self._init_kwargs["data"].get("predict_batch_size", self.default_val_batch_size),
                    num_workers=self._init_kwargs["data"].get("num_workers", 0),
                    pin_memory=self._init_kwargs["data"].get("pin_memory", self._is_gpu()),
                    persistent_workers=self._init_kwargs["data"].get("persistent_workers", False),
                    shuffle_train=False,
                    shuffle_eval=False,
                    dtype=self._init_kwargs["data"].get("dtype", torch.float),
                ),
                task_type="singletask_univariate" if self._init_kwargs["model"].get("univariate", False)
                        else "singletask_multivariate",
            )
            yhat = self.trainers[i].predict_y(self.models[i], dm.predict_dataloader(), **kwargs)
            preds.append(yhat)

        predictions = np.array(preds)
        if not individual_preds:
            predictions = np.mean(predictions, axis=0)
        if self.normalize and self.scalers["Y"] is not None:
            if individual_preds:
                predictions = np.array([self.scalers["Y"].inverse_transform(p) for p in predictions])
            else:
                predictions = self.scalers["Y"].inverse_transform(predictions)
        return predictions

    def predict_params(
        self,
        C: np.ndarray,
        individual_preds: bool = False,
        model_includes_mus: bool = True,
        **kwargs,
    ):
        if not hasattr(self, "models") or self.models is None:
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")

        Cq = self._maybe_scale_C(C)
        X_zero = np.zeros((len(Cq), self.x_dim), dtype=np.float32)
        Y_zero = np.zeros((len(Cq), self.y_dim), dtype=np.float32)

        out_betas, out_mus = [], []
        for i in range(len(self.models)):
            dm = self._build_datamodule(
                C=Cq,
                X=X_zero,
                Y=Y_zero if kwargs.pop("uses_y", True) else None,
                predict_idx=np.arange(len(Cq)),
                data_kwargs=dict(
                    train_batch_size=self._init_kwargs["data"].get("train_batch_size", self.default_train_batch_size),
                    val_batch_size=self._init_kwargs["data"].get("val_batch_size", self.default_val_batch_size),
                    test_batch_size=self._init_kwargs["data"].get("test_batch_size", self.default_test_batch_size),
                    predict_batch_size=self._init_kwargs["data"].get("predict_batch_size", self.default_val_batch_size),
                    num_workers=self._init_kwargs["data"].get("num_workers", 0),
                    pin_memory=self._init_kwargs["data"].get("pin_memory", self._is_gpu()),
                    persistent_workers=self._init_kwargs["data"].get("persistent_workers", False),
                    shuffle_train=False,
                    shuffle_eval=False,
                    dtype=self._init_kwargs["data"].get("dtype", torch.float),
                ),
                task_type="singletask_univariate" if self._init_kwargs["model"].get("univariate", False)
                        else "singletask_multivariate",
            )
            pred = self.trainers[i].predict_params(self.models[i], dm.predict_dataloader(), **kwargs)
            if model_includes_mus:
                out_betas.append(pred[0]); out_mus.append(pred[1])
            else:
                out_betas.append(pred)

        if model_includes_mus:
            betas = np.array(out_betas); mus = np.array(out_mus)
            return (betas, mus) if individual_preds else (np.mean(betas, axis=0), np.mean(mus, axis=0))
        else:
            betas = np.array(out_betas)
            return betas if individual_preds else np.mean(betas, axis=0)

    def fit(self, *args, **kwargs) -> None:
        """
        Fit contextualized model to data.

        Accepts either:
          - (C, X, Y)  [canonical order], OR
          - (X, Y, C)  [README order], OR
          - kw-only: C=..., X=..., (Y=...)
        """
        self.models, self.trainers = [], []

        # normalize argument order 
        C_in = kwargs.pop("C", None)
        X_in = kwargs.pop("X", None)
        Y_in = kwargs.pop("Y", None)

        if (C_in is not None) and (X_in is not None):
            C, X, Y = C_in, X_in, Y_in
        else:
            if len(args) == 3:
                A, B, Carg = args
                if A.shape[0] == B.shape[0] == Carg.shape[0]:
                    if (B.ndim == 1) or (B.ndim == 2 and B.shape[1] <= 4):
                        X, Y, C = A, B, Carg
                    else:
                        C, X, Y = A, B, Carg
                else:
                    raise ValueError("Mismatched sample counts among provided arrays.")
            elif len(args) == 2:
                A, B = args
                if A.shape[0] != B.shape[0]:
                    raise ValueError("Mismatched sample counts for two-argument fit.")
                # Assume (C, X) by default
                C, X, Y = A, B, None
            else:
                raise ValueError("fit expects (C,X[,Y]) or (X,Y,C) or kw-only C=..., X=...")

        # Optional scaling
        if self.normalize:
            if self.scalers["C"] is None: self.scalers["C"] = StandardScaler().fit(C)
            C = self.scalers["C"].transform(C)
            if self.scalers["X"] is None: self.scalers["X"] = StandardScaler().fit(X)
            X = self.scalers["X"].transform(X)

        self.context_dim = C.shape[-1]
        self.x_dim = X.shape[-1]

        if Y is not None:
            if len(Y.shape) == 1:
                Y = np.expand_dims(Y, 1)
            if self.normalize and not np.array_equal(np.unique(Y), np.array([0, 1])):
                if self.scalers["Y"] is None: self.scalers["Y"] = StandardScaler().fit(Y)
                Y = self.scalers["Y"].transform(Y)
            self.y_dim = Y.shape[-1]
            args = (C, X, Y)
        else:
            self.y_dim = self.x_dim
            args = (C, X)

        organized = self._organize_and_expand_fit_kwargs(**kwargs)
        self.n_bootstraps = organized["wrapper"].get("n_bootstraps", self.n_bootstraps)

        n = C.shape[0]
        val_split = organized["data"].get("val_split", self.default_val_split)
        use_val = val_split > 0.0

        for b in range(self.n_bootstraps):
            # Model (LightningModule)
            _model_kwargs = dict(organized["model"])
            _model_kwargs.pop("univariate", None)  # handled via task_type below
            model = self.base_constructor(**_model_kwargs)
            self.model_ = model

            # Indices
            train_idx, val_idx = self._split_train_data(
                C, X, (args[2] if len(args) == 3 else None),
                Y_required=(len(args) == 3),
                val_split=val_split,
            )
            test_idx = None

            # DataModule
            task_type = "singletask_univariate" if organized["model"].get("univariate", False) else "singletask_multivariate"
            dm = self._build_datamodule(
                C=args[0], X=args[1], Y=(args[2] if len(args) == 3 else None),
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                data_kwargs=dict(
                    train_batch_size=organized["data"].get("train_batch_size", self.default_train_batch_size),
                    val_batch_size=organized["data"].get("val_batch_size", self.default_val_batch_size),
                    test_batch_size=organized["data"].get("test_batch_size", self.default_test_batch_size),
                    predict_batch_size=organized["data"].get("predict_batch_size", self.default_val_batch_size),
                    num_workers=organized["data"].get("num_workers", 0),
                    pin_memory=organized["data"].get("pin_memory", self._is_gpu()),
                    persistent_workers=organized["data"].get("persistent_workers", organized["data"].get("num_workers", 0) > 0),
                    drop_last=organized["data"].get("drop_last", False),
                    shuffle_train=organized["data"].get("shuffle_train", True),
                    shuffle_eval=organized["data"].get("shuffle_eval", False),
                    dtype=organized["data"].get("dtype", torch.float),
                ),
                task_type=task_type,
            )

            # Trainer (fresh callbacks)
            trainer_kwargs = copy.deepcopy(organized["trainer"])
            trainer_kwargs["callbacks"] = [f(b) for f in trainer_kwargs.get("callback_constructors", [])]
            trainer_kwargs.pop("callback_constructors", None)

            # Build via factory (respects strategy strings and env)
            from contextualized.regression.trainers import make_trainer_with_env
            trainer = make_trainer_with_env(
                self.trainer_constructor,
                **trainer_kwargs,
            )

            # Ensure checkpoint dir if used
            for cb in trainer_kwargs.get("callbacks", []):
                if isinstance(cb, ModelCheckpoint):
                    os.makedirs(cb.dirpath, exist_ok=True)

            # Fit (omit val loader if no val split)
            if use_val and dm.val_dataloader() is not None:
                trainer.fit(
                    model,
                    train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader(),
                    **organized["fit"],
                )
            else:
                trainer.fit(
                    model,
                    train_dataloaders=dm.train_dataloader(),
                    **organized["fit"],
                )

            # Load best checkpoint if enabled
            if trainer_kwargs.get("enable_checkpointing", False):
                ckpt_cb = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
                if ckpt_cb and ckpt_cb.best_model_path and os.path.exists(ckpt_cb.best_model_path):
                    best = torch.load(ckpt_cb.best_model_path, map_location="cpu")
                    model.load_state_dict(best["state_dict"])

            self.models.append(model)
            self.trainers.append(trainer)
