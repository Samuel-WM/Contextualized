# --- imports you need above the class ---
import copy
import os
from typing import *
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy

from contextualized.functions import LINK_FUNCTIONS
from contextualized.regression import REGULARIZERS, LOSSES
from contextualized.regression.datamodules import ContextualizedRegressionDataModule

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_BOOTSTRAPS = 1
DEFAULT_ES_PATIENCE = 1
DEFAULT_VAL_BATCH_SIZE = 16
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_ENCODER_TYPE = "mlp"
DEFAULT_ENCODER_WIDTH = 25
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_ENCODER_LINK_FN = LINK_FUNCTIONS["identity"]
DEFAULT_NORMALIZE = False


def _is_distributed() -> bool:
    """Check if we're in a distributed context."""
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    """Get current process rank."""
    if _is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return _get_rank() == 0

def _flatten_pl_predict_output(preds):
    """
    Lightning can return:
      - list[dict]  (single dataloader)
      - list[list[dict]] (multiple dataloaders)
    Normalize to list[dict].
    """
    if preds is None:
        return []
    if len(preds) > 0 and isinstance(preds[0], list):
        out = []
        for sub in preds:
            out.extend(sub)
        return out
    return preds


def _pack_local_pred_payload(pred_list: list) -> dict:
    """
    Convert list[dict] -> dict[str, np.ndarray] by concatenating along axis 0.
    Assumes each dict entry is either a torch.Tensor (CPU) or a Python scalar.
    """
    pred_list = _flatten_pl_predict_output(pred_list)
    if not pred_list:
        return {}

    # Union of keys across batches (some models include extra keys)
    keys = set()
    for d in pred_list:
        keys.update(d.keys())

    packed = {}
    for k in keys:
        chunks = []
        for d in pred_list:
            if k not in d:
                continue
            v = d[k]
            if torch.is_tensor(v):
                chunks.append(v.detach().cpu().numpy())
            else:
                chunks.append(np.asarray(v))
        if not chunks:
            continue
        # Concatenate on first dim where possible; fallback to stack
        try:
            packed[k] = np.concatenate(chunks, axis=0)
        except Exception:
            packed[k] = np.stack(chunks, axis=0)
    return packed


def _gather_object_to_rank0(obj):
    """
    Gather arbitrary Python objects to rank 0.
    Returns: list[obj] on rank 0, None on non-zero ranks.
    """
    if not _is_distributed():
        return [obj]

    world_size = dist.get_world_size()
    if world_size == 1:
        return [obj]

    if _is_main_process():
        gathered = [None for _ in range(world_size)]
        dist.gather_object(obj, object_gather_list=gathered, dst=0)
        return gathered
    else:
        dist.gather_object(obj, object_gather_list=None, dst=0)
        return None


def _merge_packed_payloads(payloads: list) -> dict:
    """
    Merge list[dict[str, np.ndarray]] -> dict[str, np.ndarray] by concatenation axis 0.
    """
    merged = {}
    if not payloads:
        return merged

    keys = set()
    for p in payloads:
        if p:
            keys.update(p.keys())

    for k in keys:
        chunks = [p[k] for p in payloads if p and (k in p) and (p[k] is not None) and (len(p[k]) > 0)]
        if not chunks:
            continue
        merged[k] = np.concatenate(chunks, axis=0)
    return merged


def _stable_sort_and_dedupe_by_key(payload: dict, primary: str, secondary: tuple = ()) -> dict:
    """
    Sort payload arrays by a composite key (primary + optional secondary indices),
    then dedupe (needed because DistributedSampler may pad/duplicate).
    """
    if (payload is None) or (primary not in payload) or (len(payload[primary]) == 0):
        return payload

    primary_arr = payload[primary].astype(np.int64)

    # Build composite key
    if secondary:
        parts = [primary_arr]
        for s in secondary:
            if s in payload:
                parts.append(payload[s].astype(np.int64))
        if len(parts) == 1:
            key = primary_arr
        else:
            # lexsort uses last key as primary; reverse order
            order = np.lexsort(tuple(reversed(parts)))
            key_sorted = np.stack([p[order] for p in parts], axis=1)
            # Dedup by full composite row
            _, uniq_pos = np.unique(key_sorted, axis=0, return_index=True)
            keep = order[np.sort(uniq_pos)]
    else:
        order = np.argsort(primary_arr, kind="mergesort")
        key_sorted = primary_arr[order]
        _, uniq_pos = np.unique(key_sorted, return_index=True)
        keep = order[np.sort(uniq_pos)]

    out = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and (v.shape[0] == primary_arr.shape[0]):
            out[k] = v[keep]
        else:
            out[k] = v
    return out



class SKLearnWrapper:
    """
    An sklearn-like wrapper for Contextualized models.
    
    FIXED VERSION with proper DDP handling for:
    - Prediction (avoids duplicate computation)
    - Data loading (proper num_workers)
    - Distributed inference
    """

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

        self._trainer_init_kwargs = kwargs.pop("trainer_kwargs", None)

        self.n_bootstraps = 1
        self.models = None
        self.trainers = None
        
        # Track if we trained with DDP (affects prediction strategy)
        self._trained_with_ddp = False
        self._trained_devices = 1

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

        self.convenience_kwargs = [
            "alpha",
            "l1_ratio",
            "mu_ratio",
            "subtype_probabilities",
            "width",
            "layers",
            "encoder_link_fn",
        ]

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

        if self._trainer_init_kwargs is not None:
            self._init_kwargs["trainer"].update(self._trainer_init_kwargs)

        for kw in unrecognized:
            print(f"Received unknown keyword argument {kw}, probably ignoring.")

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
    
    def _default_num_workers(self, devices: int) -> int:
        """
        Heuristic for default DataLoader workers.
        FIXED: CPU also benefits from workers for I/O overlap.
        """
        try:
            n_cpu = os.cpu_count() or 0
        except Exception:
            n_cpu = 0

        if n_cpu <= 0:
            return 0

        # For CPU-only, still use some workers for data loading overlap
        if self.accelerator not in ("cuda", "gpu"):
            return min(2, n_cpu)

        world_size_env = os.environ.get("WORLD_SIZE", None)
        if world_size_env is not None:
            try:
                world_size = max(1, int(world_size_env))
            except ValueError:
                world_size = 1
        else:
            world_size = max(1, devices)

        cpu_per_rank = max(1, n_cpu // world_size)
        # 2-4 workers per rank, capped
        return int(min(4, max(2, cpu_per_rank // 2)))

    def _organize_and_expand_fit_kwargs(self, **kwargs):
        """
        Expand/normalize kwargs for data/model/trainer/wrapper/fit.
        FIXED: Better DDP defaults and tracking.
        """
        organized, unrecognized = self._organize_kwargs(**kwargs)

        for category, cat_kwargs in self._init_kwargs.items():
            for k, v in cat_kwargs.items():
                organized[category].setdefault(k, v)

        max_epochs_cli = kwargs.get("max_epochs", None)
        epochs_cli = kwargs.get("epochs", None)
        if max_epochs_cli is not None:
            organized["trainer"]["max_epochs"] = int(max_epochs_cli)
        elif epochs_cli is not None:
            organized["trainer"]["max_epochs"] = int(epochs_cli)
        else:
            organized["trainer"].setdefault("max_epochs", 3)

        current_val_split = organized["data"].get("val_split", self.default_val_split)
        organized["data"]["val_split"] = current_val_split
        use_val = float(current_val_split) > 0.0

        organized["trainer"].setdefault("accelerator", self.accelerator)
        organized["trainer"].setdefault("enable_progress_bar", False)
        organized["trainer"].setdefault("logger", False)
        organized["trainer"].setdefault("enable_checkpointing", False)
        organized["trainer"].setdefault("num_sanity_val_steps", 0)
        
        # FIXED: Default to mixed precision on GPU
        if self.accelerator in ("cuda", "gpu"):
            organized["trainer"].setdefault("precision", "16-mixed")
        else:
            organized["trainer"].setdefault("precision", 32)

        if not use_val:
            organized["trainer"].setdefault("limit_val_batches", 0)

        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        if "devices" not in organized["trainer"]:
            # When torchrun is active, devices must match world_size
            organized["trainer"]["devices"] = world_size_env if world_size_env > 1 else 1

        devices_cfg = organized["trainer"].get("devices", 1)
        if isinstance(devices_cfg, int):
            devices = devices_cfg
        elif isinstance(devices_cfg, (list, tuple)):
            devices = len(devices_cfg)
        else:
            devices = 1
        
        # Validate: if torchrun sets WORLD_SIZE > 1, devices must match
        if world_size_env > 1 and devices != world_size_env:
            if _is_main_process():
                print(f"[WARNING] torchrun WORLD_SIZE={world_size_env} but devices={devices}. "
                      f"Overriding devices to {world_size_env}.")
            devices = world_size_env
            organized["trainer"]["devices"] = devices

        # Track for prediction strategy
        self._trained_devices = devices
        self._trained_with_ddp = devices > 1

        if "strategy" not in organized["trainer"]:
            if devices > 1 or world_size_env > 1:
                from datetime import timedelta
                # Check if we're under torchrun (process group may already exist)
                if world_size_env > 1:
                    # torchrun case: let Lightning use existing process group
                    organized["trainer"]["strategy"] = "ddp"
                else:
                    # Lightning-spawned DDP case
                    organized["trainer"]["strategy"] = DDPStrategy(
                        process_group_backend="nccl" if torch.cuda.is_available() else "gloo",
                        find_unused_parameters=False,
                        broadcast_buffers=False,
                        timeout=timedelta(minutes=30),
                    )
            else:
                organized["trainer"]["strategy"] = "auto"

        if (
            organized["trainer"].get("strategy") in ("auto", None)
            and organized["trainer"].get("devices", 1) == 1
            and world_size_env == 1  # Not under torchrun
            and "plugins" not in organized["trainer"]
        ):
            organized["trainer"]["plugins"] = [LightningEnvironment()]

        def maybe_add(cat, k, default):
            if k in self.acceptable_kwargs[cat]:
                organized[cat][k] = organized[cat].get(k, default)

        maybe_add("model", "learning_rate", self.default_learning_rate)
        maybe_add("model", "context_dim", self.context_dim)
        maybe_add("model", "x_dim", self.x_dim)
        maybe_add("model", "y_dim", self.y_dim)
        if organized["model"].get("num_archetypes", 1) == 0:
            organized["model"].pop("num_archetypes", None)

        maybe_add("data", "train_batch_size", self.default_train_batch_size)
        maybe_add("data", "val_batch_size", self.default_val_batch_size)
        maybe_add("data", "test_batch_size", self.default_test_batch_size)
        maybe_add("data", "predict_batch_size", self.default_val_batch_size)

        # FIXED: Better num_workers default
        default_nw = self._default_num_workers(devices)
        maybe_add("data", "num_workers", default_nw)

        maybe_add("data", "pin_memory", self.accelerator in ("cuda", "gpu"))

        persistent_default = organized["data"].get("num_workers", 0) > 0
        maybe_add("data", "persistent_workers", persistent_default)

        drop_last_default = devices > 1
        maybe_add("data", "drop_last", drop_last_default)

        maybe_add("data", "shuffle_train", True)
        maybe_add("data", "shuffle_eval", False)
        maybe_add("data", "dtype", torch.float)

        maybe_add("wrapper", "n_bootstraps", self.default_n_bootstraps)

        es_monitor = organized["wrapper"].get("es_monitor", "val_loss" if use_val else "train_loss")
        es_mode = organized["wrapper"].get("es_mode", "min")
        es_patience = organized["wrapper"].get("es_patience", self.default_es_patience)
        es_verbose = organized["wrapper"].get("es_verbose", False)
        es_min_delta = organized["wrapper"].get("es_min_delta", 0.0)

        cb_ctors = organized["trainer"].get("callback_constructors", [])

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

        for kw in unrecognized:
            print(f"Received unknown keyword argument {kw}, probably ignoring.")

        cb_list = organized["trainer"].get("callbacks", [])
        cb_list = [self._retarget_or_strip_early_stopping(cb, use_val) for cb in cb_list]
        organized["trainer"]["callbacks"] = cb_list

        ctor_list = organized["trainer"].get("callback_constructors", [])

        def _wrap_ctor(ctor):
            def _wrapped(i):
                cb = ctor(i)
                return self._retarget_or_strip_early_stopping(cb, use_val)
            return _wrapped

        organized["trainer"]["callback_constructors"] = [_wrap_ctor(c) for c in ctor_list]

        return organized

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
            pin_memory=(self.accelerator in ("cuda", "gpu")),
            persistent_workers=False,
            drop_last=False,
            shuffle_train=True,
            shuffle_eval=False,
            dtype=torch.float,
        )
        if data_kwargs:
            dk.update(data_kwargs)

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
        return dm

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
        """Return (train_idx, val_idx) over rows."""
        if Y_required and Y is None:
            raise ValueError("Y is required but was not provided.")
        n = C.shape[0]
        vs = self.default_val_split if val_split is None else float(val_split)
        if vs <= 0.0:
            idx = np.arange(n)
            return idx, None
        
        # FIXED: Handle small datasets
        min_val_samples = max(1, int(n * vs))
        if min_val_samples < 2:
            # Too small for validation split
            idx = np.arange(n)
            return idx, None
        
        # CRITICAL FIX: Use deterministic random_state for DDP
        # All ranks MUST get the same train/val split
        if random_state is None:
            random_state = 42  # Fixed seed for reproducibility across ranks
            
        tr_idx, va_idx = train_test_split(
            np.arange(n),
            test_size=vs,
            shuffle=shuffle,
            random_state=random_state,
        )
        return tr_idx, va_idx

    def _maybe_scale_C(self, C: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["C"] is not None:
            return self.scalers["C"].transform(C)
        return C

    def _maybe_scale_X(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["X"] is not None:
            return self.scalers["X"].transform(X)
        return X

    def _get_inference_device(self) -> torch.device:
        """
        Get the device to use for inference.
        FIXED: Always use single device for prediction to avoid DDP complexity.
        """
        if self.accelerator in ("cuda", "gpu") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def predict(self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs):
        if not hasattr(self, "models") or self.models is None:
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")

        Cq = self._maybe_scale_C(C)
        Xq = self._maybe_scale_X(X)
        Yq = np.zeros((len(Cq), self.y_dim), dtype=np.float32)

        dm = self._build_datamodule(
            C=Cq,
            X=Xq,
            Y=Yq,
            predict_idx=np.arange(len(Cq)),
            data_kwargs=dict(
                train_batch_size=self._init_kwargs["data"].get("train_batch_size", self.default_train_batch_size),
                val_batch_size=self._init_kwargs["data"].get("val_batch_size", self.default_val_batch_size),
                test_batch_size=self._init_kwargs["data"].get("test_batch_size", self.default_test_batch_size),
                predict_batch_size=self._init_kwargs["data"].get("predict_batch_size", self.default_val_batch_size),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                shuffle_train=False,
                shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate" if self._init_kwargs["model"].get("univariate", False)
                    else "singletask_multivariate",
        )

        # Let Lightning handle sharding under DDP
        preds = []
        n_expected = len(Cq)

        for i in range(len(self.models)):
            model = self.models[i]
            model.eval()

            # Prefer the trainer created during fit (keeps strategy/devices consistent)
            trainer = None
            if hasattr(self, "trainers") and self.trainers is not None and i < len(self.trainers):
                trainer = self.trainers[i]

            if _is_distributed() and trainer is not None:
                # ---- DDP path: use trainer.predict + gather outputs to rank 0 ----
                local_pred = trainer.predict(model, datamodule=dm)

                local_packed = _pack_local_pred_payload(local_pred)
                gathered = _gather_object_to_rank0(local_packed)

                if not _is_main_process():
                    # Non-zero ranks return nothing; rank 0 will return the final answer.
                    return None

                merged = _merge_packed_payloads(gathered)

                # Sort/dedupe by orig_idx (DistributedSampler may pad)
                merged = _stable_sort_and_dedupe_by_key(merged, primary="orig_idx")

                if "betas" not in merged or "mus" not in merged or "orig_idx" not in merged:
                    raise RuntimeError("predict: Missing required keys in gathered payload: need orig_idx, betas, mus.")

                orig_idx = merged["orig_idx"].astype(np.int64)
                betas = torch.as_tensor(merged["betas"])
                mus = torch.as_tensor(merged["mus"])

                # Ensure we are aligned to query order
                # (orig_idx is row-id into the query arrays because predict_idx=np.arange(n))
                C_sorted = torch.as_tensor(Cq[orig_idx], dtype=betas.dtype)
                X_sorted = torch.as_tensor(Xq[orig_idx], dtype=betas.dtype)

                # Compute yhat on rank 0 in correct global order
                with torch.no_grad():
                    yhat = model._predict_y(C_sorted, X_sorted, betas, mus).detach().cpu().numpy()

                # If DDP padded, we may have > n_expected; trim safely by orig_idx range
                # (should not happen if orig_idx is in [0, n_expected))
                if yhat.shape[0] != n_expected:
                    # Build dense output in original query order
                    dense = np.zeros((n_expected,) + yhat.shape[1:], dtype=yhat.dtype)
                    dense[orig_idx] = yhat
                    yhat = dense

                preds.append(yhat)

            else:
                # ---- Single-process fallback: iterate predict_dataloader directly ----
                dm.setup(stage="predict")
                pred_loader = dm.predict_dataloader()

                out_batches = []
                device = self._get_inference_device()
                model.to(device)

                with torch.no_grad():
                    for b_idx, batch in enumerate(pred_loader):
                        batch = {
                            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in batch.items()
                        }

                        out = model.predict_step(batch, b_idx)
                        betas = out["betas"]
                        mus = out["mus"]

                        # IMPORTANT: use the *batch* for C/X, not the output payload
                        yb = model._predict_y(batch["contexts"], batch["predictors"], betas, mus)
                        out_batches.append(yb.detach().cpu())

                yhat = torch.cat(out_batches, dim=0).numpy()
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

        uses_y = kwargs.pop("uses_y", True)

        dm = self._build_datamodule(
            C=Cq,
            X=X_zero,
            Y=Y_zero if uses_y else None,
            predict_idx=np.arange(len(Cq)),
            data_kwargs=dict(
                train_batch_size=self._init_kwargs["data"].get("train_batch_size", self.default_train_batch_size),
                val_batch_size=self._init_kwargs["data"].get("val_batch_size", self.default_val_batch_size),
                test_batch_size=self._init_kwargs["data"].get("test_batch_size", self.default_test_batch_size),
                predict_batch_size=self._init_kwargs["data"].get("predict_batch_size", self.default_val_batch_size),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                shuffle_train=False,
                shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate" if self._init_kwargs["model"].get("univariate", False)
                    else "singletask_multivariate",
        )

        out_betas, out_mus = [], []
        n_expected = len(Cq)

        for i in range(len(self.models)):
            model = self.models[i]
            model.eval()

            trainer = None
            if hasattr(self, "trainers") and self.trainers is not None and i < len(self.trainers):
                trainer = self.trainers[i]

            if _is_distributed() and trainer is not None:
                local_pred = trainer.predict(model, datamodule=dm)
                local_packed = _pack_local_pred_payload(local_pred)
                gathered = _gather_object_to_rank0(local_packed)

                if not _is_main_process():
                    return (None, None) if model_includes_mus else None


                merged = _merge_packed_payloads(gathered)
                merged = _stable_sort_and_dedupe_by_key(merged, primary="orig_idx")

                if "betas" not in merged or "orig_idx" not in merged:
                    raise RuntimeError("predict_params: Missing required keys in gathered payload: need orig_idx, betas.")

                orig_idx = merged["orig_idx"].astype(np.int64)

                betas_i = merged["betas"]
                if betas_i.shape[0] != n_expected:
                    dense_b = np.zeros((n_expected,) + betas_i.shape[1:], dtype=betas_i.dtype)
                    dense_b[orig_idx] = betas_i
                    betas_i = dense_b

                out_betas.append(betas_i)

                if model_includes_mus:
                    if "mus" not in merged:
                        raise RuntimeError("predict_params: model_includes_mus=True but mus missing in payload.")
                    mus_i = merged["mus"]
                    if mus_i.shape[0] != n_expected:
                        dense_m = np.zeros((n_expected,) + mus_i.shape[1:], dtype=mus_i.dtype)
                        dense_m[orig_idx] = mus_i
                        mus_i = dense_m
                    out_mus.append(mus_i)

            else:
                # Single-process fallback (local ordered)
                dm.setup(stage="predict")
                pred_loader = dm.predict_dataloader()

                device = self._get_inference_device()
                model.to(device)

                beta_batches, mu_batches = [], []
                with torch.no_grad():
                    for b_idx, batch in enumerate(pred_loader):
                        batch = {
                            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in batch.items()
                        }
                        out = model.predict_step(batch, b_idx)
                        beta_batches.append(out["betas"].detach().cpu())
                        if model_includes_mus:
                            mu_batches.append(out["mus"].detach().cpu())

                betas_i = torch.cat(beta_batches, dim=0).numpy()
                out_betas.append(betas_i)

                if model_includes_mus:
                    mus_i = torch.cat(mu_batches, dim=0).numpy()
                    out_mus.append(mus_i)

        betas = np.array(out_betas)
        if model_includes_mus:
            mus = np.array(out_mus)
            return (betas, mus) if individual_preds else (np.mean(betas, axis=0), np.mean(mus, axis=0))

        return betas if individual_preds else np.mean(betas, axis=0)


    def fit(self, *args, **kwargs) -> None:
        """
        Fit contextualized model to data.
        FIXED: Proper DDP handling and device tracking.
        """
        self.models, self.trainers = [], []

        # Normalize argument order 
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
                C, X, Y = A, B, None
            else:
                raise ValueError("fit expects (C,X[,Y]) or (X,Y,C) or kw-only C=..., X=...")

        # Optional scaling
        if self.normalize:
            if self.scalers["C"] is None:
                self.scalers["C"] = StandardScaler().fit(C)
            C = self.scalers["C"].transform(C)
            if self.scalers["X"] is None:
                self.scalers["X"] = StandardScaler().fit(X)
            X = self.scalers["X"].transform(X)

        self.context_dim = C.shape[-1]
        self.x_dim = X.shape[-1]

        if Y is not None:
            if len(Y.shape) == 1:
                Y = np.expand_dims(Y, 1)
            if self.normalize and not np.array_equal(np.unique(Y), np.array([0, 1])):
                if self.scalers["Y"] is None:
                    self.scalers["Y"] = StandardScaler().fit(Y)
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
            # Model
            _model_kwargs = dict(organized["model"])
            _model_kwargs.pop("univariate", None)
            model = self.base_constructor(**_model_kwargs)
            self.model_ = model

            # Indices
            train_idx, val_idx = self._split_train_data(
                C, X, (args[2] if len(args) == 3 else None),
                Y_required=(len(args) == 3),
                val_split=val_split,
            )
            print(f"[RANK {os.environ.get('RANK', 0)}] train_idx[:5]={train_idx[:5]}, val_idx[:5]={val_idx[:5] if val_idx is not None else None}")

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
                    pin_memory=organized["data"].get("pin_memory", self.accelerator in ("cuda", "gpu")),
                    persistent_workers=organized["data"].get("persistent_workers", False),
                    drop_last=organized["data"].get("drop_last", False),
                    shuffle_train=organized["data"].get("shuffle_train", True),
                    shuffle_eval=organized["data"].get("shuffle_eval", False),
                    dtype=organized["data"].get("dtype", torch.float),
                ),
                task_type=task_type,
            )

            # Trainer
            trainer_kwargs = copy.deepcopy(organized["trainer"])
            trainer_kwargs["callbacks"] = [f(b) for f in trainer_kwargs.get("callback_constructors", [])]
            trainer_kwargs.pop("callback_constructors", None)

            from contextualized.regression.trainers import make_trainer_with_env
            trainer = make_trainer_with_env(
                self.trainer_constructor,
                **trainer_kwargs,
            )

            for cb in trainer_kwargs.get("callbacks", []):
                if isinstance(cb, ModelCheckpoint):
                    os.makedirs(cb.dirpath, exist_ok=True)

            # Ensure all ranks have setup data before training
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Fit
            trainer.fit(
                model,
                datamodule=dm,
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