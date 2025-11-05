"""Contextualized.ML: A statistical machine learning toolbox for estimating
	models, distributions, and functions with context-specific parameters.
	For more details, please refer to contextualized.ml.
"""
import torch

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")  # use TF32 kernels
    except Exception:
        pass
from contextualized import analysis
from contextualized import dags
from contextualized import easy
from contextualized import regression
from contextualized import baselines
from contextualized import utils
from contextualized.utils import *


import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# single-node default (disable IB unless you know you need it)
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
from .utils.engine import pick_engine  # optional re-export
__all__ = ["pick_engine"]
