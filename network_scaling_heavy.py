#!/usr/bin/env python3
"""
HEAVY ContextualizedCorrelationNetworks DDP Scaling Benchmark

This benchmark tests multi-GPU scaling with the actual ContextualizedCorrelationNetworks
model, but configured for maximum compute to properly stress-test GPU parallelism.

Key optimizations for heavier compute:
1. Larger encoder networks (more layers, wider hidden dims)
2. More archetypes (more mixture components to learn)
3. Multiple bootstraps (ensemble of models)
4. Larger batch sizes to saturate GPU memory
5. More training epochs
6. Increased data dimensionality (more PCs)

The goal is to make the model heavy enough that:
- Forward/backward pass takes significant time (50-200ms per batch)
- GPU compute dominates over NCCL sync overhead
- Multi-GPU scaling approaches theoretical limits (85-95% efficiency)

Usage:
  # 1-GPU baseline
  python ccn_scaling_heavy.py --epochs 20 --devices 1 --label 1gpu_baseline

  # Multi-GPU with torchrun
  torchrun --standalone --nproc_per_node=4 ccn_scaling_heavy.py --epochs 20 --label 4gpu_ddp
"""

import os
import time
import csv
import warnings
import pickle
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.distributed as dist

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from contextualized.easy import ContextualizedCorrelationNetworks


# ================= CONFIGURATION =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

PATH_L1000 = os.path.join(DATA_DIR, "trt_cp_smiles_qc.csv")
PATH_CTLS = os.path.join(DATA_DIR, "ctrls.csv")

# INCREASED: More PCs = larger feature space = more compute
N_DATA_PCS = 100  # Was 50
N_CONTEXT_PCS = 100  # Control profile PCs

PERTURBATION_HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)


# ================= DISTRIBUTED HELPERS =================

def is_global_zero() -> bool:
    """Return True only on global rank 0."""
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank() == 0
        except Exception:
            return True
    return int(os.environ.get("GLOBAL_RANK", os.environ.get("RANK", "0"))) == 0


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def print_rank0(msg: str):
    if is_global_zero():
        print(msg, flush=True)


# ================= ENVIRONMENT SETUP =================

def set_env_defaults():
    """Optimized environment for heavy CCN training."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cpu_count = os.cpu_count() or 8
    threads = max(1, cpu_count // max(world_size, 1))
    
    os.environ.setdefault("OMP_NUM_THREADS", str(min(threads, 4)))
    os.environ.setdefault("MKL_NUM_THREADS", str(min(threads, 4)))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # NCCL optimizations
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
    
    # PyTorch optimizations
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass
    
    # Deterministic seeds
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


# ================= FINGERPRINT HELPER =================

def smiles_to_morgan_fp(smiles: str) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(morgan_gen.GetOptions().fpSize, dtype=np.float32)
        fp = morgan_gen.GetFingerprint(mol)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(morgan_gen.GetOptions().fpSize, dtype=np.float32)


# ================= DATA LOADING WITH CACHE =================

def get_cache_path(subsample_fraction: Optional[float], n_data_pcs: int) -> str:
    """Generate cache path based on config."""
    suffix = f"_sub{subsample_fraction}" if subsample_fraction else ""
    suffix += f"_pcs{n_data_pcs}"
    return os.path.join(DATA_DIR, f"ccn_heavy_cache{suffix}.pkl")


def load_and_preprocess(
    subsample_fraction: Optional[float] = None,
    use_cache: bool = True,
    n_data_pcs: int = N_DATA_PCS,
    n_context_pcs: int = N_CONTEXT_PCS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data with configurable dimensionality.
    Higher dimensions = more compute in the model.
    """
    cache_path = get_cache_path(subsample_fraction, n_data_pcs)
    
    # Try cache
    if use_cache and os.path.exists(cache_path):
        print_rank0(f"[DATA] Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return (
            cached['C_train'], cached['X_train_norm'],
            cached['C_test'], cached['X_test_norm'],
            cached['cell_ids_train'], cached['cell_ids_test']
        )
    
    # Wait for rank 0 to create cache
    if not is_global_zero() and use_cache:
        wait_count = 0
        while not os.path.exists(cache_path) and wait_count < 600:
            time.sleep(1)
            wait_count += 1
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return (
                cached['C_train'], cached['X_train_norm'],
                cached['C_test'], cached['X_test_norm'],
                cached['cell_ids_train'], cached['cell_ids_test']
            )
    
    print_rank0(f"[DATA] Loading L1000 from {PATH_L1000}")
    df = pd.read_csv(PATH_L1000, engine="pyarrow")
    
    df = df[df["pert_type"].isin(["trt_cp"])]
    
    bad = (
        (df["distil_cc_q75"] < 0.2) |
        (df["distil_cc_q75"] == -666) |
        (df["distil_cc_q75"].isna()) |
        (df["pct_self_rank_q25"] > 5) |
        (df["pct_self_rank_q25"] == -666) |
        (df["pct_self_rank_q25"].isna())
    )
    df = df[~bad]
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"] != ""]
    
    print_rank0(f"[DATA] Samples after QC: {len(df)}")
    
    if subsample_fraction is not None:
        df = df.sample(frac=subsample_fraction, random_state=RANDOM_STATE)
        print_rank0(f"[DATA] Subsampled to {len(df)} ({subsample_fraction*100:.1f}%)")
    
    # Split by perturbation
    unique_smiles = df["canonical_smiles"].unique()
    print_rank0(f"[DATA] Unique perturbations: {len(unique_smiles)}")
    
    smiles_train, smiles_test = train_test_split(
        unique_smiles, test_size=PERTURBATION_HOLDOUT_SIZE, random_state=RANDOM_STATE
    )
    
    df_train = df[df["canonical_smiles"].isin(smiles_train)].copy()
    df_test = df[df["canonical_smiles"].isin(smiles_test)].copy()
    
    print_rank0(f"[DATA] Train: {len(df_train)}, Test: {len(df_test)}")
    
    # Handle missing values
    pert_time_mean = df_train.loc[df_train["pert_time"] != -666, "pert_time"].mean()
    pert_dose_mean = df_train.loc[df_train["pert_dose"] != -666, "pert_dose"].mean()
    
    for df_split in [df_train, df_test]:
        df_split["ignore_flag_pert_time"] = (df_split["pert_time"] == -666).astype(int)
        df_split["ignore_flag_pert_dose"] = (df_split["pert_dose"] == -666).astype(int)
        df_split["pert_time"] = df_split["pert_time"].replace(-666, pert_time_mean)
        df_split["pert_dose"] = df_split["pert_dose"].replace(-666, pert_dose_mean)
    
    def process_split(df_split, name):
        numeric_cols = df_split.select_dtypes(include=[np.number]).columns
        drop_cols = ["pert_dose", "pert_dose_unit", "pert_time", "distil_cc_q75", "pct_self_rank_q25"]
        feature_cols = [c for c in numeric_cols if c not in drop_cols]
        X_raw = df_split[feature_cols].values.astype(np.float32)
        
        print_rank0(f"[DATA] [{name}] Generating fingerprints...")
        fps = np.stack([smiles_to_morgan_fp(s) for s in df_split["canonical_smiles"]])
        print_rank0(f"[DATA] [{name}] Fingerprint shape: {fps.shape}")
        
        pert_time = df_split["pert_time"].to_numpy().reshape(-1, 1).astype(np.float32)
        pert_dose = df_split["pert_dose"].to_numpy().reshape(-1, 1).astype(np.float32)
        ign_t = df_split["ignore_flag_pert_time"].to_numpy().reshape(-1, 1).astype(np.float32)
        ign_d = df_split["ignore_flag_pert_dose"].to_numpy().reshape(-1, 1).astype(np.float32)
        
        return X_raw, fps, pert_time, pert_dose, ign_t, ign_d, df_split["cell_id"].to_numpy()
    
    X_train_raw, morgan_train, pt_train, pd_train, ign_t_train, ign_d_train, cells_train = process_split(df_train, "train")
    X_test_raw, morgan_test, pt_test, pd_test, ign_t_test, ign_d_test, cells_test = process_split(df_test, "test")
    
    # Scale features
    print_rank0("[DATA] Scaling gene expression...")
    scaler_genes = StandardScaler()
    X_train_scaled = scaler_genes.fit_transform(X_train_raw)
    X_test_scaled = scaler_genes.transform(X_test_raw)
    
    # Load controls
    print_rank0(f"[DATA] Loading controls from {PATH_CTLS}")
    ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)
    
    unique_cells = np.union1d(np.unique(cells_train), np.unique(cells_test))
    ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells)]
    
    scaler_ctrls = StandardScaler()
    ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)
    
    # INCREASED: More control PCs
    actual_n_ctrl_pcs = min(n_context_pcs, ctrls_scaled.shape[0], ctrls_scaled.shape[1])
    print_rank0(f"[DATA] Using {actual_n_ctrl_pcs} control PCs")
    
    pca_ctrls = PCA(n_components=actual_n_ctrl_pcs, random_state=RANDOM_STATE)
    ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)
    cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
    
    if not cell2vec:
        raise ValueError("No overlapping cell IDs")
    
    print_rank0(f"[DATA] Control embeddings for {len(cell2vec)} cells")
    
    def build_context(df_split, X_scaled, morgan, pt, pd, ign_t, ign_d, name, scaler=None, fit=False):
        cell_ids = df_split["cell_id"].to_numpy()
        unique_cells_split = np.sort(df_split["cell_id"].unique())
        
        all_cont = []
        valid_cells = []
        
        for cell_id in unique_cells_split:
            if cell_id not in cell2vec:
                continue
            mask = cell_ids == cell_id
            if mask.sum() == 0:
                continue
            valid_cells.append(cell_id)
            cont = np.hstack([
                np.tile(cell2vec[cell_id], (mask.sum(), 1)),
                pt[mask],
                pd[mask],
            ]).astype(np.float32)
            all_cont.append(cont)
        
        if fit:
            all_cont_stacked = np.vstack(all_cont)
            scaler = StandardScaler()
            scaler.fit(all_cont_stacked)
        
        X_list, C_list, cid_list = [], [], []
        
        for i, cell_id in enumerate(valid_cells):
            mask = cell_ids == cell_id
            X_cell = X_scaled[mask]
            cont_scaled = scaler.transform(all_cont[i])
            C_cell = np.hstack([
                cont_scaled,
                morgan[mask],
                ign_t[mask],
                ign_d[mask],
            ]).astype(np.float32)
            
            X_list.append(X_cell)
            C_list.append(C_cell)
            cid_list.append(cell_ids[mask])
        
        X_final = np.vstack(X_list)
        C_final = np.vstack(C_list)
        cell_ids_final = np.concatenate(cid_list)
        
        return X_final, C_final, cell_ids_final, scaler
    
    print_rank0("[DATA] Building context matrices...")
    X_train, C_train, cell_ids_train, ctx_scaler = build_context(
        df_train, X_train_scaled, morgan_train, pt_train, pd_train, ign_t_train, ign_d_train, "train", fit=True
    )
    X_test, C_test, cell_ids_test, _ = build_context(
        df_test, X_test_scaled, morgan_test, pt_test, pd_test, ign_t_test, ign_d_test, "test", scaler=ctx_scaler
    )
    
    print_rank0(f"[DATA] Context shapes: C_train={C_train.shape}, C_test={C_test.shape}")
    
    # INCREASED: More data PCs
    actual_n_data_pcs = min(n_data_pcs, X_train.shape[1], X_train.shape[0])
    print_rank0(f"[DATA] Using {actual_n_data_pcs} data PCs")
    
    pca_data = PCA(n_components=actual_n_data_pcs, random_state=RANDOM_STATE)
    X_train_pca = pca_data.fit_transform(X_train)
    X_test_pca = pca_data.transform(X_test)
    
    pca_scaler = StandardScaler()
    X_train_norm = pca_scaler.fit_transform(X_train_pca).astype(np.float32)
    X_test_norm = pca_scaler.transform(X_test_pca).astype(np.float32)
    
    print_rank0(f"[DATA] Final: X_train={X_train_norm.shape}, X_test={X_test_norm.shape}")
    print_rank0(f"[DATA] Final: C_train={C_train.shape}, C_test={C_test.shape}")
    
    # Save cache
    if use_cache and is_global_zero():
        cache_data = {
            'C_train': C_train, 'X_train_norm': X_train_norm,
            'C_test': C_test, 'X_test_norm': X_test_norm,
            'cell_ids_train': cell_ids_train, 'cell_ids_test': cell_ids_test,
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print_rank0(f"[DATA] Saved cache: {cache_path}")
    
    return C_train, X_train_norm, C_test, X_test_norm, cell_ids_train, cell_ids_test


# ================= BENCHMARK RESULT =================

@dataclass
class BenchResult:
    label: str
    wall_seconds: float
    train_mse_mean: float
    test_mse_mean: float
    num_gpus: int
    batch_size_per_gpu: int
    effective_batch_size: int
    samples_per_second: float
    num_archetypes: int
    encoder_width: int
    encoder_layers: int
    n_bootstraps: int
    speedup: float = 1.0
    efficiency: float = 100.0


# ================= MAIN BENCHMARK =================

def run_ccn_benchmark(
    label: str,
    C_train: np.ndarray,
    X_train_norm: np.ndarray,
    C_test: np.ndarray,
    X_test_norm: np.ndarray,
    epochs: int,
    devices: int,
    batch_size_per_gpu: int = 512,
    num_workers: int = 4,
    # Heavy CCN parameters
    num_archetypes: int = 64,
    encoder_width: int = 256,
    encoder_layers: int = 6,
    n_bootstraps: int = 3,
    warmup_epochs: int = 1,
    baseline_time: Optional[float] = None,
) -> BenchResult:
    """
    Run ContextualizedCorrelationNetworks benchmark with heavy configuration.
    
    Key parameters for increased compute:
    - num_archetypes: More mixture components (64 vs default 16)
    - encoder_width: Wider encoder networks (256 vs default 25)
    - encoder_layers: Deeper encoders (6 vs default 3)
    - n_bootstraps: Ensemble of models (3 vs default 1)
    """
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = get_rank()
    local_rank = get_local_rank()
    launched_with_torchrun = world_size > 1
    
    # Device setup
    if torch.cuda.is_available() and devices > 0:
        accelerator = "gpu"
        if launched_with_torchrun:
            devices = world_size
    else:
        accelerator = "cpu"
        devices = 1
        num_workers = 0
    
    # Reduce workers for multi-GPU
    if launched_with_torchrun and num_workers > 2:
        num_workers = 2
    
    # Batch size: scale with GPUs for proper throughput scaling
    effective_batch = batch_size_per_gpu * max(world_size, 1)
    
    print_rank0(f"\n{'='*70}")
    print_rank0(f"[{label}] HEAVY CCN BENCHMARK")
    print_rank0(f"{'='*70}")
    print_rank0(f"  World size: {world_size}")
    print_rank0(f"  Accelerator: {accelerator}")
    print_rank0(f"  Devices: {devices}")
    print_rank0(f"  Batch size per GPU: {batch_size_per_gpu}")
    print_rank0(f"  Effective batch size: {effective_batch}")
    print_rank0(f"  Epochs: {epochs} (+ {warmup_epochs} warmup)")
    print_rank0(f"  Num workers: {num_workers}")
    print_rank0(f"  --- CCN Config (HEAVY) ---")
    print_rank0(f"  Archetypes: {num_archetypes}")
    print_rank0(f"  Encoder width: {encoder_width}")
    print_rank0(f"  Encoder layers: {encoder_layers}")
    print_rank0(f"  Bootstraps: {n_bootstraps}")
    print_rank0(f"  Data dims: C={C_train.shape[1]}, X={X_train_norm.shape[1]}")
    
    # Log per-process info
    print(
        f"[{label}] [RANK {rank} / LOCAL {local_rank}] "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True
    )
    
    # Strategy configuration
    strategy_kwarg = "auto"
    if accelerator == "gpu" and launched_with_torchrun and world_size > 1:
        try:
            from pytorch_lightning.strategies import DDPStrategy
            strategy_kwarg = DDPStrategy(
                process_group_backend="nccl",
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            print_rank0(f"[{label}] Using DDPStrategy with NCCL + gradient_as_bucket_view")
        except Exception as e:
            strategy_kwarg = "ddp"
            print_rank0(f"[{label}] Falling back to strategy='ddp': {e}")
    
    # Trainer kwargs
    trainer_kwargs = {
        "max_epochs": epochs + warmup_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
        "num_sanity_val_steps": 0,
        "precision": "16-mixed" if accelerator == "gpu" else 32,
        "strategy": strategy_kwarg,
    }
    
    print_rank0(f"[{label}] Trainer kwargs: {trainer_kwargs}")
    
    # Construct HEAVY CCN model
    print_rank0(f"[{label}] Constructing ContextualizedCorrelationNetworks...")
    
    ccn = ContextualizedCorrelationNetworks(
        encoder_type="mlp",
        num_archetypes=num_archetypes,
        n_bootstraps=n_bootstraps,
        encoder_kwargs={
            "width": encoder_width,
            "layers": encoder_layers,
        },
        trainer_kwargs=trainer_kwargs,
        es_patience=0,  # No early stopping for benchmark
    )
    
    # Estimate parameter count
    # CCN params ≈ n_bootstraps × (encoder_params + archetype_params + correlation_params)
    # encoder_params ≈ (context_dim × width + width × width × (layers-1) + width × archetypes)
    # archetype_params ≈ archetypes × x_dim × x_dim (correlation matrices)
    context_dim = C_train.shape[1]
    x_dim = X_train_norm.shape[1]
    encoder_params = context_dim * encoder_width + encoder_width * encoder_width * (encoder_layers - 1) + encoder_width * num_archetypes
    archetype_params = num_archetypes * x_dim * x_dim
    total_params = n_bootstraps * (encoder_params + archetype_params)
    print_rank0(f"[{label}] Estimated parameters: ~{total_params:,} ({total_params/1e6:.2f}M)")
    
    # Synchronize before training
    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print_rank0(f"[{label}] Starting training...")
    t0 = time.time()
    
    ccn.fit(
        C_train,
        X_train_norm,
        train_batch_size=batch_size_per_gpu,
        val_batch_size=batch_size_per_gpu,
        test_batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=(accelerator == "gpu"),
    )
    
    # Synchronize after training
    barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    wall = time.time() - t0
    
    # Adjust for warmup
    if warmup_epochs > 0 and epochs > 0:
        wall_per_epoch = wall / (epochs + warmup_epochs)
        wall = wall_per_epoch * epochs
    
    print_rank0(f"[{label}] Training completed in {wall:.2f}s")
    
    # Metrics
    n_samples = C_train.shape[0]
    samples_per_sec = (n_samples * epochs) / max(wall, 1e-6)
    
    speedup = 1.0
    efficiency = 100.0
    if baseline_time is not None and baseline_time > 0:
        speedup = baseline_time / wall
        efficiency = (speedup / world_size) * 100
    
    train_mse = float("nan")
    test_mse = float("nan")
    
    if is_global_zero():
        try:
            print_rank0(f"[{label}] Computing MSE...")
            mse_train_vec = ccn.measure_mses(C_train, X_train_norm, individual_preds=False)
            mse_test_vec = ccn.measure_mses(C_test, X_test_norm, individual_preds=False)
            train_mse = float(np.mean(mse_train_vec))
            test_mse = float(np.mean(mse_test_vec))
        except Exception as e:
            warnings.warn(f"[{label}] measure_mses failed: {e}")
        
        print_rank0(f"\n[{label}] RESULTS:")
        print_rank0(f"  Wall time: {wall:.2f}s")
        print_rank0(f"  Samples/sec: {samples_per_sec:.1f}")
        print_rank0(f"  Train MSE: {train_mse:.6f}")
        print_rank0(f"  Test MSE: {test_mse:.6f}")
        if baseline_time:
            print_rank0(f"  Speedup: {speedup:.2f}x")
            print_rank0(f"  Efficiency: {efficiency:.1f}%")
    
    return BenchResult(
        label=label,
        wall_seconds=wall,
        train_mse_mean=train_mse,
        test_mse_mean=test_mse,
        num_gpus=world_size,
        batch_size_per_gpu=batch_size_per_gpu,
        effective_batch_size=effective_batch,
        samples_per_second=samples_per_sec,
        num_archetypes=num_archetypes,
        encoder_width=encoder_width,
        encoder_layers=encoder_layers,
        n_bootstraps=n_bootstraps,
        speedup=speedup,
        efficiency=efficiency,
    )


# ================= CSV OUTPUT =================

def save_results_csv(results: List[BenchResult], outdir: str):
    if not is_global_zero():
        return
    
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ccn_heavy_scaling_results.csv")
    
    write_header = not os.path.exists(path)
    
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "label", "wall_seconds", "train_mse", "test_mse",
                "num_gpus", "batch_per_gpu", "effective_batch", "samples_per_sec",
                "archetypes", "encoder_width", "encoder_layers", "bootstraps",
                "speedup", "efficiency"
            ])
        for r in results:
            writer.writerow([
                r.label,
                f"{r.wall_seconds:.4f}",
                f"{r.train_mse_mean:.6f}",
                f"{r.test_mse_mean:.6f}",
                r.num_gpus,
                r.batch_size_per_gpu,
                r.effective_batch_size,
                f"{r.samples_per_second:.2f}",
                r.num_archetypes,
                r.encoder_width,
                r.encoder_layers,
                r.n_bootstraps,
                f"{r.speedup:.2f}",
                f"{r.efficiency:.1f}",
            ])
    
    print_rank0(f"\n[OUTPUT] Results appended to: {path}")


# ================= CLI =================

def parse_args():
    import argparse
    
    ap = argparse.ArgumentParser(description="Heavy ContextualizedCorrelationNetworks Scaling Benchmark")
    
    # Training config
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512,
                    help="Batch size per GPU")
    ap.add_argument("--num-workers", type=int, default=4)
    
    # CCN architecture (HEAVY defaults)
    ap.add_argument("--archetypes", type=int, default=64,
                    help="Number of archetypes (default: 64, original: 16)")
    ap.add_argument("--encoder-width", type=int, default=256,
                    help="Encoder hidden width (default: 256, original: 25)")
    ap.add_argument("--encoder-layers", type=int, default=6,
                    help="Encoder depth (default: 6, original: 3)")
    ap.add_argument("--bootstraps", type=int, default=3,
                    help="Number of bootstrap models (default: 3, original: 1)")
    
    # Data config
    ap.add_argument("--data-pcs", type=int, default=100,
                    help="Number of data PCs (default: 100, original: 50)")
    ap.add_argument("--context-pcs", type=int, default=100,
                    help="Number of context PCs (default: 100)")
    ap.add_argument("--subsample-fraction", type=float, default=None)
    
    # Runtime config
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="bench_results_ccn_heavy")
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--baseline-time", type=float, default=None)
    ap.add_argument("--no-cache", action="store_true")
    
    return ap.parse_args()


# ================= MAIN =================

def main():
    args = parse_args()
    set_env_defaults()
    
    world_size = get_world_size()
    
    # Auto-generate label if not provided
    if args.label:
        label = args.label
    else:
        label = f"{world_size}gpu_ccn_heavy"
    
    print_rank0("\n" + "="*70)
    print_rank0("HEAVY ContextualizedCorrelationNetworks SCALING BENCHMARK")
    print_rank0("="*70)
    print_rank0(f"  World size: {world_size}")
    print_rank0(f"  Epochs: {args.epochs}")
    print_rank0(f"  Batch size: {args.batch_size}")
    print_rank0(f"  Archetypes: {args.archetypes}")
    print_rank0(f"  Encoder: {args.encoder_width}w × {args.encoder_layers}L")
    print_rank0(f"  Bootstraps: {args.bootstraps}")
    print_rank0(f"  Data PCs: {args.data_pcs}")
    
    # Load data
    C_train, X_train_norm, C_test, X_test_norm, _, _ = load_and_preprocess(
        subsample_fraction=args.subsample_fraction,
        use_cache=not args.no_cache,
        n_data_pcs=args.data_pcs,
        n_context_pcs=args.context_pcs,
    )
    
    barrier()
    
    # Run benchmark
    result = run_ccn_benchmark(
        label=label,
        C_train=C_train,
        X_train_norm=X_train_norm,
        C_test=C_test,
        X_test_norm=X_test_norm,
        epochs=args.epochs,
        devices=args.devices,
        batch_size_per_gpu=args.batch_size,
        num_workers=args.num_workers,
        num_archetypes=args.archetypes,
        encoder_width=args.encoder_width,
        encoder_layers=args.encoder_layers,
        n_bootstraps=args.bootstraps,
        warmup_epochs=args.warmup_epochs,
        baseline_time=args.baseline_time,
    )
    
    # Save results
    if is_global_zero():
        save_results_csv([result], args.outdir)


if __name__ == "__main__":
    main()