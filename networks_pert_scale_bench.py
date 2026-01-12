#!/usr/bin/env python3
# Benchmark script that preprocesses unseen_pert data and compares 1-GPU training vs 2-GPU DDP training for a simple MLP regressor.

import os
import time
import csv
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# Paths and basic config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

PATH_L1000 = os.path.join(DATA_DIR, "trt_cp_smiles_qc.csv")
PATH_CTLS = os.path.join(DATA_DIR, "ctrls.csv")

N_DATA_PCS = 50
PERTURBATION_HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)


# Environment and RNG seeding
def set_env_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def set_seeds(rank: int):
    np.random.seed(RANDOM_STATE + rank)
    torch.manual_seed(RANDOM_STATE + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE + rank)


# Fingerprint helper
def smiles_to_morgan_fp(smiles: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Invalid SMILES: {smiles}")
            return np.zeros(morgan_gen.GetOptions().fpSize, dtype=np.float32)
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.array(fp, dtype=np.float32)
        return arr
    except Exception as e:
        warnings.warn(f"Error processing SMILES {smiles}: {e}")
        return np.zeros(morgan_gen.GetOptions().fpSize, dtype=np.float32)


# Data preprocessing for unseen_pert
def load_and_preprocess(
    subsample_fraction: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"Reading L1000 data from {PATH_L1000}")
    df = pd.read_csv(PATH_L1000, engine="pyarrow")

    df = df[df["pert_type"].isin(["trt_cp"])]

    bad = (
        (df["distil_cc_q75"] < 0.2)
        | (df["distil_cc_q75"] == -666)
        | (df["distil_cc_q75"].isna())
        | (df["pct_self_rank_q25"] > 5)
        | (df["pct_self_rank_q25"] == -666)
        | (df["pct_self_rank_q25"].isna())
    )
    df = df[~bad]

    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"] != ""]

    print(f"Remaining samples after QC + SMILES filter: {len(df)}")

    if subsample_fraction is not None:
        df = df.sample(frac=subsample_fraction, random_state=RANDOM_STATE)
        print(f"Subsampled to {len(df)} samples ({subsample_fraction * 100:.1f}% of data)")

    unique_smiles = df["canonical_smiles"].unique()
    print(f"Found {len(unique_smiles)} unique perturbations (SMILES)")
    smiles_train, smiles_test = train_test_split(
        unique_smiles,
        test_size=PERTURBATION_HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )

    df_train = df[df["canonical_smiles"].isin(smiles_train)].copy()
    df_test = df[df["canonical_smiles"].isin(smiles_test)].copy()

    print(f"Perturbation split: {len(smiles_train)} train, {len(smiles_test)} test perturbations")
    print(f"Sample split: {len(df_train)} train, {len(df_test)} test samples")

    pert_time_mean = None
    pert_dose_mean = None

    for df_split, split_name in ((df_train, "train"), (df_test, "test")):
        df_split["ignore_flag_pert_time"] = (df_split["pert_time"] == -666).astype(int)
        df_split["ignore_flag_pert_dose"] = (df_split["pert_dose"] == -666).astype(int)

        for col in ["pert_time", "pert_dose"]:
            if split_name == "train":
                mean_val = df_split.loc[df_split[col] != -666, col].mean()
                if col == "pert_time":
                    pert_time_mean = mean_val
                else:
                    pert_dose_mean = mean_val
            else:
                mean_val = pert_time_mean if col == "pert_time" else pert_dose_mean

            df_split[col] = df_split[col].replace(-666, mean_val)

    def process_data_split(df_split, split_name):
        numeric_cols = df_split.select_dtypes(include=[np.number]).columns
        drop_cols = [
            "pert_dose",
            "pert_dose_unit",
            "pert_time",
            "distil_cc_q75",
            "pct_self_rank_q25",
        ]
        feature_cols = [c for c in numeric_cols if c not in drop_cols]
        X_raw = df_split[feature_cols].values.astype(np.float32)

        print(f"[{split_name}] Generating Morgan fingerprints...")
        fps = np.stack([smiles_to_morgan_fp(s) for s in df_split["canonical_smiles"]])
        print(f"[{split_name}] Morgan shape: {fps.shape}")

        pert_time = df_split["pert_time"].to_numpy().reshape(-1, 1).astype(np.float32)
        pert_dose = df_split["pert_dose"].to_numpy().reshape(-1, 1).astype(np.float32)
        ignore_time = df_split["ignore_flag_pert_time"].to_numpy().reshape(-1, 1).astype(np.float32)
        ignore_dose = df_split["ignore_flag_pert_dose"].to_numpy().reshape(-1, 1).astype(np.float32)

        return X_raw, fps, pert_time, pert_dose, ignore_time, ignore_dose

    (X_raw_train, morgan_train, pt_train, pd_train, ign_t_train, ign_d_train) = process_data_split(
        df_train, "train"
    )
    (X_raw_test, morgan_test, pt_test, pd_test, ign_t_test, ign_d_test) = process_data_split(
        df_test, "test"
    )

    print("Scaling gene expression...")
    scaler_genes = StandardScaler()
    X_train_scaled = scaler_genes.fit_transform(X_raw_train)
    X_test_scaled = scaler_genes.transform(X_raw_test)

    morgan_train_scaled = morgan_train.astype(np.float32)
    morgan_test_scaled = morgan_test.astype(np.float32)

    print(f"Reading control profiles from {PATH_CTLS}")
    ctrls_df = pd.read_csv(PATH_CTLS, index_col=0)

    unique_cells_train = np.sort(df_train["cell_id"].unique())
    unique_cells_test = np.sort(df_test["cell_id"].unique())
    unique_cells_all = np.sort(np.union1d(unique_cells_train, unique_cells_test))

    ctrls_df = ctrls_df.loc[ctrls_df.index.intersection(unique_cells_all)]
    scaler_ctrls = StandardScaler()
    ctrls_scaled = scaler_ctrls.fit_transform(ctrls_df.values)

    n_cells = ctrls_scaled.shape[0]
    n_ctrl_pcs = min(50, n_cells)

    pca_ctrls = PCA(n_components=n_ctrl_pcs, random_state=RANDOM_STATE)
    ctrls_pcs = pca_ctrls.fit_transform(ctrls_scaled)

    cell2vec = dict(zip(ctrls_df.index, ctrls_pcs))
    if not cell2vec:
        raise ValueError("No overlapping cell IDs between L1000 and ctrls.csv")

    print(f"Control embeddings for {len(cell2vec)} cells (PCs={n_ctrl_pcs})")

    def build_context_matrix(
        df_split,
        X_scaled,
        morgan_scaled,
        pt,
        pd,
        ign_t,
        ign_d,
        split_name,
        scaler_context=None,
        is_train=False,
    ):
        cell_ids = df_split["cell_id"].to_numpy()
        unique_cells_split = np.sort(df_split["cell_id"].unique())

        all_continuous_context = []
        valid_cells = []

        for cell_id in unique_cells_split:
            if cell_id not in cell2vec:
                print(f"[{split_name}] Warning: cell {cell_id} not in control embeddings; skipping")
                continue
            mask = cell_ids == cell_id
            if mask.sum() == 0:
                continue

            valid_cells.append(cell_id)
            cont = np.hstack(
                [
                    np.tile(cell2vec[cell_id], (mask.sum(), 1)),
                    pt[mask],
                    pd[mask],
                ]
            ).astype(np.float32)
            all_continuous_context.append(cont)

        if is_train:
            all_cont = np.vstack(all_continuous_context)
            scaler_context = StandardScaler()
            scaler_context.fit(all_cont)
            print(f"[{split_name}] Context scaler fit on {all_cont.shape} continuous features")

        if scaler_context is None:
            raise ValueError("scaler_context must be provided for non-training split")

        X_list, C_list, cid_list = [], [], []

        for i, cell_id in enumerate(valid_cells):
            mask = cell_ids == cell_id
            X_cell = X_scaled[mask]
            cont_scaled = scaler_context.transform(all_continuous_context[i])
            C_cell = np.hstack(
                [
                    cont_scaled,
                    morgan_scaled[mask],
                    ign_t[mask],
                    ign_d[mask],
                ]
            ).astype(np.float32)

            X_list.append(X_cell)
            C_list.append(C_cell)
            cid_list.append(cell_ids[mask])

        if not X_list:
            raise RuntimeError(f"No data for split {split_name}")

        X_final = np.vstack(X_list)
        C_final = np.vstack(C_list)
        cell_ids_final = np.concatenate(cid_list)

        return X_final, C_final, cell_ids_final, scaler_context

    print("Building context matrices...")
    X_train, C_train, cell_ids_train, scaler_context = build_context_matrix(
        df_train,
        X_train_scaled,
        morgan_train_scaled,
        pt_train,
        pd_train,
        ign_t_train,
        ign_d_train,
        "train",
        is_train=True,
    )
    X_test, C_test, cell_ids_test, _ = build_context_matrix(
        df_test,
        X_test_scaled,
        morgan_test_scaled,
        pt_test,
        pd_test,
        ign_t_test,
        ign_d_test,
        "test",
        scaler_context=scaler_context,
        is_train=False,
    )

    print(f"C_train: {C_train.shape}, X_train: {X_train.shape}")
    print(f"C_test:  {C_test.shape}, X_test:  {X_test.shape}")

    print("PCA + scaling on gene features...")
    pca_data = PCA(n_components=N_DATA_PCS, random_state=RANDOM_STATE)
    X_train_pca = pca_data.fit_transform(X_train)
    X_test_pca = pca_data.transform(X_test)

    pca_scaler = StandardScaler()
    X_train_norm = pca_scaler.fit_transform(X_train_pca)
    X_test_norm = pca_scaler.transform(X_test_pca)

    print(f"Final X_train_norm: {X_train_norm.shape}, X_test_norm: {X_test_norm.shape}")

    return C_train, X_train_norm, C_test, X_test_norm, cell_ids_train, cell_ids_test


@dataclass
class BenchResult:
    label: str
    wall_seconds: float
    samples_total: int
    throughput_sps: float
    train_mse_mean: float
    test_mse_mean: float


class SimpleRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_single_gpu(
    epochs: int,
    batch_size: int,
    num_workers: int,
    subsample_fraction: Optional[float],
) -> BenchResult:
    label = "1gpu_single"
    print("\n================ 1-GPU baseline (single process) ================")

    C_train, X_train_norm, C_test, X_test_norm, _, _ = load_and_preprocess(
        subsample_fraction=subsample_fraction
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[{label}] Using CUDA on device {device}")
    else:
        device = torch.device("cpu")
        print(f"[{label}] CUDA not available, using CPU")

    C_train_t = torch.from_numpy(C_train).float()
    X_train_t = torch.from_numpy(X_train_norm).float()
    C_test_t = torch.from_numpy(C_test).float()
    X_test_t = torch.from_numpy(X_test_norm).float()

    train_ds = TensorDataset(C_train_t, X_train_t)
    test_ds = TensorDataset(C_test_t, X_test_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    in_dim = C_train.shape[1]
    out_dim = X_train_norm.shape[1]

    model = SimpleRegressor(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()

    n_samples = C_train.shape[0]
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_C, batch_X in train_loader:
            batch_C = batch_C.to(device, non_blocking=True)
            batch_X = batch_X.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_C)
            loss = criterion(preds, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_C.size(0)

        epoch_loss /= n_samples
        print(f"[{label}] Epoch {epoch+1}/{epochs} - train MSE {epoch_loss:.6f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0

    samples_total = n_samples * epochs
    throughput = samples_total / max(wall, 1e-9)

    def eval_mse(loader, split_name: str) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_C, batch_X in loader:
                batch_C = batch_C.to(device, non_blocking=True)
                batch_X = batch_X.to(device, non_blocking=True)
                preds = model(batch_C)
                loss = criterion(preds, batch_X)
                bsz = batch_C.size(0)
                total_loss += loss.item() * bsz
                count += bsz
        mse = total_loss / max(count, 1)
        print(f"[{label}] {split_name} MSE {mse:.6f}")
        return mse

    train_mse = eval_mse(train_loader, "train")
    test_mse = eval_mse(test_loader, "test")

    print(f"\n[{label}] run complete")
    print(f"  wall time (s):           {wall:.2f}")
    print(f"  total samples:          {samples_total}")
    print(f"  throughput (samples/s): {throughput:.2f}")
    print(f"  final train MSE:        {train_mse:.6f}")
    print(f"  final test MSE:         {test_mse:.6f}")

    return BenchResult(
        label=label,
        wall_seconds=wall,
        samples_total=samples_total,
        throughput_sps=throughput,
        train_mse_mean=train_mse,
        test_mse_mean=test_mse,
    )


def ddp_worker(
    rank: int,
    world_size: int,
    port: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    subsample_fraction: Optional[float],
    result_dict,
):
    set_seeds(rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    label = "2gpu_ddp"
    if rank == 0:
        print("\n================ 2-GPU DDP baseline ================")
        print(f"[{label}] world_size={world_size}, backend=gloo, init_method={init_method}")
        if torch.cuda.is_available():
            print(f"[{label}] Using GPUs 0 and 1 with DDP")

    C_train, X_train_norm, C_test, X_test_norm, _, _ = load_and_preprocess(
        subsample_fraction=subsample_fraction
    )

    C_train_t = torch.from_numpy(C_train).float()
    X_train_t = torch.from_numpy(X_train_norm).float()
    C_test_t = torch.from_numpy(C_test).float()
    X_test_t = torch.from_numpy(X_test_norm).float()

    train_ds = TensorDataset(C_train_t, X_train_t)
    test_ds = TensorDataset(C_test_t, X_test_t)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    in_dim = C_train.shape[1]
    out_dim = X_train_norm.shape[1]

    model = SimpleRegressor(in_dim, out_dim).to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()

    n_samples = C_train.shape[0]

    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        count_seen = 0

        for batch_C, batch_X in train_loader:
            batch_C = batch_C.to(device, non_blocking=True)
            batch_X = batch_X.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = ddp_model(batch_C)
            loss = criterion(preds, batch_X)
            loss.backward()
            optimizer.step()

            bsz = batch_C.size(0)
            running_loss += loss.item() * bsz
            count_seen += bsz

        loss_tensor = torch.tensor([running_loss, count_seen], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            total_loss, total_count = loss_tensor.tolist()
            epoch_loss = total_loss / max(total_count, 1.0)
            print(f"[{label}] Epoch {epoch+1}/{epochs} - train MSE {epoch_loss:.6f}")

    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0

    if rank == 0:
        eval_model = ddp_model.module
        eval_model.eval()

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_loader_full = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        def eval_mse(loader, split_name: str) -> float:
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for batch_C, batch_X in loader:
                    batch_C = batch_C.to(device, non_blocking=True)
                    batch_X = batch_X.to(device, non_blocking=True)
                    preds = eval_model(batch_C)
                    loss = criterion(preds, batch_X)
                    bsz = batch_C.size(0)
                    total_loss += loss.item() * bsz
                    count += bsz
            mse = total_loss / max(count, 1)
            print(f"[{label}] {split_name} MSE {mse:.6f}")
            return mse

        train_mse = eval_mse(train_loader_full, "train")
        test_mse = eval_mse(test_loader, "test")

        samples_total = n_samples * epochs
        throughput = samples_total / max(wall, 1e-9)

        print(f"\n[{label}] run complete")
        print(f"  wall time (s):           {wall:.2f}")
        print(f"  total samples:          {samples_total}")
        print(f"  throughput (samples/s): {throughput:.2f}")
        print(f"  final train MSE:        {train_mse:.6f}")
        print(f"  final test MSE:         {test_mse:.6f}")

        result_dict["2gpu_ddp"] = BenchResult(
            label=label,
            wall_seconds=wall,
            samples_total=samples_total,
            throughput_sps=throughput,
            train_mse_mean=train_mse,
            test_mse_mean=test_mse,
        )

    dist.destroy_process_group()


# CSV writer
def save_results_csv(results: List[BenchResult], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "scale_results_unseen_ddp.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "wall_seconds",
                "samples_total",
                "throughput_samples_per_s",
                "train_mse_mean",
                "test_mse_mean",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.label,
                    f"{r.wall_seconds:.6f}",
                    r.samples_total,
                    f"{r.throughput_sps:.6f}",
                    f"{r.train_mse_mean:.6f}",
                    f"{r.test_mse_mean:.6f}",
                ]
            )
    print(f"\nSaved CSV → {path}")


# CLI and main
def parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on HPC).",
    )
    ap.add_argument(
        "--subsample-fraction",
        type=float,
        default=None,
        help="Optional fraction of rows to subsample for quick tests",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="bench_out_unseen",
    )
    ap.add_argument(
        "--ddp-port",
        type=str,
        default="29611",
        help="TCP port for DDP init_method (tcp://127.0.0.1:PORT).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    set_env_defaults()

    results: List[BenchResult] = []

    res_1gpu = run_single_gpu(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subsample_fraction=args.subsample_fraction,
    )
    results.append(res_1gpu)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        world_size = 2
        port = args.ddp_port

        manager = mp.Manager()
        result_dict = manager.dict()

        mp.spawn(
            ddp_worker,
            args=(
                world_size,
                port,
                args.epochs,
                args.batch_size,
                args.num_workers,
                args.subsample_fraction,
                result_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        if "2gpu_ddp" in result_dict:
            results.append(result_dict["2gpu_ddp"])
        else:
            print("\n[WARN] DDP finished but no result in result_dict['2gpu_ddp'].")
    else:
        print("\n[Info] < 2 GPUs visible; skipping 2-GPU DDP benchmark.")

    save_results_csv(results, args.outdir)


if __name__ == "__main__":
    main()
