"""
Data generators used for Contextualized regression training.
"""

from abc import abstractmethod
import torch
from torch.utils.data import Dataset


class MultivariateDataset(Dataset):
    """
    Simple multivariate dataset with context, predictors, and outcomes.
    """
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype
    
    def __len__(self):
        return len(self.C)
 
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "contexts": self.C[idx],
            "predictors": self.X[idx].expand(self.y_dim, -1),
            "outcomes": self.Y[idx].unsqueeze(-1),
        }


class UnivariateDataset(Dataset):
    """
    Simple univariate dataset with context, predictors, and one outcome.
    """
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype
    
    def __len__(self):
        return len(self.C)
 
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "contexts": self.C[idx],
            "predictors": self.X[idx].expand(self.y_dim, -1).unsqueeze(-1),
            "outcomes": self.Y[idx].expand(self.x_dim, -1).T.unsqueeze(-1),
        }


class MultitaskMultivariateDataset(Dataset):
    """
    Multi-task Multivariate Dataset.
    """
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype
    
    def __len__(self):
        return len(self.C) * self.y_dim
    
    def __getitem__(self, idx):
        # Get task-split sample indices
        n_i = idx // self.y_dim
        y_i = idx % self.y_dim
        # Create a one-hot encoding for the task
        t = torch.zeros(self.y_dim)
        t[y_i] = 1
        return {
            "idx": idx,
            "contexts": self.C[n_i],
            "task": t,
            "predictors": self.X[n_i],
            "outcomes": self.Y[n_i, y_i].unsqueeze(0),
            "sample_idx": n_i,
            "outcome_idx": y_i,
        }

    # def __next__(self):
    #     if self.y_i >= self.y_dim:
    #         self.n_i += 1
    #         self.y_i = 0
    #     if self.n_i >= self.n:
    #         self.n_i = 0
    #         raise StopIteration
    #     t = torch.zeros(self.y_dim)
    #     t[self.y_i] = 1
    #     ret = (
    #         self.C[self.n_i],
    #         t,
    #         self.X[self.n_i],
    #         self.Y[self.n_i, self.y_i].unsqueeze(0),
    #         self.n_i,
    #         self.y_i,
    #     )
    #     self.y_i += 1
    #     return ret


class MultitaskUnivariateDataset(Dataset):
    """
    Multitask Univariate Dataset.
    Splits each sample into univariate X and Y feature pairs for univariate regression tasks.
    """ 
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype
    
    def __len__(self):
        return len(self.C) * self.x_dim * self.y_dim
 
    def __getitem__(self, idx):
        # Get task-split sample indices
        n_i = idx // (self.x_dim * self.y_dim)
        x_i = (idx // self.y_dim) % self.x_dim
        y_i = idx % self.y_dim
        # Create a one-hot encoding for the task
        t = torch.zeros(self.x_dim + self.y_dim)
        t[x_i] = 1
        t[self.x_dim + y_i] = 1
        return {
            "idx": idx,
            "contexts": self.C[n_i],
            "task": t,
            "predictors": self.X[n_i, x_i].unsqueeze(0),
            "outcomes": self.Y[n_i, y_i].unsqueeze(0),
            "sample_idx": n_i,
            "predictor_idx": x_i,
            "outcome_idx": y_i,
        }