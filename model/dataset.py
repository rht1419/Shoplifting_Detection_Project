"""PyTorch Dataset and DataLoader wrappers for sequence features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapper for sequence feature tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize Dataset.

        Args:
            X: Sequence array of shape (N, T, F).
            y: Label array of shape (N,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    features_dir: str | Path = "data/features",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    """Load pre-processed numpy arrays and build PyTorch DataLoaders.

    Args:
        features_dir: Path to directory containing X_train.npy, etc.
        batch_size: Batch size.
        num_workers: DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    path = Path(features_dir)

    loaders = []
    for split in ["train", "val", "test"]:
        x_file = path / f"X_{split}.npy"
        y_file = path / f"y_{split}.npy"

        if x_file.exists() and y_file.exists():
            X = np.load(x_file)
            y = np.load(y_file)
            if len(X) > 0:
                ds = SequenceDataset(X, y)
                shuffle = (split == "train")
                loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                loaders.append(loader)
            else:
                loaders.append(None)
        else:
            loaders.append(None)

    return loaders[0], loaders[1], loaders[2]
