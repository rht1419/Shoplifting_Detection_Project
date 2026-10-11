"""PyTorch training loop for recurrent sequence models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.network import ShopliftingGRU, ShopliftingLSTM
from model.evaluate import evaluate_predictions


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)
        count += len(x)

    return total_loss / max(1, count)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run evaluation over a validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    count = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y.cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets)

        total_loss += loss.item() * len(x)
        count += len(x)

    if len(all_probs) == 0:
        return 0.0, np.array([]), np.array([])

    all_probs = np.concatenate(all_probs, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    avg_loss = total_loss / max(1, count)
    return avg_loss, all_probs, all_targets


def train_recurrent_model(
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    input_dim: int = 76,
    hidden_dim: int = 64,
    num_layers: int = 2,
    model_type: str = "gru",
    epochs: int = 30,
    lr: float = 1e-3,
    pos_weight: float = 2.0,
    device_name: str = "cpu",
    artifacts_dir: str | Path = "artifacts/models",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train PyTorch recurrent model and save best checkpoint.

    Returns:
        Tuple of (best_model, training_history_dict)
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")

    if model_type.lower() == "lstm":
        model = ShopliftingLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    else:
        model = ShopliftingGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_auc = -1.0
    best_checkpoint_path = artifacts_dir / "best_model.pt"

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_roc_auc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_loss)

        val_loss, val_pr_auc, val_roc_auc = 0.0, 0.0, 0.0
        if val_loader is not None:
            val_loss, val_probs, val_targets = evaluate_epoch(model, val_loader, criterion, device)
            if len(val_targets) > 0:
                metrics = evaluate_predictions(val_targets, val_probs)
                val_pr_auc = metrics["pr_auc"]
                val_roc_auc = metrics["roc_auc"]

            scheduler.step(val_pr_auc)

        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)

        # Checkpointing best model by validation PR-AUC
        score = val_pr_auc if val_loader is not None else -train_loss
        if score > best_val_auc:
            best_val_auc = score
            torch.save({
                "epoch": epoch,
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "state_dict": model.state_dict(),
                "val_pr_auc": val_pr_auc,
            }, best_checkpoint_path)

    with open(artifacts_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best state dict
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    return model, history
