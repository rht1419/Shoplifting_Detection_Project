"""PyTorch GRU and LSTM neural network models for suspicious sequence classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShopliftingGRU(nn.Module):
    """Lightweight GRU network for sequence classification on laptop CPU."""

    def __init__(
        self,
        input_dim: int = 77,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection & LayerNorm
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sequence tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        # x: (B, T, F)
        h = self.input_proj(x)       # (B, T, H)
        h = self.layer_norm(h)

        out, _ = self.gru(h)          # out: (B, T, H)

        # Take last time step output (or mean-pooling over time)
        last_out = out[:, -1, :]      # (B, H)

        logits = self.classifier(last_out)  # (B, 1)
        return logits


class ShopliftingLSTM(nn.Module):
    """Lightweight LSTM network for sequence classification."""

    def __init__(
        self,
        input_dim: int = 77,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.layer_norm(h)

        out, _ = self.lstm(h)
        last_out = out[:, -1, :]

        logits = self.classifier(last_out)
        return logits
