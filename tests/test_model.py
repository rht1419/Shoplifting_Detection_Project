"""Unit tests for PyTorch GRU/LSTM models and baseline trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.baseline import train_baseline_model, evaluate_baseline_model
from model.network import ShopliftingGRU, ShopliftingLSTM
from model.evaluate import evaluate_predictions, find_optimal_threshold


def test_gru_forward_shape():
    model = ShopliftingGRU(input_dim=76, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 32, 76)  # batch_size=4, seq_len=32, features=76

    logits = model(x)
    assert logits.shape == (4, 1)


def test_lstm_forward_shape():
    model = ShopliftingLSTM(input_dim=76, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 32, 76)

    logits = model(x)
    assert logits.shape == (4, 1)


def test_baseline_trainer():
    # Create synthetic dataset (N=20, T=32, F=76)
    X_train = np.random.randn(20, 32, 76).astype(np.float32)
    y_train = np.array([1]*10 + [0]*10, dtype=np.int64)

    pipeline = train_baseline_model(X_train, y_train, model_type="logistic")
    metrics = evaluate_baseline_model(pipeline, X_train, y_train)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.5


def test_evaluation_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

    opt_thresh = find_optimal_threshold(y_true, y_prob, target_recall=0.8)
    assert 0.05 <= opt_thresh <= 0.95
