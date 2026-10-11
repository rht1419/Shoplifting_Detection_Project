"""CLI entry point for running Module 4 classifier training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.baseline import train_baseline_model, evaluate_baseline_model, save_baseline_model
from model.dataset import get_dataloaders
from model.evaluate import evaluate_predictions, find_optimal_threshold, compute_event_recall
from model.train import train_recurrent_model
from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate suspicious behaviour classifiers.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--features-dir", default=None, help="Directory containing X_train.npy, etc.")
    parser.add_argument("--model-type", default=None, choices=["gru", "lstm", "baseline"], help="Model architecture type.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--artifacts-dir", default=None, help="Directory to save checkpoints and metrics.")

    args = parser.parse_args()
    config = load_config(args.config)

    feat_cfg = config.get("features", {})
    model_cfg = config.get("model", {})

    features_dir = args.features_dir or feat_cfg.get("output_dir", "data/features")
    model_type = (args.model_type or model_cfg.get("model_type", "gru")).lower()
    epochs = args.epochs or model_cfg.get("epochs", 30)
    batch_size = args.batch_size or model_cfg.get("batch_size", 32)
    lr = args.lr or model_cfg.get("learning_rate", 0.001)
    artifacts_dir = args.artifacts_dir or model_cfg.get("artifacts_dir", "artifacts/models")

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Module 4: Classifier Training & Evaluation ===")
    print(f"Features directory: {features_dir}")
    print(f"Model architecture: {model_type.upper()}")
    print(f"Artifacts output: {artifacts_dir}")

    path = Path(features_dir)
    x_train_file = path / "X_train.npy"
    if not x_train_file.exists():
        print(f"\n[Error] Feature file '{x_train_file}' not found.")
        print("Please run feature extraction first: `python -m pipeline.extract_features`")
        return

    # 1. Load Data
    X_train = np.load(path / "X_train.npy")
    y_train = np.load(path / "y_train.npy")

    X_val = np.load(path / "X_val.npy") if (path / "X_val.npy").exists() else np.empty((0,))
    y_val = np.load(path / "y_val.npy") if (path / "y_val.npy").exists() else np.empty((0,))

    X_test = np.load(path / "X_test.npy") if (path / "X_test.npy").exists() else np.empty((0,))
    y_test = np.load(path / "y_test.npy") if (path / "y_test.npy").exists() else np.empty((0,))

    print(f"\nLoaded dataset shapes:")
    print(f"  Train: {X_train.shape} (Positives: {np.sum(y_train==1)})")
    print(f"  Val:   {X_val.shape} (Positives: {np.sum(y_val==1)})")
    print(f"  Test:  {X_test.shape} (Positives: {np.sum(y_test==1)})")

    if model_type == "baseline":
        # 2A. Train Tabular Baseline
        print("\nTraining Baseline Logistic Regression / Random Forest model...")
        pipeline = train_baseline_model(X_train, y_train, model_type="logistic")

        metrics_val = evaluate_baseline_model(pipeline, X_val, y_val) if len(X_val) > 0 else {}
        metrics_test = evaluate_baseline_model(pipeline, X_test, y_test) if len(X_test) > 0 else {}

        print("\nBaseline Results:")
        print(f"  Val  ROC-AUC: {metrics_val.get('roc_auc', 0.0):.4f} | PR-AUC: {metrics_val.get('pr_auc', 0.0):.4f}")
        print(f"  Test ROC-AUC: {metrics_test.get('roc_auc', 0.0):.4f} | PR-AUC: {metrics_test.get('pr_auc', 0.0):.4f}")

        save_baseline_model(pipeline, Path(artifacts_dir) / "baseline_model.joblib")
    else:
        # 2B. Train PyTorch GRU / LSTM
        train_loader, val_loader, test_loader = get_dataloaders(features_dir, batch_size=batch_size)
        if train_loader is None:
            print("[Error] Failed to create DataLoaders.")
            return

        input_dim = X_train.shape[-1] if len(X_train) > 0 else 76
        hidden_dim = model_cfg.get("hidden_dim", 64)
        num_layers = model_cfg.get("num_layers", 2)
        pos_weight = model_cfg.get("pos_weight", 2.0)

        print(f"\nTraining PyTorch {model_type.upper()} model for {epochs} epochs...")
        model, history = train_recurrent_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            model_type=model_type,
            epochs=epochs,
            lr=lr,
            pos_weight=pos_weight,
            artifacts_dir=artifacts_dir,
        )

        # 3. Evaluation on Test Split
        model.eval()
        if test_loader is not None and len(y_test) > 0:
            with torch.no_grad():
                all_probs = []
                for x, _ in test_loader:
                    logits = model(x)
                    probs = torch.sigmoid(logits).numpy().flatten()
                    all_probs.append(probs)
                all_probs = np.concatenate(all_probs, axis=0)

            opt_thresh = find_optimal_threshold(y_test, all_probs)
            metrics_test = evaluate_predictions(y_test, all_probs, threshold=opt_thresh)

            print(f"\n{model_type.upper()} Test Evaluation (Threshold={opt_thresh:.2f}):")
            print(f"  ROC-AUC:   {metrics_test['roc_auc']:.4f}")
            print(f"  PR-AUC:    {metrics_test['pr_auc']:.4f}")
            print(f"  Precision: {metrics_test['precision']:.4f}")
            print(f"  Recall:    {metrics_test['recall']:.4f}")
            print(f"  F1 Score:  {metrics_test['f1']:.4f}")

            # Clip-level event recall
            meta_file = path / "meta_test.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_test = json.load(f)
                evt_metrics = compute_event_recall(meta_test, y_test, all_probs, threshold=opt_thresh)
                print(f"  Event Recall: {evt_metrics['event_recall']:.4f} ({evt_metrics['detected_events']}/{evt_metrics['total_events']} shoplifting clips detected)")

            with open(Path(artifacts_dir) / "test_evaluation.json", "w", encoding="utf-8") as f:
                json.dump(metrics_test, f, indent=2)

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
