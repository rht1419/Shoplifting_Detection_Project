"""CLI entry point for running Module 3 feature extraction and sequence building."""

from __future__ import annotations

import argparse
from pathlib import Path

from features.sequence_builder import build_sequence_dataset
from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Extract normalized sequence features from keypoint logs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--keypoints-dir", default=None, help="Directory containing keypoint JSONL files.")
    parser.add_argument("--manifest", default=None, help="Path to clip manifest CSV.")
    parser.add_argument("--output-dir", default=None, help="Output directory for numpy feature datasets.")
    parser.add_argument("--window-size", type=int, default=None, help="Sliding window size in frames.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride in frames.")

    args = parser.parse_args()
    config = load_config(args.config)

    feat_cfg = config.get("features", {})
    data_cfg = config.get("data", {})

    keypoints_dir = args.keypoints_dir or feat_cfg.get("keypoints_dir", "data/keypoints")
    manifest_path = args.manifest or data_cfg.get("manifest_path", "data/manifests/clips.csv")
    output_dir = args.output_dir or feat_cfg.get("output_dir", "data/features")
    window_size = args.window_size or feat_cfg.get("window_size", 32)
    stride = args.stride or feat_cfg.get("stride", 8)

    print(f"=== Module 3: Feature Extraction ===")
    print(f"Keypoints directory: {keypoints_dir}")
    print(f"Manifest CSV: {manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: {window_size} frames, Stride: {stride} frames")

    if not Path(keypoints_dir).exists():
        print(f"\n[Warning] Keypoints directory '{keypoints_dir}' does not exist.")
        print("Please run keypoint extraction first (`python -m pipeline.batch_extract`) or populate keypoints.")

    summary = build_sequence_dataset(
        keypoints_dir=keypoints_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        window_size=window_size,
        stride=stride,
    )

    print("\nExtraction Summary:")
    for split, info in summary.get("splits", {}).items():
        print(f"  [{split.upper()}] Samples: {info['samples']} | Positives: {info['positives']} | Negatives: {info['negatives']} | Shape: {info['shape']}")

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()
