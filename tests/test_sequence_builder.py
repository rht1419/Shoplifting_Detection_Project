"""Unit tests for sequence windowing and dataset builder."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from features.sequence_builder import extract_features_for_track, build_sequence_dataset
from tests.test_normalization import create_dummy_keypoints


def test_extract_features_for_track():
    dummy_kpts = create_dummy_keypoints().tolist()

    records = []
    for f_idx in range(40):
        records.append({
            "camera_id": "cam1",
            "frame_idx": f_idx,
            "timestamp_ms": f_idx * 66.6,
            "track_id": 1,
            "bbox_xyxy": [250.0, 80.0, 390.0, 440.0],
            "keypoints_xyc": dummy_kpts,
            "detection_confidence": 0.9,
            "track_age": f_idx + 1,
        })

    windows, meta = extract_features_for_track(records, window_size=32, stride=8)

    # 40 frames with window 32 and stride 8 -> (40 - 32)/8 + 1 = 2 windows
    assert windows.shape == (2, 32, 77)
    assert len(meta) == 2
    assert meta[0]["start_frame"] == 0
    assert meta[0]["end_frame"] == 31
    assert meta[1]["start_frame"] == 8
    assert meta[1]["end_frame"] == 39


def test_dataset_builder(tmp_path: Path):
    keypoints_dir = tmp_path / "keypoints"
    keypoints_dir.mkdir()

    manifest_path = tmp_path / "clips.csv"
    output_dir = tmp_path / "features"

    dummy_kpts = create_dummy_keypoints().tolist()

    # Create dummy JSONL keypoint log for clip1
    records1 = []
    for f_idx in range(40):
        records1.append({
            "camera_id": "cam1",
            "frame_idx": f_idx,
            "timestamp_ms": f_idx * 66.6,
            "track_id": 1,
            "bbox_xyxy": [250.0, 80.0, 390.0, 440.0],
            "keypoints_xyc": dummy_kpts,
            "detection_confidence": 0.9,
            "track_age": f_idx + 1,
        })

    with open(keypoints_dir / "clip1.jsonl", "w", encoding="utf-8") as f:
        for r in records1:
            f.write(json.dumps(r) + "\n")

    # Write manifest CSV
    manifest_content = "clip_id,label,source_path,frame_count,source_fps,width,height,duration_ms,target_fps\n"
    manifest_content += "clip1,shoplifting,data/shoplifting/clip1.mp4,40,30.0,640,480,1333.3,15.0\n"
    manifest_path.write_text(manifest_content)

    summary = build_sequence_dataset(
        keypoints_dir=keypoints_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        window_size=32,
        stride=8,
    )

    assert (output_dir / "X_train.npy").exists() or (output_dir / "X_test.npy").exists() or (output_dir / "X_val.npy").exists()
    assert (output_dir / "dataset_summary.json").exists()
