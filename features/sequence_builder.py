"""Sequence window dataset generator with clip-level split isolation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.kinematics import compute_joint_angles, compute_kinematics
from features.normalization import normalize_keypoints, LEFT_WRIST, RIGHT_WRIST
from features.spatial_interaction import compute_spatial_interactions


def extract_features_for_frame(
    keypoints_xyc: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    shelf_polygons: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Extract a unified 1D feature vector for a single frame.

    Vector Layout:
        - 34 floats: Normalized (x, y) keypoints for 17 joints.
        - 17 floats: Confidence scores for 17 joints.
        - 6 floats: Joint angles (l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip).
        - 15 floats: Spatial interaction features.
        - Total: 72 features per frame.

    Args:
        keypoints_xyc: (17, 3) keypoints array.
        bbox_xyxy: Optional bounding box tuple.
        shelf_polygons: Optional list of shelf polygon arrays.

    Returns:
        1D float32 numpy array of shape (72,).
    """
    pts = np.array(keypoints_xyc, dtype=np.float32)
    norm_pts = normalize_keypoints(pts, bbox_xyxy)

    # 1. Normalized keypoints (34 features)
    norm_xy = norm_pts[:, :2].flatten()

    # 2. Joint confidences (17 features)
    confidences = pts[:, 2].flatten()

    # 3. Joint angles (6 features)
    angles_dict = compute_joint_angles(norm_pts)
    angles = np.array(list(angles_dict.values()), dtype=np.float32)

    # 4. Spatial interactions (15 features)
    spatial_dict = compute_spatial_interactions(norm_pts, shelf_polygons)
    spatial = np.array(list(spatial_dict.values()), dtype=np.float32)

    return np.concatenate([norm_xy, confidences, angles, spatial], axis=0)


def extract_features_for_track(
    records: list[dict[str, Any]],
    window_size: 32 = 32,
    stride: 8 = 8,
    shelf_polygons: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract sequence feature windows for a single person track sequence.

    Args:
        records: Chronologically sorted keypoint records for one track ID.
        window_size: Window length in frames (e.g. 32 frames).
        stride: Window stride step in frames (e.g. 8 frames).
        shelf_polygons: Optional shelf polygons.

    Returns:
        Tuple of:
            windows: numpy array of shape (N_windows, window_size, feature_dim)
            window_meta: list of metadata dicts per window
    """
    if len(records) < window_size:
        return np.empty((0, window_size, 77), dtype=np.float32), []

    # 1. Extract per-frame features
    raw_keypoints = []
    frame_features = []
    for r in records:
        kpts = np.array(r["keypoints_xyc"], dtype=np.float32)
        bbox = tuple(r["bbox_xyxy"]) if "bbox_xyxy" in r and r["bbox_xyxy"] is not None else None
        feat = extract_features_for_frame(kpts, bbox, shelf_polygons)
        frame_features.append(feat)
        norm_kpts = normalize_keypoints(kpts, bbox)
        raw_keypoints.append(norm_kpts)

    frame_features = np.array(frame_features, dtype=np.float32)  # (T, 73)
    raw_keypoints = np.array(raw_keypoints, dtype=np.float32)    # (T, 17, 3)

    # 2. Compute wrist kinematic features (velocities & accelerations) (4 features)
    vels, accs = compute_kinematics(raw_keypoints, fps=15.0)
    l_wrist_vel_mag = np.linalg.norm(vels[:, LEFT_WRIST], axis=1, keepdims=True)
    r_wrist_vel_mag = np.linalg.norm(vels[:, RIGHT_WRIST], axis=1, keepdims=True)
    l_wrist_acc_mag = np.linalg.norm(accs[:, LEFT_WRIST], axis=1, keepdims=True)
    r_wrist_acc_mag = np.linalg.norm(accs[:, RIGHT_WRIST], axis=1, keepdims=True)

    kinematic_feats = np.hstack([l_wrist_vel_mag, r_wrist_vel_mag, l_wrist_acc_mag, r_wrist_acc_mag])  # (T, 4)
    full_frame_feats = np.hstack([frame_features, kinematic_feats])  # (T, 77)

    # 3. Sliding window generation
    windows = []
    window_meta = []
    T = len(full_frame_feats)

    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size
        win = full_frame_feats[start_idx:end_idx]
        windows.append(win)
        meta = {
            "start_frame": records[start_idx].get("frame_index", records[start_idx].get("frame_idx", 0)),
            "end_frame": records[end_idx - 1].get("frame_index", records[end_idx - 1].get("frame_idx", 0)),
            "start_timestamp_ms": records[start_idx].get("timestamp_ms", 0),
            "end_timestamp_ms": records[end_idx - 1].get("timestamp_ms", 0),
            "track_id": records[0]["track_id"],
            "camera_id": records[0].get("camera_id", "cam1"),
        }
        window_meta.append(meta)

    if len(windows) == 0:
        return np.empty((0, window_size, 77), dtype=np.float32), []

    return np.array(windows, dtype=np.float32), window_meta


def extract_features_for_clip(
    keypoints_jsonl_path: str | Path,
    label: int,
    window_size: int = 32,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract features for all tracks in a single keypoint JSONL file.

    Returns:
        Tuple of (X_windows, y_labels, metadata_list)
    """
    path = Path(keypoints_jsonl_path)
    if not path.exists():
        return np.empty((0, window_size, 77), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    # Read records grouped by track_id
    tracks: dict[int, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t_id = rec.get("track_id", 0)
            if t_id not in tracks:
                tracks[t_id] = []
            tracks[t_id].append(rec)

    all_windows = []
    all_labels = []
    all_meta = []

    for t_id, recs in tracks.items():
        recs_sorted = sorted(recs, key=lambda x: x.get("frame_index", x.get("frame_idx", 0)))
        wins, meta = extract_features_for_track(recs_sorted, window_size, stride)
        if len(wins) > 0:
            all_windows.append(wins)
            all_labels.extend([label] * len(wins))
            for m in meta:
                m["clip_id"] = path.stem
                all_meta.append(m)

    if len(all_windows) == 0:
        return np.empty((0, window_size, 77), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    X = np.concatenate(all_windows, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    return X, y, all_meta


def build_sequence_dataset(
    keypoints_dir: str | Path = "data/keypoints",
    manifest_path: str | Path = "data/manifests/clips.csv",
    output_dir: str | Path = "data/features",
    window_size: int = 32,
    stride: int = 8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract sequence features and split dataset at the clip level.

    Returns:
        Summary dict containing tensor shapes and split stats.
    """
    keypoints_dir = Path(keypoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read Manifest
    manifest_df = pd.read_csv(manifest_path)
    # Mapping label: shoplifting -> 1, normal -> 0
    label_map = {"shoplifting": 1, "normal": 0, "1": 1, "0": 0}
    manifest_df["binary_label"] = manifest_df["label"].astype(str).str.lower().map(label_map)

    # 2. Split clips deterministically
    np.random.seed(seed)
    clips = manifest_df[["clip_id", "binary_label"]].drop_duplicates().to_dict("records")
    np.random.shuffle(clips)

    n_total = len(clips)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))

    test_clips = set(c["clip_id"] for c in clips[:n_test])
    val_clips = set(c["clip_id"] for c in clips[n_test : n_test + n_val])
    train_clips = set(c["clip_id"] for c in clips[n_test + n_val :])

    splits = {"train": (train_clips, [], [], []), "val": (val_clips, [], [], []), "test": (test_clips, [], [], [])}

    # 3. Process clips
    for idx, row in manifest_df.iterrows():
        clip_id = str(row["clip_id"])
        label = int(row["binary_label"])
        jsonl_path = keypoints_dir / f"{clip_id}.jsonl"

        X_clip, y_clip, meta_clip = extract_features_for_clip(jsonl_path, label, window_size, stride)
        if len(X_clip) == 0:
            continue

        if clip_id in test_clips:
            split_name = "test"
        elif clip_id in val_clips:
            split_name = "val"
        else:
            split_name = "train"

        splits[split_name][1].append(X_clip)
        splits[split_name][2].append(y_clip)
        splits[split_name][3].extend(meta_clip)

    summary = {"window_size": window_size, "stride": stride, "splits": {}}

    for split_name, (_, X_list, y_list, meta_list) in splits.items():
        if len(X_list) > 0:
            X_arr = np.concatenate(X_list, axis=0)
            y_arr = np.concatenate(y_list, axis=0)
        else:
            X_arr = np.empty((0, window_size, 77), dtype=np.float32)
            y_arr = np.empty((0,), dtype=np.int64)

        np.save(output_dir / f"X_{split_name}.npy", X_arr)
        np.save(output_dir / f"y_{split_name}.npy", y_arr)

        with open(output_dir / f"meta_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2)

        summary["splits"][split_name] = {
            "samples": int(len(X_arr)),
            "positives": int(np.sum(y_arr == 1)) if len(y_arr) > 0 else 0,
            "negatives": int(np.sum(y_arr == 0)) if len(y_arr) > 0 else 0,
            "shape": list(X_arr.shape),
        }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
