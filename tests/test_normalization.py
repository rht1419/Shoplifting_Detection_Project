"""Unit tests for pose normalization module."""

from __future__ import annotations

import numpy as np
import pytest

from features.normalization import normalize_keypoints, get_anchor_point, get_scale_factor


def create_dummy_keypoints() -> np.ndarray:
    """Create synthetic COCO-17 keypoint array for testing."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Assign confidences
    kpts[:, 2] = 0.9

    # Set coordinates
    kpts[0] = [320.0, 100.0, 0.9]   # Nose
    kpts[5] = [300.0, 150.0, 0.9]   # Left Shoulder
    kpts[6] = [340.0, 150.0, 0.9]   # Right Shoulder
    kpts[7] = [280.0, 200.0, 0.9]   # Left Elbow
    kpts[8] = [360.0, 200.0, 0.9]   # Right Elbow
    kpts[9] = [270.0, 250.0, 0.9]   # Left Wrist
    kpts[10] = [370.0, 250.0, 0.9]  # Right Wrist
    kpts[11] = [305.0, 270.0, 0.9]  # Left Hip
    kpts[12] = [335.0, 270.0, 0.9]  # Right Hip
    kpts[13] = [305.0, 350.0, 0.9]  # Left Knee
    kpts[14] = [335.0, 350.0, 0.9]  # Right Knee
    kpts[15] = [305.0, 420.0, 0.9]  # Left Ankle
    kpts[16] = [335.0, 420.0, 0.9]  # Right Ankle
    return kpts


def test_anchor_point_calculation():
    kpts = create_dummy_keypoints()
    anchor = get_anchor_point(kpts)

    # Mid-hip expected: (305 + 335) / 2 = 320, (270 + 270) / 2 = 270
    np.testing.assert_allclose(anchor, [320.0, 270.0], rtol=1e-3)


def test_scale_factor_calculation():
    kpts = create_dummy_keypoints()
    scale = get_scale_factor(kpts)

    # Torso height expected: distance between mid-shoulder (320, 150) and mid-hip (320, 270) = 120
    assert abs(scale - 120.0) < 1e-2


def test_normalize_keypoints():
    kpts = create_dummy_keypoints()
    norm_kpts = normalize_keypoints(kpts)

    assert norm_kpts.shape == (17, 3)
    # Mid-hip normalized should be (0, 0)
    mid_hip_norm = (norm_kpts[11, :2] + norm_kpts[12, :2]) / 2.0
    np.testing.assert_allclose(mid_hip_norm, [0.0, 0.0], atol=1e-4)

    # Confidences must remain unchanged
    np.testing.assert_allclose(norm_kpts[:, 2], kpts[:, 2])
