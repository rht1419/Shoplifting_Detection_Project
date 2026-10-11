"""Unit tests for joint angles, kinematics, and spatial interactions."""

from __future__ import annotations

import numpy as np

from features.kinematics import compute_angle_3pt, compute_joint_angles, compute_kinematics
from features.spatial_interaction import compute_spatial_interactions
from tests.test_normalization import create_dummy_keypoints


def test_angle_3pt_perpendicular():
    pA = np.array([0.0, 1.0, 1.0])
    pB = np.array([0.0, 0.0, 1.0])
    pC = np.array([1.0, 0.0, 1.0])

    angle = compute_angle_3pt(pA, pB, pC)
    # 90 degrees = pi / 2 = ~1.570796 rad
    np.testing.assert_allclose(angle, np.pi / 2, rtol=1e-4)


def test_compute_joint_angles():
    kpts = create_dummy_keypoints()
    angles = compute_joint_angles(kpts)

    assert "l_elbow_angle" in angles
    assert "r_elbow_angle" in angles
    assert "l_shoulder_angle" in angles
    assert "r_shoulder_angle" in angles
    assert "l_hip_angle" in angles
    assert "r_hip_angle" in angles

    for v in angles.values():
        assert 0.0 <= v <= np.pi


def test_spatial_interactions():
    kpts = create_dummy_keypoints()
    spatial = compute_spatial_interactions(kpts)

    assert "l_wrist_to_same_hip" in spatial
    assert "r_wrist_to_same_hip" in spatial
    assert "wrist_to_wrist" in spatial
    assert "l_hand_near_waist" in spatial
    assert spatial["l_wrist_to_same_hip"] >= 0.0


def test_kinematics_sequence():
    seq = np.stack([create_dummy_keypoints() for _ in range(10)])
    # Add movement to wrists across time
    for i in range(10):
        seq[i, 9, 0] += i * 5.0  # Left wrist moving along X

    vels, accs = compute_kinematics(seq, fps=15.0)

    assert vels.shape == (10, 17, 2)
    assert accs.shape == (10, 17, 2)
    # Left wrist x-velocity should be ~75.0 (5.0 * 15 FPS)
    assert abs(vels[1, 9, 0] - 75.0) < 1e-2
