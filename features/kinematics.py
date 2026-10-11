"""Kinematic feature extraction: joint angles, velocities, and accelerations."""

from __future__ import annotations

import numpy as np

from features.normalization import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
)


def compute_angle_3pt(pA: np.ndarray, pB: np.ndarray, pC: np.ndarray) -> float:
    """Compute 2D angle in radians between BA and BC with vertex at B.

    Returns:
        Angle in radians [0, pi]. Returns 0.0 if vectors are degenerate.
    """
    vBA = pA[:2] - pB[:2]
    vBC = pC[:2] - pB[:2]

    norm_BA = np.linalg.norm(vBA)
    norm_BC = np.linalg.norm(vBC)

    if norm_BA < 1e-6 or norm_BC < 1e-6:
        return 0.0

    cosine = np.dot(vBA, vBC) / (norm_BA * norm_BC)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def compute_joint_angles(keypoints: np.ndarray) -> dict[str, float]:
    """Compute key anatomical joint angles for a single pose frame.

    Args:
        keypoints: Array of shape (17, 3) with (x, y, c).

    Returns:
        Dictionary of joint angle features in radians.
    """
    pts = np.array(keypoints, dtype=np.float32)

    l_elbow_angle = compute_angle_3pt(pts[LEFT_SHOULDER], pts[LEFT_ELBOW], pts[LEFT_WRIST])
    r_elbow_angle = compute_angle_3pt(pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], pts[RIGHT_WRIST])

    l_shoulder_angle = compute_angle_3pt(pts[LEFT_HIP], pts[LEFT_SHOULDER], pts[LEFT_ELBOW])
    r_shoulder_angle = compute_angle_3pt(pts[RIGHT_HIP], pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW])

    l_hip_angle = compute_angle_3pt(pts[LEFT_SHOULDER], pts[LEFT_HIP], pts[LEFT_KNEE])
    r_hip_angle = compute_angle_3pt(pts[RIGHT_SHOULDER], pts[RIGHT_HIP], pts[RIGHT_KNEE])

    return {
        "l_elbow_angle": l_elbow_angle,
        "r_elbow_angle": r_elbow_angle,
        "l_shoulder_angle": l_shoulder_angle,
        "r_shoulder_angle": r_shoulder_angle,
        "l_hip_angle": l_hip_angle,
        "r_hip_angle": r_hip_angle,
    }


def compute_kinematics(
    sequence_keypoints: np.ndarray,
    fps: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-to-frame joint velocities and accelerations across a sequence window.

    Args:
        sequence_keypoints: Array of shape (T, 17, 3) containing normalized keypoints.
        fps: Target sampling rate in frames per second.

    Returns:
        Tuple of (velocities, accelerations) arrays:
            velocities: shape (T, 17, 2)
            accelerations: shape (T, 17, 2)
    """
    seq = np.array(sequence_keypoints, dtype=np.float32)
    T, N, _ = seq.shape
    dt = 1.0 / fps

    velocities = np.zeros((T, N, 2), dtype=np.float32)
    accelerations = np.zeros((T, N, 2), dtype=np.float32)

    if T < 2:
        return velocities, accelerations

    # 1st derivative: Velocity
    diff = seq[1:, :, :2] - seq[:-1, :, :2]
    velocities[1:] = diff / dt
    velocities[0] = velocities[1]  # Padding first frame with second frame velocity

    # 2nd derivative: Acceleration
    if T >= 3:
        acc_diff = velocities[1:] - velocities[:-1]
        accelerations[1:] = acc_diff / dt
        accelerations[0] = accelerations[1]

    return velocities, accelerations
