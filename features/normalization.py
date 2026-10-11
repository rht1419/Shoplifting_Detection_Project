"""Keypoint normalization functions for distance and scale invariance."""

from __future__ import annotations

import numpy as np

# COCO-17 Keypoint Indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def get_anchor_point(keypoints: np.ndarray, bbox_xyxy: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Compute center anchor point (mid-hip, falling back to mid-shoulder or bbox center).

    Args:
        keypoints: Array of shape (17, 3) with (x, y, confidence).
        bbox_xyxy: Optional bounding box (x1, y1, x2, y2).

    Returns:
        Anchor (x, y) point as a 2D float numpy array.
    """
    l_hip_conf = keypoints[LEFT_HIP, 2]
    r_hip_conf = keypoints[RIGHT_HIP, 2]

    # Standard anchor: Mid-hip
    if l_hip_conf > 0.2 and r_hip_conf > 0.2:
        return (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2.0

    # Fallback 1: Mid-shoulder
    l_sh_conf = keypoints[LEFT_SHOULDER, 2]
    r_sh_conf = keypoints[RIGHT_SHOULDER, 2]
    if l_sh_conf > 0.2 and r_sh_conf > 0.2:
        return (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2.0

    # Fallback 2: Bounding box center
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    # Fallback 3: Mean of all confident keypoints
    confident_pts = keypoints[keypoints[:, 2] > 0.2, :2]
    if len(confident_pts) > 0:
        return np.mean(confident_pts, axis=0)

    return np.array([0.0, 0.0], dtype=np.float32)


def get_scale_factor(keypoints: np.ndarray, bbox_xyxy: tuple[float, float, float, float] | None = None) -> float:
    """Compute scale factor (torso height, falling back to shoulder width or bbox height).

    Args:
        keypoints: Array of shape (17, 3) with (x, y, confidence).
        bbox_xyxy: Optional bounding box (x1, y1, x2, y2).

    Returns:
        Positive float scale factor.
    """
    l_sh_conf = keypoints[LEFT_SHOULDER, 2]
    r_sh_conf = keypoints[RIGHT_SHOULDER, 2]
    l_hip_conf = keypoints[LEFT_HIP, 2]
    r_hip_conf = keypoints[RIGHT_HIP, 2]

    # Primary scale: Torso height (mid-shoulder to mid-hip distance)
    if (l_sh_conf > 0.2 or r_sh_conf > 0.2) and (l_hip_conf > 0.2 or r_hip_conf > 0.2):
        sh_pt = keypoints[LEFT_SHOULDER, :2] if l_sh_conf >= r_sh_conf else keypoints[RIGHT_SHOULDER, :2]
        if l_sh_conf > 0.2 and r_sh_conf > 0.2:
            sh_pt = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2.0

        hip_pt = keypoints[LEFT_HIP, :2] if l_hip_conf >= r_hip_conf else keypoints[RIGHT_HIP, :2]
        if l_hip_conf > 0.2 and r_hip_conf > 0.2:
            hip_pt = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2.0

        torso_dist = float(np.linalg.norm(sh_pt - hip_pt))
        if torso_dist > 1.0:
            return torso_dist

    # Fallback 1: Shoulder width
    if l_sh_conf > 0.2 and r_sh_conf > 0.2:
        sh_width = float(np.linalg.norm(keypoints[LEFT_SHOULDER, :2] - keypoints[RIGHT_SHOULDER, :2]))
        if sh_width > 1.0:
            return sh_width * 1.5

    # Fallback 2: Bounding box height/max dimension
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        h = y2 - y1
        w = x2 - x1
        bbox_scale = float(max(h, w))
        if bbox_scale > 1.0:
            return bbox_scale

    return 100.0  # Default non-zero fallback scale


def normalize_keypoints(
    keypoints: np.ndarray | list[list[float]],
    bbox_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Normalize keypoints relative to anchor point and torso/bbox scale.

    Args:
        keypoints: (17, 3) keypoints array or nested list of (x, y, confidence).
        bbox_xyxy: Optional bounding box tuple.

    Returns:
        Normalized keypoints array of shape (17, 3) with centered & scaled (x, y) and untouched confidence c.
    """
    pts = np.array(keypoints, dtype=np.float32)
    if pts.shape != (17, 3):
        raise ValueError(f"Expected keypoints shape (17, 3), got {pts.shape}")

    anchor = get_anchor_point(pts, bbox_xyxy)
    scale = get_scale_factor(pts, bbox_xyxy)

    normalized = np.copy(pts)
    normalized[:, 0] = (pts[:, 0] - anchor[0]) / scale
    normalized[:, 1] = (pts[:, 1] - anchor[1]) / scale

    return normalized
