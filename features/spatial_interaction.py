"""Spatial interaction features: hand-to-body, hand-to-hand, and hand-to-shelf distances."""

from __future__ import annotations

import numpy as np

from features.normalization import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
)


def compute_spatial_interactions(
    keypoints: np.ndarray,
    shelf_polygons: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute spatial distance and interaction features for one frame of normalized keypoints.

    Args:
        keypoints: Array of shape (17, 3) containing normalized keypoints (centered at mid-hip, torso-scaled).
        shelf_polygons: Optional list of polygon vertex arrays for store shelf areas.

    Returns:
        Dictionary of spatial interaction scalar features.
    """
    pts = np.array(keypoints, dtype=np.float32)

    l_wrist = pts[LEFT_WRIST, :2]
    r_wrist = pts[RIGHT_WRIST, :2]

    l_hip = pts[LEFT_HIP, :2]
    r_hip = pts[RIGHT_HIP, :2]
    mid_hip = (l_hip + r_hip) / 2.0

    l_shoulder = pts[LEFT_SHOULDER, :2]
    r_shoulder = pts[RIGHT_SHOULDER, :2]
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0

    nose = pts[NOSE, :2]

    # Hand-to-Body distances
    l_wrist_to_same_hip = float(np.linalg.norm(l_wrist - l_hip))
    r_wrist_to_same_hip = float(np.linalg.norm(r_wrist - r_hip))

    l_wrist_to_opp_hip = float(np.linalg.norm(l_wrist - r_hip))
    r_wrist_to_opp_hip = float(np.linalg.norm(r_wrist - l_hip))

    l_wrist_to_mid_hip = float(np.linalg.norm(l_wrist - mid_hip))
    r_wrist_to_mid_hip = float(np.linalg.norm(r_wrist - mid_hip))

    # Hand-to-Head distance
    l_wrist_to_head = float(np.linalg.norm(l_wrist - nose))
    r_wrist_to_head = float(np.linalg.norm(r_wrist - nose))

    # Hand-to-Hand distance
    wrist_to_wrist = float(np.linalg.norm(l_wrist - r_wrist))

    # Hand position relative to torso (vertical elevation relative to mid-hip and mid-shoulder)
    # Since y increases downwards in image coordinates, (mid_hip[1] - wrist[1]) is positive when wrist is above hip.
    l_wrist_elevation = float(mid_hip[1] - l_wrist[1])
    r_wrist_elevation = float(mid_hip[1] - r_wrist[1])

    # Concealment / Waist interaction flags: hand is near waist/hips
    l_hand_near_waist = 1.0 if l_wrist_to_mid_hip < 0.4 else 0.0
    r_hand_near_waist = 1.0 if r_wrist_to_mid_hip < 0.4 else 0.0

    features = {
        "l_wrist_to_same_hip": l_wrist_to_same_hip,
        "r_wrist_to_same_hip": r_wrist_to_same_hip,
        "l_wrist_to_opp_hip": l_wrist_to_opp_hip,
        "r_wrist_to_opp_hip": r_wrist_to_opp_hip,
        "l_wrist_to_mid_hip": l_wrist_to_mid_hip,
        "r_wrist_to_mid_hip": r_wrist_to_mid_hip,
        "l_wrist_to_head": l_wrist_to_head,
        "r_wrist_to_head": r_wrist_to_head,
        "wrist_to_wrist": wrist_to_wrist,
        "l_wrist_elevation": l_wrist_elevation,
        "r_wrist_elevation": r_wrist_elevation,
        "l_hand_near_waist": l_hand_near_waist,
        "r_hand_near_waist": r_hand_near_waist,
    }

    # Shelf Polygon Interaction
    l_shelf_dist = 1.0
    r_shelf_dist = 1.0
    is_in_shelf_zone = 0.0

    if shelf_polygons:
        min_l = float("inf")
        min_r = float("inf")
        for poly in shelf_polygons:
            poly = np.array(poly, dtype=np.float32)
            # Distance from point to polygon approximation
            dist_l = float(np.min(np.linalg.norm(poly - l_wrist, axis=1)))
            dist_r = float(np.min(np.linalg.norm(poly - r_wrist, axis=1)))
            min_l = min(min_l, dist_l)
            min_r = min(min_r, dist_r)
        if min_l != float("inf"):
            l_shelf_dist = min_l
        if min_r != float("inf"):
            r_shelf_dist = min_r
        if l_shelf_dist < 0.3 or r_shelf_dist < 0.3:
            is_in_shelf_zone = 1.0

    features["l_shelf_dist"] = l_shelf_dist
    features["r_shelf_dist"] = r_shelf_dist
    features["is_in_shelf_zone"] = is_in_shelf_zone

    return features
