"""Step 3: Data Preprocessing and Geometric Feature Extraction from 21 MediaPipe Hand Landmarks."""

import math
from typing import Sequence
import numpy as np


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Paper Eq. (3): Euclidean distance between two 2D points."""
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def fingers_up(landmarks_px: Sequence[Sequence[float]]) -> list[int]:
    """
    Determine mirror- and handedness-invariant binary finger extension vector
    [Thumb, Index, Middle, Ring, Pinky].
    - Thumb: uses palm lateral axis u = normalize(LM5 - LM17) and anatomical ratios
      so left/right hands and mirrored webcams yield identical extension detection.
    - Other 4 fingers: Extended_i = 1 if y_tip < y_pip AND d(wrist, tip) > d(wrist, pip).
    """
    if len(landmarks_px) != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(landmarks_px)}")

    state: list[int] = []

    # Thumb: anatomical projection onto palm lateral vector u (from Pinky MCP 17 to Index MCP 5)
    ux_raw = float(landmarks_px[5][0]) - float(landmarks_px[17][0])
    uy_raw = float(landmarks_px[5][1]) - float(landmarks_px[17][1])
    u_norm = max(math.hypot(ux_raw, uy_raw), 1e-6)
    ux, uy = ux_raw / u_norm, uy_raw / u_norm

    palm_ref = max(euclidean_distance(landmarks_px[0], landmarks_px[9]), 1e-6)
    proj_4_3 = (float(landmarks_px[4][0]) - float(landmarks_px[3][0])) * ux + (
        float(landmarks_px[4][1]) - float(landmarks_px[3][1])
    ) * uy
    d_4_17 = euclidean_distance(landmarks_px[4], landmarks_px[17])
    d_3_17 = max(euclidean_distance(landmarks_px[3], landmarks_px[17]), 1e-6)
    d_4_10 = euclidean_distance(landmarks_px[4], landmarks_px[10])

    thumb_extended = 1 if (proj_4_3 > 0.0 and d_4_17 > 1.06 * d_3_17 and d_4_10 > 0.45 * palm_ref) else 0
    state.append(thumb_extended)

    # Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
    for tip_idx in (8, 12, 16, 20):
        pip_idx = tip_idx - 2
        extended = (
            1
            if (
                float(landmarks_px[tip_idx][1]) < float(landmarks_px[pip_idx][1])
                and euclidean_distance(landmarks_px[0], landmarks_px[tip_idx])
                > euclidean_distance(landmarks_px[0], landmarks_px[pip_idx])
            )
            else 0
        )
        state.append(extended)

    return state


def compute_hand_scale(landmarks_px: Sequence[Sequence[float]]) -> float:
    """Compute reference hand scale (max distance from wrist or bounding box diagonal) to normalize coordinates."""
    wrist_x, wrist_y = float(landmarks_px[0][0]), float(landmarks_px[0][1])
    max_dist = max(
        math.hypot(float(pt[0]) - wrist_x, float(pt[1]) - wrist_y)
        for pt in landmarks_px
    )
    return max(max_dist, 1e-6)


def extract_features(landmarks_px: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Extract a 54-dimensional translation-, scale-, and mirror/chirality-invariant feature vector:
    - [0..41] (42 values): Wrist-relative (x * chirality_sign, y) coordinates normalized by hand scale
    - [42..46] (5 values): Binary finger extension state [Thumb, Index, Middle, Ring, Pinky]
    - [47..53] (7 values): Key normalized geometric distances & thumb outward ratio:
        * d(thumb_tip=4, index_tip=8) / palm_size      (Pinch Index vs L-Zoom)
        * d(thumb_tip=4, middle_tip=12) / palm_size    (Pinch Middle Right-Click)
        * d(index_tip=8, middle_tip=12) / palm_size
        * d(middle_tip=12, ring_tip=16) / palm_size
        * d(ring_tip=16, pinky_tip=20) / palm_size
        * d(thumb_tip=4, pinky_tip=20) / palm_size     (Shaka Brightness)
        * d(thumb_tip=4, pinky_mcp=17) / d(thumb_ip=3, pinky_mcp=17) (Thumb outward ratio)
    """
    if len(landmarks_px) != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(landmarks_px)}")

    wrist_x, wrist_y = float(landmarks_px[0][0]), float(landmarks_px[0][1])
    scale = compute_hand_scale(landmarks_px)
    chirality_sign = -1.0 if float(landmarks_px[5][0]) > float(landmarks_px[17][0]) else 1.0

    # 1. 42 canonicalized relative coordinates
    rel_coords: list[float] = []
    for pt in landmarks_px:
        rel_coords.append(((float(pt[0]) - wrist_x) * chirality_sign) / scale)
        rel_coords.append((float(pt[1]) - wrist_y) / scale)

    # 2. 5 binary finger states
    f_up = [float(v) for v in fingers_up(landmarks_px)]

    # 3. 7 key normalized geometric features
    palm_ref = max(euclidean_distance(landmarks_px[0], landmarks_px[9]), 1e-6)
    d_thumb_index = euclidean_distance(landmarks_px[4], landmarks_px[8]) / palm_ref
    d_thumb_middle = euclidean_distance(landmarks_px[4], landmarks_px[12]) / palm_ref
    d_index_middle = euclidean_distance(landmarks_px[8], landmarks_px[12]) / palm_ref
    d_middle_ring = euclidean_distance(landmarks_px[12], landmarks_px[16]) / palm_ref
    d_ring_pinky = euclidean_distance(landmarks_px[16], landmarks_px[20]) / palm_ref
    d_thumb_pinky = euclidean_distance(landmarks_px[4], landmarks_px[20]) / palm_ref
    thumb_outward_ratio = euclidean_distance(landmarks_px[4], landmarks_px[17]) / max(
        euclidean_distance(landmarks_px[3], landmarks_px[17]), 1e-6
    )

    distances = [
        d_thumb_index,
        d_thumb_middle,
        d_index_middle,
        d_middle_ring,
        d_ring_pinky,
        d_thumb_pinky,
        thumb_outward_ratio,
    ]

    features = np.array(rel_coords + f_up + distances, dtype=np.float64)
    return features


def get_feature_names() -> list[str]:
    """Return human-readable names for all 54 features."""
    names: list[str] = []
    for i in range(21):
        names.append(f"lm{i}_x")
        names.append(f"lm{i}_y")
    names.extend(["thumb_up", "index_up", "middle_up", "ring_up", "pinky_up"])
    names.extend([
        "dist_thumb_index_norm",
        "dist_thumb_middle_norm",
        "dist_index_middle_norm",
        "dist_middle_ring_norm",
        "dist_ring_pinky_norm",
        "dist_thumb_pinky_norm",
        "thumb_outward_ratio",
    ])
    return names
