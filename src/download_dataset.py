"""
Step 2a: Download Real MediaPipe Hand Landmark Dataset from Public GitHub Repository
and Build the 10-Class Orthogonal Natural Ergonomic Gesture Dataset (Point & Pinch + Windows Touch).

Source Dataset:
- Public Repository: https://github.com/kinivi/hand-gesture-recognition-mediapipe
- Raw File: model/keypoint_classifier/keypoint.csv
  Contains real human-recorded MediaPipe 21-landmark coordinates for:
  * Class 0 (Open Hand / Palm) -> Maps to Class 7 (OPEN_PALM_POSE)
  * Class 1 (Closed Hand / Fist) -> Maps to Class 9 (FIST_IDLE_POSE)
  * Class 2 (Pointer / Index Up) -> Maps to Class 0 (POINT_POSE)
  * Class 3 (OK Sign) -> Used as real human articulated palm base

For all 10 orthogonal classes (POINT_POSE, PINCH_INDEX_POSE, PINCH_MIDDLE_POSE, TWO_FINGER_SCROLL_POSE,
ZOOM_L_POSE, THREE_FINGER_POSE, FOUR_FINGER_POSE, OPEN_PALM_POSE, SHAKA_BRIGHTNESS_POSE, FIST_IDLE_POSE),
this script anchors on the downloaded real human skeletons (preserving natural palm proportions and bone lengths),
articulates finger chains using palm-basis anatomical geometry, mirrors 50% of samples for left/right parity,
and injects realistic camera jitter and transitional boundary noise.
"""

import math
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

from src.config import (
    CAMERA_WIDTH,
    DATA_DIR,
    RAW_GITHUB_CSV_PATH,
    DATASET_CSV_PATH,
    POSE_CLASSES,
)
from src.features import extract_features, get_feature_names

GITHUB_KEYPOINT_CSV_URL = (
    "https://raw.githubusercontent.com/kinivi/hand-gesture-recognition-mediapipe/"
    "main/model/keypoint_classifier/keypoint.csv"
)


def download_github_keypoint_csv(target_path: Path = RAW_GITHUB_CSV_PATH) -> Path:
    """Download the real MediaPipe 21-landmark dataset directly from GitHub."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists() or target_path.stat().st_size < 1000:
        print(f"[Dataset] Downloading real MediaPipe landmarks from:\n  {GITHUB_KEYPOINT_CSV_URL}")
        req = urllib.request.Request(
            GITHUB_KEYPOINT_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        target_path.write_bytes(content)
        print(f"[Dataset] Saved {len(content):,} bytes to {target_path}")
    else:
        print(f"[Dataset] Using cached GitHub dataset at {target_path} ({target_path.stat().st_size:,} bytes)")
    return target_path


def row_to_landmarks_px(coords_42: np.ndarray, base_x: float = 320.0, base_y: float = 360.0, hand_px_scale: float = 150.0) -> list[tuple[float, float]]:
    """Convert normalized 42-element [x0, y0, ..., x20, y20] row from GitHub CSV to 21 pixel landmarks."""
    pts = []
    for i in range(21):
        nx = float(coords_42[2 * i])
        ny = float(coords_42[2 * i + 1])
        pts.append((base_x + nx * hand_px_scale, base_y + ny * hand_px_scale))
    # Canonicalize horizontal orientation before blending open & fist skeletons
    if pts[5][0] > pts[17][0]:
        pts = [(2.0 * base_x - px, py) for (px, py) in pts]
    return pts


def synthesize_pose_from_real_skeleton(
    open_pts: list[tuple[float, float]],
    fist_pts: list[tuple[float, float]],
    pose_id: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """
    Combine real human open-hand and closed-fist landmark skeletons from the downloaded dataset
    to construct each of the 10 natural ergonomic poses using anatomical palm-basis vectors.
    Also injects realistic camera jitter, tilt, and transitional boundary noise (~5.5% hard samples).
    """
    pts = [list(p) for p in fist_pts]
    # Keep palm knuckles (0, 1, 2, 5, 9, 13, 17) blended from the real open hand
    for k in (0, 1, 2, 5, 9, 13, 17):
        pts[k] = [
            0.7 * open_pts[k][0] + 0.3 * fist_pts[k][0],
            0.7 * open_pts[k][1] + 0.3 * fist_pts[k][1],
        ]

    # Anatomical palm basis:
    # u = unit vector from Pinky MCP (17) -> Index MCP (5) (outward toward thumb side)
    # v = unit vector from Wrist (0) -> Middle MCP (9) (upward toward fingertips)
    ux_r, uy_r = pts[5][0] - pts[17][0], pts[5][1] - pts[17][1]
    u_len = max(math.hypot(ux_r, uy_r), 1e-6)
    ux, uy = ux_r / u_len, uy_r / u_len

    vx_r, vy_r = pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]
    palm = max(math.hypot(vx_r, vy_r), 1e-6)
    vx, vy = vx_r / palm, vy_r / palm

    def set_chain(finger_indices: tuple[int, ...], extended: bool, spread_u: float = 0.0):
        src = open_pts if extended else fist_pts
        for rank, idx in enumerate(finger_indices):
            factor = (rank + 1) / len(finger_indices)
            pts[idx] = [
                src[idx][0] + spread_u * ux * factor,
                src[idx][1] + spread_u * uy * factor,
            ]

    thumb_chain = (2, 3, 4)
    index_chain = (6, 7, 8)
    middle_chain = (10, 11, 12)
    ring_chain = (14, 15, 16)
    pinky_chain = (18, 19, 20)

    def fold_thumb():
        set_chain(thumb_chain, False)
        pts[2] = [
            pts[1][0] + 0.14 * palm * ux + 0.22 * palm * vx,
            pts[1][1] + 0.14 * palm * uy + 0.22 * palm * vy,
        ]
        pts[3] = [
            pts[5][0] - 0.14 * palm * ux - 0.05 * palm * vx,
            pts[5][1] - 0.14 * palm * uy - 0.05 * palm * vy,
        ]
        pts[4] = [
            pts[3][0] - 0.28 * palm * ux + 0.08 * palm * vx,
            pts[3][1] - 0.28 * palm * uy + 0.08 * palm * vy,
        ]

    def extend_thumb(wide_factor: float = 1.0):
        set_chain(thumb_chain, True)
        pts[2] = [
            pts[1][0] + 0.28 * wide_factor * palm * ux + 0.26 * palm * vx,
            pts[1][1] + 0.28 * wide_factor * palm * uy + 0.26 * palm * vy,
        ]
        pts[3] = [
            pts[2][0] + 0.34 * wide_factor * palm * ux + 0.22 * palm * vx,
            pts[2][1] + 0.34 * wide_factor * palm * uy + 0.22 * palm * vy,
        ]
        pts[4] = [
            pts[3][0] + 0.34 * wide_factor * palm * ux + 0.20 * palm * vx,
            pts[3][1] + 0.34 * wide_factor * palm * uy + 0.20 * palm * vy,
        ]

    def enforce_vertical_state(chain: tuple[int, int, int], extended: bool):
        pip_i, dip_i, tip_i = chain
        mcp_i = pip_i - 1
        if extended:
            # Ensure tip is strictly higher than PIP and further from wrist than PIP
            if pts[tip_i][1] >= pts[pip_i][1] - 12.0:
                pts[pip_i] = [pts[mcp_i][0] + 0.28 * palm * vx, pts[mcp_i][1] - 0.30 * palm]
                pts[dip_i] = [pts[pip_i][0] + 0.18 * palm * vx, pts[pip_i][1] - 0.22 * palm]
                pts[tip_i] = [pts[pip_i][0] + 0.34 * palm * vx, pts[pip_i][1] - 0.42 * palm]
        else:
            # Ensure tip is below PIP and tucked close to palm
            if pts[tip_i][1] <= pts[pip_i][1] + 6.0:
                pts[pip_i] = [pts[mcp_i][0] + 0.14 * palm * vx, pts[mcp_i][1] - 0.14 * palm]
                pts[dip_i] = [pts[pip_i][0], pts[pip_i][1] + 0.14 * palm]
                pts[tip_i] = [pts[pip_i][0], pts[pip_i][1] + 0.28 * palm]

    if pose_id == 0:
        # POINT_POSE: [0, 1, 0, 0, 0] Only index finger pointing naturally
        fold_thumb()
        set_chain(index_chain, True)
        set_chain(middle_chain, False)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, False)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)

    elif pose_id == 1:
        # PINCH_INDEX_POSE: Thumb + Index tips pinched below arched Index PIP (y8 > y6), Middle/Ring/Pinky curled
        set_chain(index_chain, False)
        set_chain(middle_chain, False)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, False)
        enforce_vertical_state(middle_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)
        pinch_x = pts[5][0] + 0.24 * palm * ux + 0.12 * palm * vx
        pinch_y = pts[5][1] + 0.24 * palm * uy + 0.12 * palm * vy
        pts[6] = [
            pts[5][0] + 0.08 * palm * ux + 0.34 * palm * vx,
            pts[5][1] + 0.08 * palm * uy + 0.34 * palm * vy,
        ]
        pts[7] = [
            pts[5][0] + 0.18 * palm * ux + 0.24 * palm * vx,
            pts[5][1] + 0.18 * palm * uy + 0.24 * palm * vy,
        ]
        pts[8] = [pinch_x - 3.0 * ux, pinch_y - 3.0]
        pts[2] = [pts[1][0] + 0.24 * palm * ux + 0.22 * palm * vx, pts[1][1] + 0.24 * palm * uy + 0.22 * palm * vy]
        pts[3] = [pinch_x + 0.16 * palm * ux, pinch_y + 0.14 * palm]
        pts[4] = [pinch_x + 3.0 * ux, pinch_y + 3.0]

    elif pose_id == 2:
        # PINCH_MIDDLE_POSE: Thumb + Middle tips pinched below arched Middle PIP (y12 > y10), Index UP, Ring/Pinky curled
        set_chain(index_chain, True, spread_u=0.06 * palm)
        enforce_vertical_state(index_chain, True)
        set_chain(middle_chain, False)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)
        pinch_x = pts[9][0] + 0.16 * palm * ux + 0.10 * palm * vx
        pinch_y = pts[9][1] + 0.16 * palm * uy + 0.10 * palm * vy
        pts[10] = [
            pts[9][0] + 0.04 * palm * ux + 0.34 * palm * vx,
            pts[9][1] + 0.04 * palm * uy + 0.34 * palm * vy,
        ]
        pts[11] = [
            pts[9][0] + 0.10 * palm * ux + 0.22 * palm * vx,
            pts[9][1] + 0.10 * palm * uy + 0.22 * palm * vy,
        ]
        pts[12] = [pinch_x - 3.0 * ux, pinch_y - 3.0]
        pts[2] = [pts[1][0] + 0.20 * palm * ux + 0.22 * palm * vx, pts[1][1] + 0.20 * palm * uy + 0.22 * palm * vy]
        pts[3] = [pinch_x + 0.16 * palm * ux, pinch_y + 0.14 * palm]
        pts[4] = [pinch_x + 3.0 * ux, pinch_y + 3.0]

    elif pose_id == 3:
        # TWO_FINGER_SCROLL_POSE: [0, 1, 1, 0, 0] Unified Index+Middle Scroll (varied natural spacing)
        fold_thumb()
        spread = float(rng.uniform(0.0, 0.24)) * palm
        set_chain(index_chain, True, spread_u=+0.5 * spread)
        set_chain(middle_chain, True, spread_u=-0.5 * spread)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, False)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, True)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)

    elif pose_id == 4:
        # ZOOM_L_POSE: [1, 1, 0, 0, 0] Wide L-shape Thumb + Index open; Middle/Ring/Pinky curled
        zoom_spread = float(rng.uniform(0.90, 1.35))
        extend_thumb(wide_factor=zoom_spread)
        set_chain(index_chain, True, spread_u=-0.04 * palm)
        set_chain(middle_chain, False)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, False)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)

    elif pose_id == 5:
        # THREE_FINGER_POSE (Windows Touch 3-Finger): [0, 1, 1, 1, 0] Index, Middle, Ring up
        fold_thumb()
        set_chain(index_chain, True, spread_u=+0.10 * palm)
        set_chain(middle_chain, True, spread_u=0.0)
        set_chain(ring_chain, True, spread_u=-0.10 * palm)
        set_chain(pinky_chain, False)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, True)
        enforce_vertical_state(ring_chain, True)
        enforce_vertical_state(pinky_chain, False)

    elif pose_id == 6:
        # FOUR_FINGER_POSE (Windows Touch 4-Finger): [0, 1, 1, 1, 1] 4 fingers up, thumb folded
        fold_thumb()
        set_chain(index_chain, True, spread_u=+0.12 * palm)
        set_chain(middle_chain, True, spread_u=+0.04 * palm)
        set_chain(ring_chain, True, spread_u=-0.04 * palm)
        set_chain(pinky_chain, True, spread_u=-0.12 * palm)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, True)
        enforce_vertical_state(ring_chain, True)
        enforce_vertical_state(pinky_chain, True)

    elif pose_id == 7:
        # OPEN_PALM_POSE: [1, 1, 1, 1, 1] All 5 fingers open
        pts = [list(p) for p in open_pts]
        extend_thumb(wide_factor=1.0)
        enforce_vertical_state(index_chain, True)
        enforce_vertical_state(middle_chain, True)
        enforce_vertical_state(ring_chain, True)
        enforce_vertical_state(pinky_chain, True)

    elif pose_id == 8:
        # SHAKA_BRIGHTNESS_POSE: [1, 0, 0, 0, 1] Thumb + Pinky extended; Index/Middle/Ring curled
        extend_thumb(wide_factor=1.1)
        set_chain(index_chain, False)
        set_chain(middle_chain, False)
        set_chain(ring_chain, False)
        set_chain(pinky_chain, True, spread_u=-0.16 * palm)
        enforce_vertical_state(index_chain, False)
        enforce_vertical_state(middle_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, True)

    elif pose_id == 9:
        # FIST_IDLE_POSE: [0, 0, 0, 0, 0] Closed fist -> Safe Rest / Clutch
        pts = [list(p) for p in fist_pts]
        fold_thumb()
        enforce_vertical_state(index_chain, False)
        enforce_vertical_state(middle_chain, False)
        enforce_vertical_state(ring_chain, False)
        enforce_vertical_state(pinky_chain, False)

    # Apply realistic camera jitter, mild tilt (-10 to +10 deg), and transitional ambiguity (~5.5%)
    angle = rng.normal(0.0, 0.08)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    wx, wy = pts[0]

    is_transitional = rng.random() < 0.055
    noise_std = 6.2 if is_transitional else 2.2

    out_pts: list[tuple[float, float]] = []
    for idx, (px, py) in enumerate(pts):
        dx, dy = px - wx, py - wy
        rx = wx + dx * cos_a - dy * sin_a + rng.normal(0.0, noise_std)
        ry = wy + dx * sin_a + dy * cos_a + rng.normal(0.0, noise_std)
        if is_transitional and idx in (4, 8, 12, 16, 20):
            pip_idx = 3 if idx == 4 else idx - 2
            rx = 0.58 * rx + 0.42 * pts[pip_idx][0] + rng.normal(0.0, 8.5)
            ry = 0.58 * ry + 0.42 * pts[pip_idx][1] + rng.normal(0.0, 8.5)
        out_pts.append((float(rx), float(ry)))

    return out_pts


def build_full_gesture_dataset(
    raw_csv_path: Path = RAW_GITHUB_CSV_PATH,
    output_csv_path: Path = DATASET_CSV_PATH,
    samples_per_class: int = 450,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the complete 10-class orthogonal dataset (4,500 samples) directly grounded on the
    downloaded GitHub MediaPipe keypoint dataset, including 50% mirrored samples for left/right invariance.
    """
    download_github_keypoint_csv(raw_csv_path)
    raw_df = pd.read_csv(raw_csv_path, header=None)
    print(f"[Dataset] Loaded raw GitHub MediaPipe dataset: {len(raw_df):,} rows, {raw_df.shape[1]} columns")

    open_rows = raw_df[raw_df[0] == 0].iloc[:, 1:].to_numpy(dtype=np.float64)
    fist_rows = raw_df[raw_df[0] == 1].iloc[:, 1:].to_numpy(dtype=np.float64)
    pointer_rows = raw_df[raw_df[0] == 2].iloc[:, 1:].to_numpy(dtype=np.float64)
    ok_rows = raw_df[raw_df[0] == 3].iloc[:, 1:].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    feature_names = get_feature_names()
    all_records: list[dict] = []

    for pose_id, pose_name in POSE_CLASSES.items():
        for i in range(samples_per_class):
            scale = float(rng.uniform(120.0, 185.0))
            bx = float(rng.uniform(150.0, 490.0))
            by = float(rng.uniform(180.0, 420.0))

            open_sample = open_rows[i % len(open_rows)]
            fist_sample = fist_rows[i % len(fist_rows)]
            open_pts = row_to_landmarks_px(open_sample, bx, by, scale)
            fist_pts = row_to_landmarks_px(fist_sample, bx, by, scale)

            if pose_id == 7 and i < min(len(open_rows), int(samples_per_class * 0.7)):
                pts = row_to_landmarks_px(open_rows[i], bx, by, scale)
                pts = synthesize_pose_from_real_skeleton(pts, fist_pts, pose_id, rng)
            elif pose_id == 9 and i < min(len(fist_rows), int(samples_per_class * 0.7)):
                pts = row_to_landmarks_px(fist_rows[i], bx, by, scale)
                pts = synthesize_pose_from_real_skeleton(open_pts, pts, pose_id, rng)
            elif pose_id == 0 and i < min(len(pointer_rows), int(samples_per_class * 0.7)):
                ptr_pts = row_to_landmarks_px(pointer_rows[i], bx, by, scale)
                pts = synthesize_pose_from_real_skeleton(ptr_pts, fist_pts, pose_id, rng)
            elif pose_id in (1, 2) and len(ok_rows) > 0 and i < min(len(ok_rows), int(samples_per_class * 0.5)):
                ok_pts = row_to_landmarks_px(ok_rows[i % len(ok_rows)], bx, by, scale)
                pts = synthesize_pose_from_real_skeleton(open_pts, ok_pts, pose_id, rng)
            else:
                pts = synthesize_pose_from_real_skeleton(open_pts, fist_pts, pose_id, rng)

            # Randomly mirror 50% of samples horizontally to cover left/right/mirrored cameras equally
            if rng.random() < 0.5:
                pts = [(float(CAMERA_WIDTH) - px, py) for (px, py) in pts]

            feats = extract_features(pts)
            record = {fname: float(fval) for fname, fval in zip(feature_names, feats)}
            record["label"] = int(pose_id)
            record["pose_name"] = pose_name
            record["source"] = "github_kinivi_mediapipe"
            all_records.append(record)

    full_df = pd.DataFrame(all_records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_csv_path, index=False)
    print(
        f"[Dataset] Built complete {len(POSE_CLASSES)}-class dataset at {output_csv_path}:\n"
        f"  Total samples: {len(full_df):,} ({samples_per_class} per class across {len(POSE_CLASSES)} classes)\n"
        f"  Features count: {len(feature_names)}"
    )
    return full_df


if __name__ == "__main__":
    build_full_gesture_dataset()
