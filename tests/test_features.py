import numpy as np
import pytest
from src.config import (
    POSE_CLASSES,
    SMOOTHING_ALPHA,
    ACTION_DELAY,
    PINCH_THRESHOLD,
    CONFIRM_FRAMES,
    DRAG_HOLD_DURATION,
    DOUBLE_PINCH_WINDOW,
    STATIC_HOLD_DURATION,
)
from src.features import euclidean_distance, fingers_up, extract_features, get_feature_names


def make_base_hand(offset_x: float = 300.0, offset_y: float = 350.0, scale: float = 1.0) -> list[tuple[float, float]]:
    """Create a 21-landmark hand template with configurable anatomical finger states (all folded by default)."""
    pts = [(offset_x, offset_y)] * 21
    pts[0] = (offset_x, offset_y)
    # Thumb (1..4) anatomically folded inward toward middle PIP (LM10)
    pts[1] = (offset_x - 20 * scale, offset_y - 18 * scale)
    pts[2] = (offset_x - 32 * scale, offset_y - 40 * scale)
    pts[3] = (offset_x - 30 * scale, offset_y - 60 * scale)
    pts[4] = (offset_x - 18 * scale, offset_y - 65 * scale)
    # Fingers (5..8 Index, 9..12 Middle, 13..16 Ring, 17..20 Pinky)
    finger_x_offsets = [-25, -8, 10, 28]
    for f_idx, fx in enumerate(finger_x_offsets):
        base = 5 + f_idx * 4
        bx = offset_x + fx * scale
        pts[base] = (bx, offset_y - 50 * scale)       # MCP
        pts[base + 1] = (bx, offset_y - 75 * scale)   # PIP
        pts[base + 2] = (bx, offset_y - 65 * scale)   # DIP (folded down: y > PIP)
        pts[base + 3] = (bx, offset_y - 55 * scale)   # TIP (folded down: y_tip > y_pip)
    return pts


def set_finger_extended(pts: list[tuple[float, float]], finger_idx: int, extended: bool = True, scale: float = 1.0):
    """finger_idx: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky"""
    ox, oy = pts[0]
    if finger_idx == 0:
        if extended:
            # Anatomically outward on the LM5 (thumb) side
            pts[3] = (ox - 48 * scale, oy - 55 * scale)
            pts[4] = (ox - 68 * scale, oy - 68 * scale)
        else:
            # Anatomically tucked inward across palm
            pts[3] = (ox - 30 * scale, oy - 60 * scale)
            pts[4] = (ox - 18 * scale, oy - 65 * scale)
    else:
        base = 5 + (finger_idx - 1) * 4
        bx, pip_y = pts[base + 1]
        if extended:
            pts[base + 2] = (bx, pip_y - 20 * scale)
            pts[base + 3] = (bx, pip_y - 40 * scale)  # y_tip < y_pip
        else:
            pts[base + 2] = (bx, pip_y + 10 * scale)
            pts[base + 3] = (bx, pip_y + 20 * scale)  # y_tip > y_pip


def test_config_constants():
    assert len(POSE_CLASSES) == 10
    assert SMOOTHING_ALPHA == 8.0
    assert ACTION_DELAY == 0.4
    assert PINCH_THRESHOLD == 30.0
    assert CONFIRM_FRAMES == 3
    assert DRAG_HOLD_DURATION == 0.35
    assert DOUBLE_PINCH_WINDOW == 0.45
    assert STATIC_HOLD_DURATION == 0.85


def test_euclidean_distance():
    assert pytest.approx(euclidean_distance((0.0, 0.0), (3.0, 4.0))) == 5.0
    assert pytest.approx(euclidean_distance((10.0, 20.0), (10.0, 20.0))) == 0.0


def test_fingers_up_all_configurations():
    pts = make_base_hand()
    assert fingers_up(pts) == [0, 0, 0, 0, 0]

    # Point pose [0, 1, 0, 0, 0]
    set_finger_extended(pts, 1, True)
    assert fingers_up(pts) == [0, 1, 0, 0, 0]

    # Extend middle -> [0, 1, 1, 0, 0]
    set_finger_extended(pts, 2, True)
    assert fingers_up(pts) == [0, 1, 1, 0, 0]

    # Extend all 5 fingers -> [1, 1, 1, 1, 1]
    for i in range(5):
        set_finger_extended(pts, i, True)
    assert fingers_up(pts) == [1, 1, 1, 1, 1]


def test_extract_features_shape_and_translation_invariance():
    assert len(get_feature_names()) == 54

    pts1 = make_base_hand(offset_x=100.0, offset_y=200.0, scale=1.0)
    set_finger_extended(pts1, 1, True)

    pts2 = make_base_hand(offset_x=450.0, offset_y=380.0, scale=1.0)
    set_finger_extended(pts2, 1, True)

    feat1 = extract_features(pts1)
    feat2 = extract_features(pts2)

    assert feat1.shape == (54,)
    assert np.allclose(feat1, feat2, atol=1e-5)


@pytest.mark.parametrize(
    "expected_state",
    [
        [0, 0, 0, 0, 0],  # FIST_IDLE_POSE
        [0, 1, 0, 0, 0],  # POINT_POSE
        [1, 0, 0, 0, 1],  # SHAKA_BRIGHTNESS_POSE
        [1, 1, 1, 1, 1],  # OPEN_PALM_POSE
    ],
)
def test_mirror_and_handedness_invariance(expected_state):
    pts = make_base_hand(offset_x=300.0, offset_y=350.0, scale=1.2)
    for idx, flag in enumerate(expected_state):
        set_finger_extended(pts, idx, bool(flag), scale=1.2)

    # Horizontally mirror around x = 320 (simulating left vs right hand or mirrored webcam)
    mirrored_pts = [(640.0 - px, py) for (px, py) in pts]

    assert fingers_up(pts) == expected_state
    assert fingers_up(mirrored_pts) == expected_state

    feat_orig = extract_features(pts)
    feat_mirror = extract_features(mirrored_pts)
    assert feat_orig.shape == (54,)
    assert np.allclose(feat_orig, feat_mirror, atol=1e-5)
