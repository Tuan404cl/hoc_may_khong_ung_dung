from src.floating_ui import build_debug_telemetry
from tests.test_features import make_base_hand, set_finger_extended


def test_build_debug_telemetry_extracts_finger_states_and_distances():
    pts = make_base_hand(320.0, 240.0)
    set_finger_extended(pts, 1, True)

    info = {
        "pose_id": 0,
        "pose_name": "POINT_POSE",
        "action": "CURSOR_MOVE",
        "confidence": 0.96,
        "model_key": "KNN",
        "model_name": "K-Nearest Neighbors (KNN, k=5)",
        "cursor_pos": (960, 540),
    }

    telemetry = build_debug_telemetry(pts, info, fps=60.0)
    assert telemetry["fingers_up"] == [0, 1, 0, 0, 0]
    assert telemetry["fingers_str"] == "[Cái:0 | Trỏ:1 | Giữa:0 | Áp:0 | Út:0]"
    assert telemetry["pinch_dist_px"] > 0.0
    assert telemetry["thumb_middle_dist_px"] > 0.0
    assert telemetry["index_middle_dist_px"] > 0.0
    assert telemetry["action"] == "CURSOR_MOVE"
    assert telemetry["pose_name"] == "POINT_POSE"
