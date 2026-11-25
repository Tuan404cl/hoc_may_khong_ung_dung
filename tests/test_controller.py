import pytest
from src.controller import CursorSmoother, ActionDebouncer, VirtualMouseController
from tests.test_features import make_base_hand, set_finger_extended


def feed_pose(ctrl: VirtualMouseController, clock: list[float], pose_id: int, pts, frames: int = 3, dt: float = 0.02):
    """Feed `frames` consecutive identical poses to pass Temporal Pose Stabilizer."""
    res = None
    for _ in range(frames):
        clock[0] += dt
        res = ctrl.process_frame(pose_id=pose_id, landmarks_px=pts)
    return res


def test_cursor_smoother_ema_formula():
    # Paper Eq. (4): x_smooth_t = x_smooth_{t-1} + (1 / alpha) * (x_raw_t - x_smooth_{t-1})
    smoother = CursorSmoother(alpha=8.0)
    x0, y0 = smoother.update(100.0, 200.0)
    assert (x0, y0) == (100.0, 200.0)

    x1, y1 = smoother.update(180.0, 280.0)
    assert pytest.approx(x1) == 110.0  # 100 + (180 - 100) / 8
    assert pytest.approx(y1) == 210.0  # 200 + (280 - 200) / 8

    # Pinch-approach Cursor Freeze (alpha_override = 32.0) slows cursor jump 4x more
    x2, y2 = smoother.update(174.0, 274.0, alpha_override=32.0)
    assert pytest.approx(x2) == 112.0  # 110 + (174 - 110) / 32
    assert pytest.approx(y2) == 212.0  # 210 + (274 - 210) / 32


def test_action_debouncer_400ms():
    clock = [10.0]
    debouncer = ActionDebouncer(delay=0.4, time_fn=lambda: clock[0])

    assert debouncer.allow_action() is True
    clock[0] = 10.2  # Only 200ms elapsed
    assert debouncer.allow_action() is False
    clock[0] = 10.45  # 450ms elapsed
    assert debouncer.allow_action() is True


def test_single_frame_ghost_noise_rejected():
    clock = [0.0]
    ctrl = VirtualMouseController(dry_run=True, confirm_frames=3, time_fn=lambda: clock[0])
    pts = make_base_hand(320.0, 240.0)

    # Establish steady POINT_POSE
    res_move = feed_pose(ctrl, clock, 0, pts, frames=3, dt=0.02)
    assert res_move["action"] == "CURSOR_MOVE"

    # Inject 2 noisy frames of PINCH_MIDDLE_POSE (Right Click) -> MUST NOT trigger click!
    clock[0] += 0.5
    r1 = ctrl.process_frame(pose_id=2, landmarks_px=pts)
    clock[0] += 0.02
    r2 = ctrl.process_frame(pose_id=2, landmarks_px=pts)
    assert r1["action"] == "CURSOR_MOVE"
    assert r2["action"] == "CURSOR_MOVE"

    # Also test 1-frame noise of PINCH_INDEX_POSE when starting from empty state
    ctrl.reset_states()
    clock[0] += 0.5
    r_init = ctrl.process_frame(pose_id=1, landmarks_px=pts)
    assert r_init["action"] == "NONE"


def test_ergonomic_10_poses_and_windows_touch_actions():
    clock = [0.0]
    ctrl = VirtualMouseController(
        screen_width=1920,
        screen_height=1080,
        dry_run=True,
        confirm_frames=3,
        time_fn=lambda: clock[0],
    )

    # 1. Class 0: POINT_POSE -> CURSOR_MOVE (+ Pinch-Approach Cursor Freeze)
    pts_far = make_base_hand(320.0, 240.0)
    set_finger_extended(pts_far, 1, True)
    res_move = feed_pose(ctrl, clock, 0, pts_far, frames=3, dt=0.02)
    assert res_move["action"] == "CURSOR_MOVE"

    # Simulate pinch approach (d(4,8)/palm_ref < 0.45) freezes cursor movement strongly
    prev_pos = res_move["cursor_pos"]
    pts_approach = make_base_hand(420.0, 240.0)
    set_finger_extended(pts_approach, 1, True)
    pts_approach[4] = (pts_approach[8][0], pts_approach[8][1] + 15.0)
    clock[0] += 0.02
    res_frozen = ctrl.process_frame(pose_id=0, landmarks_px=pts_approach)
    assert res_frozen["action"] == "CURSOR_MOVE"
    assert abs(res_frozen["cursor_pos"][0] - prev_pos[0]) < 15

    # 2. Class 1: PINCH_INDEX_POSE -> Quick Tap (<0.35s) & Release = LEFT_CLICK
    res_hold = feed_pose(ctrl, clock, 1, pts_far, frames=3, dt=0.02)
    assert res_hold["action"] == "PINCH_HOLDING"
    # Release back to POINT_POSE after < 0.35s -> LEFT_CLICK
    res_click = feed_pose(ctrl, clock, 0, pts_far, frames=3, dt=0.02)
    assert res_click["action"] == "LEFT_CLICK"

    # 3. Quick 2nd Pinch Tap within 0.45s -> DOUBLE_CLICK
    feed_pose(ctrl, clock, 1, pts_far, frames=3, dt=0.02)
    res_dbl = feed_pose(ctrl, clock, 0, pts_far, frames=3, dt=0.02)
    assert res_dbl["action"] == "DOUBLE_CLICK"

    # 4. Hold PINCH_INDEX_POSE >= 0.35s -> DRAG_START -> DRAG_MOVE -> DRAG_RELEASE
    ctrl.reset_states()
    feed_pose(ctrl, clock, 1, make_base_hand(300.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.36
    res_drag_start = ctrl.process_frame(pose_id=1, landmarks_px=make_base_hand(300.0, 240.0))
    assert res_drag_start["action"] == "DRAG_START"
    assert ctrl.is_dragging is True

    clock[0] += 0.03
    res_drag_move = ctrl.process_frame(pose_id=1, landmarks_px=make_base_hand(360.0, 260.0))
    assert res_drag_move["action"] == "DRAG_MOVE"

    res_drag_end = feed_pose(ctrl, clock, 0, make_base_hand(360.0, 260.0), frames=3, dt=0.02)
    assert ctrl.is_dragging is False
    assert res_drag_end["drag_released"] is True
    assert res_drag_end["action"] == "DRAG_RELEASE"

    # 5. Class 2: PINCH_MIDDLE_POSE -> RIGHT_CLICK (latched once per pinch)
    ctrl.reset_states()
    clock[0] += 0.5
    res_rc1 = feed_pose(ctrl, clock, 2, pts_far, frames=3, dt=0.02)
    assert res_rc1["action"] == "RIGHT_CLICK"
    clock[0] += 0.5
    res_rc2 = ctrl.process_frame(pose_id=2, landmarks_px=pts_far)
    assert res_rc2["action"] == "NONE"

    # 6. Class 3: TWO_FINGER_SCROLL_POSE -> SCROLL_VERTICAL & SCROLL_HORIZONTAL
    ctrl.reset_states()
    feed_pose(ctrl, clock, 3, make_base_hand(320.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.05
    res_sv = ctrl.process_frame(pose_id=3, landmarks_px=make_base_hand(320.0, 190.0))
    assert res_sv["action"] == "SCROLL_VERTICAL"

    ctrl.reset_states()
    feed_pose(ctrl, clock, 3, make_base_hand(320.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.05
    res_sh = ctrl.process_frame(pose_id=3, landmarks_px=make_base_hand(380.0, 240.0))
    assert res_sh["action"] == "SCROLL_HORIZONTAL"

    # 7. Class 4: ZOOM_L_POSE -> ZOOM_IN & ZOOM_OUT
    ctrl.reset_states()
    clock[0] += 0.5
    p_l1 = make_base_hand(320.0, 240.0)
    p_l1[4] = (280.0, 220.0)
    p_l1[8] = (290.0, 180.0)
    feed_pose(ctrl, clock, 4, p_l1, frames=3, dt=0.02)
    clock[0] += 0.5
    p_l2 = make_base_hand(320.0, 240.0)
    p_l2[4] = (240.0, 220.0)
    p_l2[8] = (300.0, 150.0)
    res_zi = ctrl.process_frame(pose_id=4, landmarks_px=p_l2)
    assert res_zi["action"] == "ZOOM_IN"

    clock[0] += 0.5
    res_zo = ctrl.process_frame(pose_id=4, landmarks_px=p_l1)
    assert res_zo["action"] == "ZOOM_OUT"

    # 8. Class 5: THREE_FINGER_POSE (Swipes + 0.85s wall-clock WIN_SEARCH)
    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 5, make_base_hand(320.0, 250.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(320.0, 200.0))["action"] == "WIN_TASK_VIEW"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 5, make_base_hand(320.0, 200.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(320.0, 260.0))["action"] == "WIN_SHOW_DESKTOP"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 5, make_base_hand(300.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(360.0, 240.0))["action"] == "WIN_SWITCH_WINDOW_NEXT"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 5, make_base_hand(360.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(300.0, 240.0))["action"] == "WIN_SWITCH_WINDOW_PREV"

    # High-FPS guard: 6 fast frames at 100 FPS (0.06s) must NOT fire WIN_SEARCH; >=0.85s static hold DOES fire
    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 5, make_base_hand(320.0, 240.0), frames=3, dt=0.01)
    for _ in range(6):
        clock[0] += 0.01
        r_fast = ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(320.0, 240.0))
        assert r_fast["action"] == "NONE"
    clock[0] += 0.85
    assert ctrl.process_frame(pose_id=5, landmarks_px=make_base_hand(320.0, 240.0))["action"] == "WIN_SEARCH"

    # 9. Class 6: FOUR_FINGER_POSE (Virtual Desktop + Snap + 0.85s WIN_ACTION_CENTER)
    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 6, make_base_hand(300.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=6, landmarks_px=make_base_hand(360.0, 240.0))["action"] == "WIN_VIRTUAL_DESKTOP_RIGHT"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 6, make_base_hand(320.0, 250.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=6, landmarks_px=make_base_hand(320.0, 200.0))["action"] == "WIN_SNAP_MAXIMIZE"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 6, make_base_hand(320.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.90
    assert ctrl.process_frame(pose_id=6, landmarks_px=make_base_hand(320.0, 240.0))["action"] == "WIN_ACTION_CENTER"

    # 10. Class 7: OPEN_PALM_POSE -> VOLUME_INCREASE / DECREASE / WIN_WIDGETS_PANEL
    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 7, make_base_hand(320.0, 260.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=7, landmarks_px=make_base_hand(320.0, 210.0))["action"] == "VOLUME_INCREASE"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 7, make_base_hand(320.0, 210.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=7, landmarks_px=make_base_hand(320.0, 260.0))["action"] == "VOLUME_DECREASE"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 7, make_base_hand(280.0, 240.0), frames=3, dt=0.02)
    clock[0] += 0.5
    assert ctrl.process_frame(pose_id=7, landmarks_px=make_base_hand(340.0, 240.0))["action"] == "WIN_WIDGETS_PANEL"

    # 11. Class 8: SHAKA_BRIGHTNESS_POSE -> Brightness Up/Down even when hand scale changes +20% (zero Zoom conflict!)
    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 8, make_base_hand(320.0, 260.0, scale=1.0), frames=3, dt=0.02)
    clock[0] += 0.5
    res_br_up = ctrl.process_frame(pose_id=8, landmarks_px=make_base_hand(320.0, 210.0, scale=1.2))
    assert res_br_up["action"] == "BRIGHTNESS_INCREASE"

    ctrl.reset_states()
    clock[0] += 0.5
    feed_pose(ctrl, clock, 8, make_base_hand(320.0, 210.0, scale=1.0), frames=3, dt=0.02)
    clock[0] += 0.5
    res_br_dn = ctrl.process_frame(pose_id=8, landmarks_px=make_base_hand(320.0, 260.0, scale=0.85))
    assert res_br_dn["action"] == "BRIGHTNESS_DECREASE"

    # 12. Class 9: FIST_IDLE_POSE -> CLUTCH_IDLE (Safe rest, no accidental OS actions)
    ctrl.reset_states()
    clock[0] += 0.5
    res_clutch = feed_pose(ctrl, clock, 9, make_base_hand(320.0, 240.0), frames=3, dt=0.02)
    assert res_clutch["action"] == "CLUTCH_IDLE"
