"""
Step 6a: System Control, Exponential Smoothing, Temporal Stabilization, Pinch-State Machine,
and Motion Management for the 10-Class Natural Ergonomic Gesture System (Point & Pinch + Windows Touch).
"""

import math
import time
from typing import Callable, Optional, Sequence
import numpy as np

from src.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    SMOOTHING_ALPHA,
    ACTION_DELAY,
    VERTICAL_MOTION_THRESHOLD,
    HORIZONTAL_MOTION_THRESHOLD,
    PINCH_SPREAD_DELTA_THRESHOLD,
    CONFIRM_FRAMES,
    DRAG_HOLD_DURATION,
    DOUBLE_PINCH_WINDOW,
    STATIC_HOLD_DURATION,
    PINCH_APPROACH_RATIO,
    POSE_CLASSES,
)
from src.features import euclidean_distance


class CursorSmoother:
    """
    Paper Section III-D Eq. (4): Exponential Moving Average (EMA) smoothing for cursor coordinates.
    x_smooth_t = x_smooth_{t-1} + (1 / alpha) * (x_raw_t - x_smooth_{t-1})
    Supports temporary higher alpha (Pinch-Approach Cursor Freeze) before clicking.
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        if alpha <= 0:
            raise ValueError("Smoothing factor alpha must be > 0")
        self.alpha = float(alpha)
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None

    def reset(self):
        self.smooth_x = None
        self.smooth_y = None

    def update(self, raw_x: float, raw_y: float, alpha_override: Optional[float] = None) -> tuple[float, float]:
        eff_alpha = float(alpha_override) if alpha_override is not None and alpha_override > 0 else self.alpha
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = float(raw_x)
            self.smooth_y = float(raw_y)
        else:
            self.smooth_x = self.smooth_x + (float(raw_x) - self.smooth_x) / eff_alpha
            self.smooth_y = self.smooth_y + (float(raw_y) - self.smooth_y) / eff_alpha
        return self.smooth_x, self.smooth_y


class ActionDebouncer:
    """
    Paper Section III-E: Time-based debouncing enforcing a minimum interval (0.4s = 400ms)
    between discrete action triggers to prevent accidental double-activations.
    """

    def __init__(self, delay: float = ACTION_DELAY, time_fn: Callable[[], float] = time.time):
        self.delay = float(delay)
        self.time_fn = time_fn
        self.last_action_time: float = -1e9

    def allow_action(self) -> bool:
        now = float(self.time_fn())
        if now - self.last_action_time >= self.delay:
            self.last_action_time = now
            return True
        return False


class VirtualMouseController:
    """
    Maps the 10 stabilized Natural Ergonomic Hand Poses + Temporal States into System Actions:
    - 0: POINT_POSE              -> CURSOR_MOVE (with Pinch-Approach Cursor Freeze)
    - 1: PINCH_INDEX_POSE        -> Quick Tap=LEFT_CLICK, Double Tap=DOUBLE_CLICK, Hold>=0.35s=DRAG_START/MOVE/RELEASE
    - 2: PINCH_MIDDLE_POSE       -> RIGHT_CLICK (latched once per pinch)
    - 3: TWO_FINGER_SCROLL_POSE  -> SCROLL_VERTICAL & SCROLL_HORIZONTAL
    - 4: ZOOM_L_POSE             -> ZOOM_IN & ZOOM_OUT (L-shape Thumb-Index spread/pinch)
    - 5: THREE_FINGER_POSE       -> WIN_TASK_VIEW, WIN_SHOW_DESKTOP, WIN_SWITCH_WINDOW_NEXT/PREV, WIN_SEARCH (0.85s hold)
    - 6: FOUR_FINGER_POSE        -> WIN_VIRTUAL_DESKTOP_LEFT/RIGHT, WIN_SNAP_MAXIMIZE/MINIMIZE, WIN_ACTION_CENTER (0.85s hold)
    - 7: OPEN_PALM_POSE          -> VOLUME_INCREASE, VOLUME_DECREASE, WIN_WIDGETS_PANEL
    - 8: SHAKA_BRIGHTNESS_POSE   -> BRIGHTNESS_INCREASE, BRIGHTNESS_DECREASE (no Zoom conflict)
    - 9: FIST_IDLE_POSE          -> CLUTCH_IDLE (Safe Rest / Pause cursor)
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
        dry_run: bool = False,
        confirm_frames: int = CONFIRM_FRAMES,
        time_fn: Callable[[], float] = time.time,
    ):
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.dry_run = bool(dry_run)
        self.confirm_frames = max(int(confirm_frames), 1)
        self.time_fn = time_fn

        self.smoother = CursorSmoother(alpha=SMOOTHING_ALPHA)
        self.debouncer = ActionDebouncer(delay=ACTION_DELAY, time_fn=time_fn)

        # Temporal Pose Stabilizer (eliminates 1-frame ghost clicks on pose transitions)
        self._candidate_pose: Optional[int] = None
        self._candidate_streak: int = 0
        self._confirmed_pose: Optional[int] = None

        # Pinch Click / Double-Click / Drag state machine
        self.pinch_active: bool = False
        self.pinch_start_time: float = 0.0
        self.last_pinch_tap_time: float = -1e9
        self.is_dragging: bool = False
        self.right_click_latched: bool = False

        # Motion tracking state variables
        self.scroll_start: Optional[tuple[float, float]] = None
        self.motion_start_pos: Optional[tuple[float, float]] = None
        self.zoom_start_span: Optional[float] = None
        self.motion_pose_id: Optional[int] = None
        self.hold_start_time: float = 0.0
        self.hold_triggered: bool = False
        self.current_brightness: int = 50

        self._pyautogui = None
        self._sbc = None
        if not self.dry_run:
            self._init_os_backends()

    def _init_os_backends(self):
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0
            self._pyautogui = pyautogui
        except Exception:
            self._pyautogui = None

        try:
            import screen_brightness_control as sbc
            self._sbc = sbc
        except Exception:
            self._sbc = None

    def _reset_motion_anchors(self):
        self.scroll_start = None
        self.motion_start_pos = None
        self.zoom_start_span = None
        self.hold_start_time = 0.0
        self.hold_triggered = False
        self.right_click_latched = False

    def reset_states(self):
        """Reset all stabilizer, pinch, and multi-frame motion tracking states."""
        if self.is_dragging:
            self._os_mouse_up()
            self.is_dragging = False
        self.pinch_active = False
        self.pinch_start_time = 0.0
        self._candidate_pose = None
        self._candidate_streak = 0
        self._confirmed_pose = None
        self.motion_pose_id = None
        self._reset_motion_anchors()

    def map_to_screen(
        self,
        cam_x: float,
        cam_y: float,
        alpha_override: Optional[float] = None,
    ) -> tuple[int, int]:
        """Paper Eq. (5): Coordinate interpolation mapping camera space to screen space."""
        sx = float(np.interp(cam_x, [0, self.camera_width], [0, self.screen_width]))
        sy = float(np.interp(cam_y, [0, self.camera_height], [0, self.screen_height]))
        smooth_x, smooth_y = self.smoother.update(sx, sy, alpha_override=alpha_override)
        return int(np.clip(smooth_x, 0, self.screen_width - 1)), int(np.clip(smooth_y, 0, self.screen_height - 1))

    def process_frame(
        self,
        pose_id: Optional[int],
        landmarks_px: Optional[Sequence[Sequence[float]]],
    ) -> dict:
        """
        Execute stabilized Stage 5 (Action Mapping & System Control) for the current frame.
        """
        now = float(self.time_fn())
        result = {
            "pose_id": pose_id,
            "pose_name": POSE_CLASSES.get(pose_id, "NO_HAND") if pose_id is not None else "NO_HAND",
            "action": "NONE",
            "cursor_pos": None,
            "drag_released": False,
        }

        if pose_id is None or landmarks_px is None or len(landmarks_px) != 21:
            if self.is_dragging:
                result["drag_released"] = True
            self.reset_states()
            return result

        # Temporal Majority / Streak Pose Stabilizer
        if pose_id == self._candidate_pose:
            self._candidate_streak += 1
        else:
            self._candidate_pose = pose_id
            self._candidate_streak = 1

        if self._candidate_streak >= self.confirm_frames:
            self._confirmed_pose = pose_id

        if self._confirmed_pose is None:
            result["pose_name"] = "STABILIZING"
            return result

        active_pose = self._confirmed_pose
        result["pose_id"] = active_pose
        result["pose_name"] = POSE_CLASSES.get(active_pose, "NO_HAND")

        # Handle exit from PINCH_INDEX_POSE (Class 1): Drag Release vs Tap Left/Double Click
        if self.pinch_active and active_pose != 1:
            self.pinch_active = False
            self._reset_motion_anchors()
            self.motion_pose_id = active_pose
            if self.is_dragging:
                self._os_mouse_up()
                self.is_dragging = False
                result["drag_released"] = True
                result["action"] = "DRAG_RELEASE"
            else:
                if now - self.last_pinch_tap_time <= DOUBLE_PINCH_WINDOW:
                    self._os_double_click()
                    self.last_pinch_tap_time = -1e9
                    result["action"] = "DOUBLE_CLICK"
                else:
                    self._os_click("left")
                    self.last_pinch_tap_time = now
                    result["action"] = "LEFT_CLICK"
            return result

        # Reset motion anchors if stabilized pose changed
        if self.motion_pose_id != active_pose:
            self._reset_motion_anchors()
            self.motion_pose_id = active_pose

        index_tip = (float(landmarks_px[8][0]), float(landmarks_px[8][1]))
        palm_center = (float(landmarks_px[9][0]), float(landmarks_px[9][1]))

        # ---------------------------------------------------------------------
        # CLASS 0: POINT_POSE -> Cursor Move (+ Pinch-Approach Cursor Freeze)
        # ---------------------------------------------------------------------
        if active_pose == 0:
            palm_ref = max(euclidean_distance(landmarks_px[0], landmarks_px[9]), 1e-6)
            pinch_ratio = euclidean_distance(landmarks_px[4], landmarks_px[8]) / palm_ref
            alpha_ov = self.smoother.alpha * 4.0 if pinch_ratio < PINCH_APPROACH_RATIO else None
            scr_x, scr_y = self.map_to_screen(index_tip[0], index_tip[1], alpha_override=alpha_ov)
            self._os_move_to(scr_x, scr_y)
            result["action"] = "CURSOR_MOVE"
            result["cursor_pos"] = (scr_x, scr_y)
            return result

        # ---------------------------------------------------------------------
        # CLASS 1: PINCH_INDEX_POSE -> Hold <0.35s: Tap/DoubleClick on release; >=0.35s: Drag & Drop
        # ---------------------------------------------------------------------
        if active_pose == 1:
            if not self.pinch_active:
                self.pinch_active = True
                self.pinch_start_time = now
                result["action"] = "PINCH_HOLDING"
                return result

            if not self.is_dragging:
                if (now - self.pinch_start_time) >= DRAG_HOLD_DURATION:
                    self._os_mouse_down()
                    self.is_dragging = True
                    scr_x, scr_y = self.map_to_screen(index_tip[0], index_tip[1])
                    result["cursor_pos"] = (scr_x, scr_y)
                    result["action"] = "DRAG_START"
                else:
                    result["action"] = "PINCH_HOLDING"
            else:
                scr_x, scr_y = self.map_to_screen(index_tip[0], index_tip[1])
                self._os_move_to(scr_x, scr_y)
                result["cursor_pos"] = (scr_x, scr_y)
                result["action"] = "DRAG_MOVE"
            return result

        # ---------------------------------------------------------------------
        # CLASS 2: PINCH_MIDDLE_POSE -> Right Click (latched once per pinch)
        # ---------------------------------------------------------------------
        if active_pose == 2:
            if not self.right_click_latched and self.debouncer.allow_action():
                self._os_click("right")
                self.right_click_latched = True
                result["action"] = "RIGHT_CLICK"
            return result

        # ---------------------------------------------------------------------
        # CLASS 3: TWO_FINGER_SCROLL_POSE -> Precision Touchpad 2-Finger Vertical/Horizontal Scroll
        # ---------------------------------------------------------------------
        if active_pose == 3:
            two_mid = (
                0.5 * (float(landmarks_px[8][0]) + float(landmarks_px[12][0])),
                0.5 * (float(landmarks_px[8][1]) + float(landmarks_px[12][1])),
            )
            if self.scroll_start is None:
                self.scroll_start = two_mid
                result["action"] = "SCROLL_READY"
                return result

            dx = two_mid[0] - self.scroll_start[0]
            dy = two_mid[1] - self.scroll_start[1]

            if abs(dy) >= abs(dx) and abs(dy) > 15.0:
                scroll_amount = int(-dy * 2.5)
                self._os_scroll_vertical(scroll_amount)
                self.scroll_start = two_mid
                result["action"] = "SCROLL_VERTICAL"
            elif abs(dx) > abs(dy) and abs(dx) > 18.0:
                hscroll_amount = int(dx * 2.5)
                self._os_scroll_horizontal(hscroll_amount)
                self.scroll_start = two_mid
                result["action"] = "SCROLL_HORIZONTAL"
            return result

        # ---------------------------------------------------------------------
        # CLASS 4: ZOOM_L_POSE -> Thumb-Index L-Frame Spread/Pinch Zoom
        # ---------------------------------------------------------------------
        if active_pose == 4:
            d_4_8 = euclidean_distance(landmarks_px[4], landmarks_px[8])
            if self.zoom_start_span is None:
                self.zoom_start_span = d_4_8
                result["action"] = "ZOOM_READY"
                return result

            delta_d = d_4_8 - self.zoom_start_span
            if abs(delta_d) >= PINCH_SPREAD_DELTA_THRESHOLD and self.debouncer.allow_action():
                if delta_d > 0:
                    self._os_hotkey("ctrl", "+")
                    result["action"] = "ZOOM_IN"
                else:
                    self._os_hotkey("ctrl", "-")
                    result["action"] = "ZOOM_OUT"
                self.zoom_start_span = d_4_8
            return result

        # ---------------------------------------------------------------------
        # CLASS 5: THREE_FINGER_POSE -> Windows Touch 3-Finger Swipes + 0.85s Search
        # ---------------------------------------------------------------------
        if active_pose == 5:
            if self.motion_start_pos is None:
                self.motion_start_pos = palm_center
                self.hold_start_time = now
                self.hold_triggered = False
                result["action"] = "THREE_FINGER_READY"
                return result

            dx = palm_center[0] - self.motion_start_pos[0]
            dy = palm_center[1] - self.motion_start_pos[1]

            if abs(dy) >= VERTICAL_MOTION_THRESHOLD and abs(dy) >= abs(dx):
                if self.debouncer.allow_action():
                    if dy < 0:
                        self._os_hotkey("win", "tab")
                        result["action"] = "WIN_TASK_VIEW"
                    else:
                        self._os_hotkey("win", "d")
                        result["action"] = "WIN_SHOW_DESKTOP"
                    self.motion_start_pos = palm_center
                    self.hold_start_time = now
                return result

            if abs(dx) >= HORIZONTAL_MOTION_THRESHOLD and abs(dx) > abs(dy):
                if self.debouncer.allow_action():
                    if dx > 0:
                        self._os_hotkey("alt", "tab")
                        result["action"] = "WIN_SWITCH_WINDOW_NEXT"
                    else:
                        self._os_hotkey("alt", "shift", "tab")
                        result["action"] = "WIN_SWITCH_WINDOW_PREV"
                    self.motion_start_pos = palm_center
                    self.hold_start_time = now
                return result

            if math.hypot(dx, dy) >= 15.0:
                self.hold_start_time = now
            elif not self.hold_triggered and (now - self.hold_start_time) >= STATIC_HOLD_DURATION:
                if self.debouncer.allow_action():
                    self._os_hotkey("win", "s")
                    result["action"] = "WIN_SEARCH"
                    self.hold_triggered = True
            return result

        # ---------------------------------------------------------------------
        # CLASS 6: FOUR_FINGER_POSE -> Windows Touch 4-Finger Swipes + 0.85s Action Center
        # ---------------------------------------------------------------------
        if active_pose == 6:
            if self.motion_start_pos is None:
                self.motion_start_pos = palm_center
                self.hold_start_time = now
                self.hold_triggered = False
                result["action"] = "FOUR_FINGER_READY"
                return result

            dx = palm_center[0] - self.motion_start_pos[0]
            dy = palm_center[1] - self.motion_start_pos[1]

            if abs(dx) >= HORIZONTAL_MOTION_THRESHOLD and abs(dx) >= abs(dy):
                if self.debouncer.allow_action():
                    if dx > 0:
                        self._os_hotkey("win", "ctrl", "right")
                        result["action"] = "WIN_VIRTUAL_DESKTOP_RIGHT"
                    else:
                        self._os_hotkey("win", "ctrl", "left")
                        result["action"] = "WIN_VIRTUAL_DESKTOP_LEFT"
                    self.motion_start_pos = palm_center
                    self.hold_start_time = now
                return result

            if abs(dy) >= VERTICAL_MOTION_THRESHOLD and abs(dy) > abs(dx):
                if self.debouncer.allow_action():
                    if dy < 0:
                        self._os_hotkey("win", "up")
                        result["action"] = "WIN_SNAP_MAXIMIZE"
                    else:
                        self._os_hotkey("win", "down")
                        result["action"] = "WIN_SNAP_MINIMIZE"
                    self.motion_start_pos = palm_center
                    self.hold_start_time = now
                return result

            if math.hypot(dx, dy) >= 15.0:
                self.hold_start_time = now
            elif not self.hold_triggered and (now - self.hold_start_time) >= STATIC_HOLD_DURATION:
                if self.debouncer.allow_action():
                    self._os_hotkey("win", "a")
                    result["action"] = "WIN_ACTION_CENTER"
                    self.hold_triggered = True
            return result

        # ---------------------------------------------------------------------
        # CLASS 7: OPEN_PALM_POSE -> Volume Up/Down + Win Widgets Swipe Right
        # ---------------------------------------------------------------------
        if active_pose == 7:
            if self.motion_start_pos is None:
                self.motion_start_pos = palm_center
                result["action"] = "OPEN_PALM_READY"
                return result

            dx = palm_center[0] - self.motion_start_pos[0]
            dy = palm_center[1] - self.motion_start_pos[1]

            if abs(dy) >= VERTICAL_MOTION_THRESHOLD and abs(dy) >= abs(dx):
                if self.debouncer.allow_action():
                    if dy < 0:
                        self._os_press_key("volumeup")
                        result["action"] = "VOLUME_INCREASE"
                    else:
                        self._os_press_key("volumedown")
                        result["action"] = "VOLUME_DECREASE"
                    self.motion_start_pos = palm_center
                return result

            if dx >= HORIZONTAL_MOTION_THRESHOLD and abs(dx) > abs(dy):
                if self.debouncer.allow_action():
                    self._os_hotkey("win", "w")
                    result["action"] = "WIN_WIDGETS_PANEL"
                    self.motion_start_pos = palm_center
                return result

            return result

        # ---------------------------------------------------------------------
        # CLASS 8: SHAKA_BRIGHTNESS_POSE -> Dedicated Brightness Up/Down (No Zoom conflict)
        # ---------------------------------------------------------------------
        if active_pose == 8:
            if self.motion_start_pos is None:
                self.motion_start_pos = palm_center
                result["action"] = "BRIGHTNESS_READY"
                return result

            dy = palm_center[1] - self.motion_start_pos[1]
            if abs(dy) >= VERTICAL_MOTION_THRESHOLD:
                if self.debouncer.allow_action():
                    if dy < 0:
                        self._os_adjust_brightness(+10)
                        result["action"] = "BRIGHTNESS_INCREASE"
                    else:
                        self._os_adjust_brightness(-10)
                        result["action"] = "BRIGHTNESS_DECREASE"
                    self.motion_start_pos = palm_center
            return result

        # ---------------------------------------------------------------------
        # CLASS 9: FIST_IDLE_POSE -> Safe Rest / Clutch (Zero OS Actions)
        # ---------------------------------------------------------------------
        if active_pose == 9:
            result["action"] = "CLUTCH_IDLE"
            return result

        return result

    # =========================================================================
    # Low-level OS Execution Wrappers (Bypassed safely when dry_run=True)
    # =========================================================================
    def _os_move_to(self, x: int, y: int):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.moveTo(x, y)

    def _os_click(self, button: str = "left"):
        if not self.dry_run and self._pyautogui is not None:
            if button == "right":
                self._pyautogui.rightClick()
            else:
                self._pyautogui.click()

    def _os_double_click(self):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.doubleClick()

    def _os_mouse_down(self):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.mouseDown()

    def _os_mouse_up(self):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.mouseUp()

    def _os_scroll_vertical(self, amount: int):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.scroll(amount)

    def _os_scroll_horizontal(self, amount: int):
        if not self.dry_run and self._pyautogui is not None:
            try:
                self._pyautogui.hscroll(amount)
            except Exception:
                pass

    def _os_press_key(self, key: str):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.press(key)

    def _os_hotkey(self, *keys: str):
        if not self.dry_run and self._pyautogui is not None:
            self._pyautogui.hotkey(*keys)

    def _os_adjust_brightness(self, delta: int):
        self.current_brightness = int(np.clip(self.current_brightness + delta, 10, 100))
        if not self.dry_run and self._sbc is not None:
            try:
                self._sbc.set_brightness(self.current_brightness)
            except Exception:
                pass
