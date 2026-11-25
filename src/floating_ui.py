"""
Always-On-Top Floating Windows Debug UI for Touchless Virtual Mouse & Windows Touchscreen Gestures.
Displays:
- Live camera feed with MediaPipe 21-landmark skeleton + distance measurement overlays
- Prominent real-time Command/Action Banner emitted by the hand
- Predicted Static Pose & ML Confidence bar
- Live 5-finger binary state [Thumb, Index, Middle, Ring, Pinky] & Euclidean distances (d_pinch, d_move)
- One-click switching across all 5 independent ML modules (KNN, DT, NB, LR, SVM)
- Timestamped Command Debug Log history
"""

from collections import deque
from datetime import datetime
import atexit
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional, Sequence
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    PINCH_THRESHOLD,
    AVAILABLE_MODELS,
)
from src.features import euclidean_distance, fingers_up, compute_hand_scale
from src.hand_tracker import MediaPipeHandTracker, ThreadedCameraCapture


ACTION_DESCRIPTIONS_VI: dict[str, str] = {
    "NONE": "Đang chờ / Ổn định khung hình",
    "CURSOR_MOVE": "Di chuyển chuột (1 Ngón Trỏ)",
    "PINCH_HOLDING": "Đang chụm Cái-Trỏ (Thả=Click | Giữ>=0.35s=Kéo thả)",
    "LEFT_CLICK": "Click Chuột Trái (Thả Chụm Cái-Trỏ)",
    "DOUBLE_CLICK": "Click Đúp Chuột (Chụm Cái-Trỏ 2 Lần)",
    "RIGHT_CLICK": "Click Chuột Phải (Chụm Cái-Giữa)",
    "DRAG_START": "Bắt đầu Kéo Thả / Giữ Chuột (Drag Start)",
    "DRAG_MOVE": "Đang Kéo Thả Chuột (Dragging...)",
    "DRAG_RELEASE": "Đã Thả Chuột (Drag Release)",
    "SCROLL_READY": "2 Ngón Trỏ-Giữa (Sẵn sàng Cuộn Dọc/Ngang)",
    "SCROLL_VERTICAL": "Cuộn Dọc Trang (Vertical Scroll)",
    "SCROLL_HORIZONTAL": "Cuộn Ngang Trang (Horizontal Scroll)",
    "ZOOM_READY": "Chữ L (Cái-Trỏ mở: Khép/Mở để Zoom)",
    "ZOOM_IN": "Phóng To Màn Hình (Zoom In: Ctrl + '+')",
    "ZOOM_OUT": "Thu Nhỏ Màn Hình (Zoom Out: Ctrl + '-')",
    "THREE_FINGER_READY": "3 Ngón (Vuốt Lên/Xuống/Ngang | Giữ 0.85s=Search)",
    "WIN_TASK_VIEW": "Windows 3 Ngón Lên: Mở Task View (Win+Tab)",
    "WIN_SHOW_DESKTOP": "Windows 3 Ngón Xuống: Hiện Desktop (Win+D)",
    "WIN_SWITCH_WINDOW_NEXT": "Windows 3 Ngón Phải: Chuyển Cửa Sổ (Alt+Tab)",
    "WIN_SWITCH_WINDOW_PREV": "Windows 3 Ngón Trái: Lùi Cửa Sổ (Alt+Shift+Tab)",
    "WIN_SEARCH": "Windows 3 Ngón Giữ 0.85s: Mở Tìm Kiếm (Win+S)",
    "FOUR_FINGER_READY": "4 Ngón (Vuốt Ngang/Lên/Xuống | Giữ 0.85s=Action Center)",
    "WIN_VIRTUAL_DESKTOP_RIGHT": "Windows 4 Ngón Phải: Màn Hình Ảo Kế (Win+Ctrl+Right)",
    "WIN_VIRTUAL_DESKTOP_LEFT": "Windows 4 Ngón Trái: Màn Hình Ảo Trước (Win+Ctrl+Left)",
    "WIN_SNAP_MAXIMIZE": "Windows 4 Ngón Lên: Phóng To Cửa Sổ (Win+Up)",
    "WIN_SNAP_MINIMIZE": "Windows 4 Ngón Xuống: Thu Nhỏ Cửa Sổ (Win+Down)",
    "WIN_ACTION_CENTER": "Windows 4 Ngón Giữ 0.85s: Action Center (Win+A)",
    "OPEN_PALM_READY": "Bàn Tay Mở (Lên/Xuống: Âm lượng | Phải: Widgets)",
    "VOLUME_INCREASE": "Tăng Âm Lượng Hệ Thống (Volume Up)",
    "VOLUME_DECREASE": "Giảm Âm Lượng Hệ Thống (Volume Down)",
    "WIN_WIDGETS_PANEL": "Windows Vuốt Trái->Phải: Mở Widgets (Win+W)",
    "BRIGHTNESS_READY": "Shaka Cái+Út (Đưa Lên/Xuống chỉnh Độ Sáng)",
    "BRIGHTNESS_INCREASE": "Tăng Độ Sáng Màn Hình (Brightness Up)",
    "BRIGHTNESS_DECREASE": "Giảm Độ Sáng Màn Hình (Brightness Down)",
    "CLUTCH_IDLE": "Nắm Đấm: Tạm Dừng / Nghỉ Tay (An Toàn)",
}


def build_debug_telemetry(
    landmarks_px: Optional[Sequence[Sequence[float]]],
    info: dict,
    fps: float,
) -> dict:
    """Compute detailed geometric and state telemetry for the floating debug UI."""
    if landmarks_px is None or len(landmarks_px) != 21:
        return {
            "hand_detected": False,
            "fingers_up": [0, 0, 0, 0, 0],
            "fingers_str": "[Không thấy bàn tay trong khung hình]",
            "pinch_dist_px": 0.0,
            "thumb_middle_dist_px": 0.0,
            "index_middle_dist_px": 0.0,
            "hand_scale_px": 0.0,
            "wrist_xy": (0, 0),
            "index_tip_xy": (0, 0),
            "pose_id": None,
            "pose_name": "NO_HAND",
            "confidence": 0.0,
            "action": "NONE",
            "action_vi": "Không phát hiện bàn tay",
            "model_key": info.get("model_key", ""),
            "model_name": info.get("model_name", ""),
            "fps": fps,
        }

    f_up = fingers_up(landmarks_px)
    f_str = f"[Cái:{f_up[0]} | Trỏ:{f_up[1]} | Giữa:{f_up[2]} | Áp:{f_up[3]} | Út:{f_up[4]}]"
    d_pinch = euclidean_distance(landmarks_px[4], landmarks_px[8])
    d_thumb_mid = euclidean_distance(landmarks_px[4], landmarks_px[12])
    d_idx_mid = euclidean_distance(landmarks_px[8], landmarks_px[12])
    h_scale = compute_hand_scale(landmarks_px)
    action_code = info.get("action", "NONE")

    return {
        "hand_detected": True,
        "fingers_up": f_up,
        "fingers_str": f_str,
        "pinch_dist_px": round(d_pinch, 1),
        "thumb_middle_dist_px": round(d_thumb_mid, 1),
        "index_middle_dist_px": round(d_idx_mid, 1),
        "hand_scale_px": round(h_scale, 1),
        "wrist_xy": (int(landmarks_px[0][0]), int(landmarks_px[0][1])),
        "index_tip_xy": (int(landmarks_px[8][0]), int(landmarks_px[8][1])),
        "pose_id": info.get("pose_id"),
        "pose_name": info.get("pose_name", "NO_HAND"),
        "confidence": float(info.get("confidence", 0.0)),
        "action": action_code,
        "action_vi": ACTION_DESCRIPTIONS_VI.get(action_code, action_code),
        "model_key": info.get("model_key", ""),
        "model_name": info.get("model_name", ""),
        "cursor_pos": info.get("cursor_pos"),
        "fps": fps,
    }


def draw_debug_annotations_on_frame(
    frame_bgr: np.ndarray,
    landmarks_px: Optional[Sequence[Sequence[float]]],
    telemetry: dict,
) -> np.ndarray:
    """Draw distance lines (Thumb-Index Pinch & Thumb-Middle Right-Click) directly on the camera frame."""
    if landmarks_px is None or len(landmarks_px) != 21:
        return frame_bgr

    p_thumb = (int(landmarks_px[4][0]), int(landmarks_px[4][1]))
    p_index = (int(landmarks_px[8][0]), int(landmarks_px[8][1]))
    p_middle = (int(landmarks_px[12][0]), int(landmarks_px[12][1]))

    # 1. Draw Thumb-Index pinch line (Cyan if < 30px threshold, else gray)
    d_pinch = telemetry["pinch_dist_px"]
    pinch_color = (0, 255, 255) if d_pinch < PINCH_THRESHOLD else (160, 160, 160)
    cv2.line(frame_bgr, p_thumb, p_index, pinch_color, 2, cv2.LINE_AA)
    mid_pinch = ((p_thumb[0] + p_index[0]) // 2, (p_thumb[1] + p_index[1]) // 2)
    cv2.putText(
        frame_bgr,
        f"pIdx:{d_pinch:.0f}px",
        (mid_pinch[0] - 24, mid_pinch[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        pinch_color,
        1,
        cv2.LINE_AA,
    )

    # 2. Draw Thumb-Middle Right-Click pinch line (Magenta if < 30px threshold, else muted purple)
    d_tm = telemetry["thumb_middle_dist_px"]
    tm_color = (255, 80, 255) if d_tm < PINCH_THRESHOLD else (140, 120, 160)
    cv2.line(frame_bgr, p_thumb, p_middle, tm_color, 2, cv2.LINE_AA)
    mid_tm = ((p_thumb[0] + p_middle[0]) // 2, (p_thumb[1] + p_middle[1]) // 2)
    cv2.putText(
        frame_bgr,
        f"pMid:{d_tm:.0f}px",
        (mid_tm[0] - 24, mid_tm[1] + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        tm_color,
        1,
        cv2.LINE_AA,
    )

    return frame_bgr


class FloatingDebugOverlayUI:
    """
    Always-on-top Windows Debug UI showing live webcam feed, hand skeleton,
    emitted commands, finger states, and instant ML model switcher.
    """

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("📌 Touchless Virtual Mouse - Pinned Debug Overlay")
        self.root.configure(bg="#111827")

        # Pin window always on top of all Windows applications
        self.is_topmost = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.96)

        # Position at top-right of screen by default so it stays out of the way while debugging
        scr_w = self.root.winfo_screenwidth()
        win_w, win_h = 500, 740
        pos_x = max(scr_w - win_w - 24, 10)
        pos_y = 24
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        self.command_history: deque[str] = deque(maxlen=8)
        self.last_logged_action: str = "NONE"
        self.last_meaningful_action: str = "NONE"
        self.last_meaningful_time: str = "--:--:--"
        self.failed_reads_count: int = 0

        self._photo_image = None
        self._build_widgets()

        self.tracker = MediaPipeHandTracker(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.cap = None
        self._open_camera()
        atexit.register(self._release_resources)

        self.prev_time = time.perf_counter()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Bind keyboard shortcuts 1-5, d, q
        self.root.bind("1", lambda _e: self._on_select_model("knn"))
        self.root.bind("2", lambda _e: self._on_select_model("dt"))
        self.root.bind("3", lambda _e: self._on_select_model("nb"))
        self.root.bind("4", lambda _e: self._on_select_model("lr"))
        self.root.bind("5", lambda _e: self._on_select_model("svm"))
        self.root.bind("d", lambda _e: self._toggle_dry_run())
        self.root.bind("q", lambda _e: self.on_close())

    def _build_widgets(self):
        # 1. Top Bar: Pin toggle, Dry-Run toggle, Opacity slider
        top_bar = tk.Frame(self.root, bg="#1f2937", padx=8, pady=6)
        top_bar.pack(fill=tk.X)

        self.pin_btn = tk.Checkbutton(
            top_bar,
            text="📌 Ghim nổi (Topmost)",
            variable=self.is_topmost,
            command=self._toggle_topmost,
            bg="#1f2937",
            fg="#60a5fa",
            selectcolor="#111827",
            activebackground="#1f2937",
            activeforeground="#93c5fd",
            font=("Segoe UI", 9, "bold"),
        )
        self.pin_btn.pack(side=tk.LEFT)

        self.mode_btn = tk.Button(
            top_bar,
            text="",
            command=self._toggle_dry_run,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
            relief=tk.FLAT,
        )
        self.mode_btn.pack(side=tk.RIGHT)
        self._refresh_mode_btn_style()

        # 2. Model Switcher Bar (5 Independent ML Modules)
        model_bar = tk.Frame(self.root, bg="#111827", padx=8, pady=5)
        model_bar.pack(fill=tk.X)
        tk.Label(
            model_bar,
            text="Mô hình ML:",
            bg="#111827",
            fg="#9ca3af",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.model_buttons: dict[str, tk.Button] = {}
        for idx, key in enumerate(["knn", "dt", "nb", "lr", "svm"], start=1):
            btn = tk.Button(
                model_bar,
                text=f"[{idx}] {key.upper()}",
                command=lambda k=key: self._on_select_model(k),
                font=("Segoe UI", 8, "bold"),
                padx=7,
                pady=2,
                relief=tk.FLAT,
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.model_buttons[key] = btn
        self._refresh_model_buttons()

        # 3. Camera Viewport (Shows exactly what camera sees + skeleton + distance lines)
        cam_container = tk.Frame(self.root, bg="#000000", bd=2, relief=tk.GROOVE)
        cam_container.pack(padx=8, pady=4)
        init_canvas = np.full((320, 480, 3), (24, 30, 42), dtype=np.uint8)
        cv2.putText(
            init_canvas,
            "Dang khoi dong Camera...",
            (115, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (120, 220, 255),
            2,
            cv2.LINE_AA,
        )
        self._photo_image = ImageTk.PhotoImage(image=Image.fromarray(init_canvas))
        self.cam_label = tk.Label(cam_container, image=self._photo_image, bg="#000000", width=480, height=320)
        self.cam_label.pack()

        # 4. Prominent Emitted Command Box (What command the hand is outputting right now)
        cmd_frame = tk.Frame(self.root, bg="#1e293b", padx=10, pady=8, bd=1, relief=tk.SOLID)
        cmd_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(
            cmd_frame,
            text="⚡ LỆNH ĐANG XUẤT RA TỪ TAY (CURRENT OUTPUT COMMAND):",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        self.lbl_current_action = tk.Label(
            cmd_frame,
            text="NONE",
            bg="#0f172a",
            fg="#38bdf8",
            font=("Consolas", 15, "bold"),
            pady=6,
        )
        self.lbl_current_action.pack(fill=tk.X, pady=(3, 2))

        self.lbl_action_desc = tk.Label(
            cmd_frame,
            text="Đang chờ cử chỉ...",
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "italic"),
            anchor="w",
        )
        self.lbl_action_desc.pack(fill=tk.X)

        self.lbl_last_triggered = tk.Label(
            cmd_frame,
            text="Lệnh kích hoạt gần nhất: Chưa có",
            bg="#1e293b",
            fg="#fbbf24",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self.lbl_last_triggered.pack(fill=tk.X, pady=(2, 0))

        # 5. Telemetry & Finger States Panel
        tele_frame = tk.Frame(self.root, bg="#111827", padx=10, pady=4)
        tele_frame.pack(fill=tk.X, padx=8)

        self.lbl_pose_conf = tk.Label(
            tele_frame,
            text="Tư thế (Pose): NO_HAND (0.0%) | FPS: 0.0",
            bg="#111827",
            fg="#4ade80",
            font=("Consolas", 10, "bold"),
            anchor="w",
        )
        self.lbl_pose_conf.pack(fill=tk.X)

        self.lbl_fingers = tk.Label(
            tele_frame,
            text="Ngón tay: [Cái:0 | Trỏ:0 | Giữa:0 | Áp:0 | Út:0]",
            bg="#111827",
            fg="#f8fafc",
            font=("Consolas", 9),
            anchor="w",
        )
        self.lbl_fingers.pack(fill=tk.X, pady=1)

        self.lbl_distances = tk.Label(
            tele_frame,
            text="Khoảng cách: d(Cái-Trỏ)=0.0px (Click/Drag) | d(Cái-Giữa)=0.0px (Right Click)",
            bg="#111827",
            fg="#cbd5e1",
            font=("Consolas", 8),
            anchor="w",
        )
        self.lbl_distances.pack(fill=tk.X, pady=1)

        # 6. Command History Log (Last 8 triggered actions for debugging)
        log_frame = tk.Frame(self.root, bg="#0f172a", padx=8, pady=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        tk.Label(
            log_frame,
            text="📜 NHẬT KÝ LỆNH VỪA PHÁT RA (DEBUG COMMAND LOG):",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        self.log_listbox = tk.Listbox(
            log_frame,
            bg="#090d16",
            fg="#a7f3d0",
            font=("Consolas", 8),
            height=6,
            bd=0,
            highlightthickness=0,
        )
        self.log_listbox.pack(fill=tk.BOTH, expand=True, pady=(3, 0))

    def _toggle_topmost(self):
        pinned = bool(self.is_topmost.get())
        self.root.attributes("-topmost", pinned)

    def _toggle_dry_run(self):
        self.app.controller.dry_run = not self.app.controller.dry_run
        self._refresh_mode_btn_style()

    def _refresh_mode_btn_style(self):
        if self.app.controller.dry_run:
            self.mode_btn.configure(
                text="🛡️ CHỈ DEBUG (Dry-Run: BẬT)",
                bg="#d97706",
                fg="#ffffff",
            )
        else:
            self.mode_btn.configure(
                text="🖱️ ĐIỀU KHIỂN THẬT (Live OS: BẬT)",
                bg="#16a34a",
                fg="#ffffff",
            )

    def _on_select_model(self, model_key: str):
        self.app.switch_model(model_key)
        self._refresh_model_buttons()

    def _refresh_model_buttons(self):
        for k, btn in self.model_buttons.items():
            if k == self.app.active_model_key:
                btn.configure(bg="#2563eb", fg="#ffffff")
            else:
                btn.configure(bg="#374151", fg="#d1d5db")

    def _append_command_log(self, action: str, pose_name: str, conf: float):
        if action in (
            "NONE",
            "SCROLL_READY",
            "ZOOM_READY",
            "THREE_FINGER_READY",
            "FOUR_FINGER_READY",
            "OPEN_PALM_READY",
            "BRIGHTNESS_READY",
            "PINCH_HOLDING",
        ):
            return
        # Avoid flooding identical continuous states every frame in the history list
        if action in ("CURSOR_MOVE", "DRAG_MOVE", "CLUTCH_IDLE") and self.last_logged_action == action:
            return

        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
        entry = f"[{now_str}] {action:<24s} <- {pose_name} ({conf*100:.0f}%)"
        self.command_history.appendleft(entry)
        self.last_logged_action = action
        self.last_meaningful_action = action
        self.last_meaningful_time = now_str

        self.log_listbox.delete(0, tk.END)
        for item in self.command_history:
            self.log_listbox.insert(tk.END, item)

    def _open_camera(self):
        if self.cap is not None:
            try:
                self.cap.stop()
            except Exception:
                pass
        self.cap = ThreadedCameraCapture(
            camera_index=self.app.camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        )
        self.failed_reads_count = 0

    def _update_loop(self):
        ok, frame = (False, None)
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()

        if ok and frame is not None:
            self.failed_reads_count = 0
            landmarks_px, frame = self.tracker.detect(frame, draw=True)

            if landmarks_px is not None:
                info = self.app.process_landmarks(landmarks_px)
            else:
                self.app.controller.process_frame(None, None)
                info = {
                    "pose_id": None,
                    "pose_name": "NO_HAND",
                    "action": "NONE",
                    "confidence": 0.0,
                    "model_key": self.app.active_model_key.upper(),
                    "model_name": self.app.active_model.model_display_name,
                }

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now
            self.smooth_fps = 0.85 * getattr(self, "smooth_fps", inst_fps) + 0.15 * inst_fps

            telemetry = build_debug_telemetry(landmarks_px, info, self.smooth_fps)
            draw_debug_annotations_on_frame(frame, landmarks_px, telemetry)

            # Fast resize camera preview to fit sleek floating window (480x320)
            preview_bgr = cv2.resize(frame, (480, 320), interpolation=cv2.INTER_NEAREST)
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(preview_rgb)
            self._photo_image = ImageTk.PhotoImage(image=img_pil)
            self.cam_label.configure(image=self._photo_image)

            # Update UI labels
            act = telemetry["action"]
            if act in ("NONE", "CLUTCH_IDLE"):
                act_color = "#94a3b8"
            elif act == "CURSOR_MOVE":
                act_color = "#38bdf8"
            else:
                act_color = "#fbbf24"
            self.lbl_current_action.configure(
                text=act,
                fg=act_color,
            )
            self.lbl_action_desc.configure(text=telemetry["action_vi"])

            self._append_command_log(act, telemetry["pose_name"], telemetry["confidence"])
            if self.last_meaningful_action != "NONE":
                self.lbl_last_triggered.configure(
                    text=f"Lệnh kích hoạt gần nhất: {self.last_meaningful_action} (lúc {self.last_meaningful_time})"
                )

            self.lbl_pose_conf.configure(
                text=f"Model [{telemetry['model_key']}] | Pose: {telemetry['pose_name']} ({telemetry['confidence']*100:.1f}%) | {self.smooth_fps:.1f} FPS"
            )
            self.lbl_fingers.configure(text=f"Ngón tay: {telemetry['fingers_str']}")
            self.lbl_distances.configure(
                text=(
                    f"d(Cái-Trỏ)={telemetry['pinch_dist_px']:4.1f}px (Click/Drag/Zoom) | "
                    f"d(Cái-Giữa)={telemetry['thumb_middle_dist_px']:4.1f}px (RightClick)"
                )
            )
        else:
            self.failed_reads_count += 1
            if self.failed_reads_count > 20:
                err_canvas = np.full((320, 480, 3), (45, 20, 25), dtype=np.uint8)
                cv2.putText(
                    err_canvas,
                    "Dang ket noi Camera...",
                    (125, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (100, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                err_rgb = cv2.cvtColor(err_canvas, cv2.COLOR_BGR2RGB)
                self._photo_image = ImageTk.PhotoImage(image=Image.fromarray(err_rgb))
                self.cam_label.configure(image=self._photo_image)
            if self.failed_reads_count % 90 == 0:
                self._open_camera()

        # Schedule next frame after 1ms instead of 15ms for maximum FPS
        self.root.after(1, self._update_loop)

    def run(self):
        self._update_loop()
        self.root.mainloop()

    def _release_resources(self):
        try:
            if self.tracker is not None:
                self.tracker.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.stop()
                self.cap = None
        except Exception:
            pass

    def on_close(self):
        self._release_resources()
        try:
            self.root.destroy()
        except Exception:
            pass
