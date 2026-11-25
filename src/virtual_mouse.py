"""
Step 6b: Real-Time Touchless Virtual Mouse & Windows Touchscreen Interface
Supports independent execution and live hot-swapping across all 5 Machine Learning modules:
  Key '1' -> KNN
  Key '2' -> Decision Tree
  Key '3' -> Gaussian Naive Bayes
  Key '4' -> Logistic Regression
  Key '5' -> Support Vector Machine (SVM)
  Key 'd' -> Toggle Dry-Run Mode (Safe preview vs Active OS Mouse/Keyboard Control)
  Key 'q' / ESC -> Exit
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np

from src.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    AVAILABLE_MODELS,
    RESULTS_DIR,
)
from src.features import extract_features
from src.models import get_model_module
from src.controller import VirtualMouseController

HOTKEY_TO_MODEL = {
    ord("1"): "knn",
    ord("2"): "dt",
    ord("3"): "nb",
    ord("4"): "lr",
    ord("5"): "svm",
}


class TouchlessVirtualMouseApp:
    """Five-stage processing pipeline from camera capture to ML classification and OS control."""

    def __init__(self, initial_model_key: str = "svm", camera_index: int = 0, dry_run: bool = False):
        self.camera_index = camera_index
        self.dry_run = dry_run

        # Pre-load all 5 independent ML model modules so user can switch instantaneously with keys 1..5
        self.loaded_models = {}
        for key in AVAILABLE_MODELS:
            mod = get_model_module(key)
            mod.load_trained()
            self.loaded_models[key] = mod

        self.active_model_key = initial_model_key.lower()
        if self.active_model_key not in self.loaded_models:
            raise ValueError(f"Invalid model '{initial_model_key}'. Choose from {list(AVAILABLE_MODELS.keys())}")

        screen_w, screen_h = 1920, 1080
        if not dry_run:
            try:
                import pyautogui
                screen_w, screen_h = pyautogui.size()
            except Exception:
                pass

        self.controller = VirtualMouseController(
            screen_width=screen_w,
            screen_height=screen_h,
            camera_width=CAMERA_WIDTH,
            camera_height=CAMERA_HEIGHT,
            dry_run=dry_run,
        )
        self.last_active_action: str = "NONE"

    @property
    def active_model(self):
        return self.loaded_models[self.active_model_key]

    def switch_model(self, new_key: str):
        if new_key in self.loaded_models:
            self.active_model_key = new_key
            self.controller.reset_states()
            print(f"[Model Switched] Now testing independent module: {self.active_model.model_display_name}")

    def process_landmarks(self, landmarks_px: list[tuple[float, float]]) -> dict:
        """Run Stage 3 (Feature Extraction), Stage 4 (ML Classification), and Stage 5 (Action Control)."""
        feats = extract_features(landmarks_px)
        pose_id, confidence = self.active_model.predict_pose(feats)
        ctrl_out = self.controller.process_frame(pose_id=pose_id, landmarks_px=landmarks_px)
        if ctrl_out["action"] != "NONE":
            self.last_active_action = ctrl_out["action"]
        ctrl_out["confidence"] = confidence
        ctrl_out["model_key"] = self.active_model_key.upper()
        ctrl_out["model_name"] = self.active_model.model_display_name
        return ctrl_out

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        info: dict,
    ) -> np.ndarray:
        """Render visual feedback overlay (Paper Section III-G) + Independent ML Module selector."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 108), (20, 24, 35), -1)
        cv2.rectangle(overlay, (0, h - 38), (w, h), (20, 24, 35), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        mode_str = "DRY-RUN (SAFE PREVIEW)" if self.controller.dry_run else "LIVE OS CONTROL"
        mode_color = (0, 215, 255) if self.controller.dry_run else (80, 255, 80)

        cv2.putText(
            frame,
            f"Model [{info['model_key']}]: {info['model_name']} | FPS: {fps:.1f}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Pose: {info['pose_name']} ({info['confidence']*100:.1f}%) | Action: {info['action']} (Last: {self.last_active_action})",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (100, 255, 180),
            2,
        )
        cv2.putText(
            frame,
            f"Mode: {mode_str} | Dragging: {'ON' if self.controller.is_dragging else 'OFF'}",
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            mode_color,
            2,
        )
        cv2.putText(
            frame,
            "Hotkeys: [1]KNN  [2]DecisionTree  [3]NaiveBayes  [4]LogisticReg  [5]SVM  |  [d]DryRun  [q]Quit",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 220, 240),
            1,
        )
        return frame

    def run_webcam(self):
        from src.hand_tracker import MediaPipeHandTracker, ThreadedCameraCapture

        tracker = MediaPipeHandTracker(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        cap = ThreadedCameraCapture(
            camera_index=self.camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        )

        prev_time = time.perf_counter()
        smooth_fps = 30.0
        print("=== TOUCHLESS VIRTUAL MOUSE & WINDOWS TOUCHSCREEN GESTURES STARTED ===")
        print(f"Active Model: {self.active_model.model_display_name}")
        print("Press keys 1..5 on the camera window to test each ML model independently.")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # Stage 2: MediaPipe Hand Tracking (Threaded frame is already horizontally flipped)
            landmarks_px, frame = tracker.detect(frame, draw=True)

            if landmarks_px is not None:
                info = self.process_landmarks(landmarks_px)
            else:
                self.controller.process_frame(None, None)
                info = {
                    "pose_id": None,
                    "pose_name": "NO_HAND",
                    "action": "NONE",
                    "confidence": 0.0,
                    "model_key": self.active_model_key.upper(),
                    "model_name": self.active_model.model_display_name,
                }

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            smooth_fps = 0.85 * smooth_fps + 0.15 * inst_fps

            self.draw_hud(frame, smooth_fps, info)
            cv2.imshow("Touchless Virtual Mouse & Windows Gestures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in HOTKEY_TO_MODEL:
                self.switch_model(HOTKEY_TO_MODEL[key])
            elif key == ord("d"):
                self.controller.dry_run = not self.controller.dry_run

        tracker.close()
        cap.stop()
        cv2.destroyAllWindows()


def generate_demo_preview_image(output_path: Path = RESULTS_DIR / "demo_hud_preview.png"):
    """Generate a visual preview image demonstrating the HUD and skeleton rendering across models."""
    from tests.test_features import make_base_hand, set_finger_extended

    app = TouchlessVirtualMouseApp(initial_model_key="lr", dry_run=True)
    canvas = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), (38, 44, 54), dtype=np.uint8)

    # Construct POINT_POSE (Index finger extended)
    pts = make_base_hand(320.0, 310.0, scale=1.35)
    set_finger_extended(pts, 1, True, scale=1.35)

    # Draw hand skeleton lines
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    for a, b in connections:
        pa = (int(pts[a][0]), int(pts[a][1]))
        pb = (int(pts[b][0]), int(pts[b][1]))
        cv2.line(canvas, pa, pb, (0, 220, 120), 2, cv2.LINE_AA)

    for idx, (px, py) in enumerate(pts):
        color = (60, 80, 255) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
        cv2.circle(canvas, (int(px), int(py)), 5, color, -1, cv2.LINE_AA)

    # Feed confirm_frames times so Temporal Pose Stabilizer confirms POINT_POSE -> CURSOR_MOVE
    info = {}
    for _ in range(app.controller.confirm_frames):
        info = app.process_landmarks(pts)
    app.draw_hud(canvas, fps=60.0, info=info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[Demo Preview Saved]: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Hand Gesture Virtual Mouse & Windows Touchscreen Controller (5 Independent ML Models)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Select which independent ML model module to run ('knn', 'dt', 'nb', 'lr', 'svm')",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in safe preview mode without moving real OS cursor or triggering hotkeys",
    )
    parser.add_argument(
        "--benchmark-demo",
        action="store_true",
        help="Generate HUD preview image and verify all 5 models in pipeline without opening webcam",
    )
    parser.add_argument(
        "--classic-cv",
        action="store_true",
        help="Use basic OpenCV window instead of the Always-On-Top Floating Windows Debug UI",
    )
    args = parser.parse_args()

    if args.benchmark_demo:
        generate_demo_preview_image()
    else:
        app = TouchlessVirtualMouseApp(
            initial_model_key=args.model,
            camera_index=args.camera,
            dry_run=args.dry_run,
        )
        if args.classic_cv:
            app.run_webcam()
        else:
            from src.floating_ui import FloatingDebugOverlayUI
            ui = FloatingDebugOverlayUI(app)
            ui.run()
