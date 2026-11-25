import time
import numpy as np
from src.hand_tracker import MediaPipeHandTracker


def test_mediapipe_hand_tracker_initializes_and_processes_frame():
    tracker = MediaPipeHandTracker()
    blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks_px, annotated = tracker.detect(blank_bgr, draw=True)
    assert landmarks_px is None
    assert annotated.shape == (480, 640, 3)
    tracker.close()


def test_mediapipe_hand_tracker_high_fps_throughput():
    tracker = MediaPipeHandTracker()
    blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warm up TFLite XNNPACK threadpool
    for _ in range(8):
        tracker.detect(blank_bgr, draw=False)
    n_frames = 50
    fps = 0.0
    for _ in range(2):
        t0 = time.perf_counter()
        for _ in range(n_frames):
            tracker.detect(blank_bgr, draw=False)
        elapsed = time.perf_counter() - t0
        fps = n_frames / max(elapsed, 1e-6)
        if fps > 30.0:
            break
    tracker.close()
    assert fps > 30.0
