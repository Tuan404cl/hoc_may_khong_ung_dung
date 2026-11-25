"""Step 2b: Interactive Webcam Tool to Collect Custom Hand Gesture Samples into CSV."""

import argparse
import cv2
import pandas as pd

from src.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    DATASET_CSV_PATH,
    POSE_CLASSES,
)
from src.features import extract_features, get_feature_names
from src.hand_tracker import MediaPipeHandTracker


def run_collection(camera_index: int = 0, output_csv: str = str(DATASET_CSV_PATH)):
    tracker = MediaPipeHandTracker(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    feature_names = get_feature_names()
    collected_rows = []

    print("=== WEBCAM GESTURE DATA COLLECTOR ===")
    for k, name in POSE_CLASSES.items():
        print(f"  Press '{k}' -> Record sample for Class {k}: {name}")
    print("  Press 'q' or ESC -> Save and Quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        landmarks_px, frame = tracker.detect(frame, draw=True)

        cv2.putText(
            frame,
            f"Collected: {len(collected_rows)} | Keys 0-9 to record Class 0..9, 'q' to save",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Collect Hand Gestures", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        target_class = None
        if ord("0") <= key <= ord("9"):
            candidate = key - ord("0")
            if candidate in POSE_CLASSES:
                target_class = candidate

        if target_class is not None and landmarks_px is not None:
            feats = extract_features(landmarks_px)
            row = {fname: float(val) for fname, val in zip(feature_names, feats)}
            row["label"] = target_class
            row["pose_name"] = POSE_CLASSES[target_class]
            row["source"] = "webcam_user"
            collected_rows.append(row)
            print(f"[Recorded] Class {target_class} ({POSE_CLASSES[target_class]}) - Total new: {len(collected_rows)}")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

    if collected_rows:
        new_df = pd.DataFrame(collected_rows)
        try:
            existing_df = pd.read_csv(output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = new_df
        combined_df.to_csv(output_csv, index=False)
        print(f"Saved {len(collected_rows)} new samples to {output_csv} (Total: {len(combined_df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect custom MediaPipe hand gesture samples via Webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--output", type=str, default=str(DATASET_CSV_PATH), help="Output CSV path")
    args = parser.parse_args()
    run_collection(args.camera, args.output)
