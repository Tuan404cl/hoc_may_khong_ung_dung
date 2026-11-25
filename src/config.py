"""Global configuration and constants from the paper & Windows Touchscreen specification."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_GITHUB_CSV_PATH = DATA_DIR / "raw_keypoint_github.csv"
DATASET_CSV_PATH = DATA_DIR / "hand_gestures_dataset.csv"

# Paper Section III-B & IV-B parameters
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
TARGET_FPS: int = 30
MAX_NUM_HANDS: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.75
MIN_TRACKING_CONFIDENCE: float = 0.75

# Paper Section III-C, III-D, III-E & IV-E tuned parameters
SMOOTHING_ALPHA: float = 8.0            # Eq. (4): Exponential moving average factor alpha = 8
ACTION_DELAY: float = 0.4               # Section III-E: 400ms temporal debouncing
PINCH_THRESHOLD: float = 30.0           # Table I: d(thumb, index) < 30 pixels
CURSOR_SPACING_THRESHOLD: float = 40.0  # Legacy reference threshold

# Motion tracking & ergonomic stability parameters
VERTICAL_MOTION_THRESHOLD: float = 28.0
HORIZONTAL_MOTION_THRESHOLD: float = 32.0
PINCH_SPREAD_DELTA_THRESHOLD: float = 18.0
CONFIRM_FRAMES: int = 3                 # Consecutive frames required to confirm pose change (anti-ghost click)
DRAG_HOLD_DURATION: float = 0.35        # Seconds holding Thumb+Index pinch before entering Drag & Drop
DOUBLE_PINCH_WINDOW: float = 0.45       # Max interval (s) between two quick pinches to emit Double Click
STATIC_HOLD_DURATION: float = 0.85      # Wall-clock hold duration (s) for 3/4-finger Search & Action Center
PINCH_APPROACH_RATIO: float = 0.45      # d(4,8)/palm_ref below which cursor smoothing increases 4x (Pinch Freeze)

# 10 Orthogonal Natural Ergonomic Hand Pose Classes (Point & Pinch + Windows Touch)
POSE_CLASSES: dict[int, str] = {
    0: "POINT_POSE",              # [0,1,0,0,0] Index finger pointing -> Cursor Move (+ Pinch-Approach Freeze)
    1: "PINCH_INDEX_POSE",        # Thumb+Index pinched, other fingers curled -> Tap=LeftClick, 2x=DoubleClick, Hold>=0.35s=Drag
    2: "PINCH_MIDDLE_POSE",       # Thumb+Middle pinched with Index extended -> Right Click (latched once per pinch)
    3: "TWO_FINGER_SCROLL_POSE",  # [0,1,1,0,0] Index+Middle up (unified class) -> Vertical & Horizontal Scroll
    4: "ZOOM_L_POSE",             # [1,1,0,0,0] Thumb+Index L-shape open -> Spread/Pinch Zoom In / Zoom Out
    5: "THREE_FINGER_POSE",       # [0,1,1,1,0] Index+Middle+Ring up -> Win Touch 3-Finger Swipe & Search
    6: "FOUR_FINGER_POSE",        # [0,1,1,1,1] Four fingers up -> Win Touch 4-Finger Swipe & Action Center
    7: "OPEN_PALM_POSE",          # [1,1,1,1,1] All 5 fingers open -> Volume Up/Down & Win Widgets Swipe
    8: "SHAKA_BRIGHTNESS_POSE",   # [1,0,0,0,1] Thumb+Pinky extended -> Brightness Up/Down (zero conflict with Zoom)
    9: "FIST_IDLE_POSE",          # [0,0,0,0,0] Closed fist -> Safe Rest / Clutch (no OS action)
}

POSE_NAME_TO_ID: dict[str, int] = {v: k for k, v in POSE_CLASSES.items()}

# 5 Independent Machine Learning Models required by the course syllabus
AVAILABLE_MODELS: dict[str, str] = {
    "knn": "K-Nearest Neighbors (KNN)",
    "dt": "Decision Tree",
    "nb": "Gaussian Naive Bayes",
    "lr": "Logistic Regression",
    "svm": "Support Vector Machine (SVM)",
}
