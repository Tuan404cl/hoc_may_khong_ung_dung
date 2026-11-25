"""Base evaluation, persistence, and error analysis engine for each independent ML model module."""

import json
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_CSV_PATH, MODELS_DIR, RESULTS_DIR, POSE_CLASSES
from src.features import get_feature_names


class BaseGestureModelModule(ABC):
    """Abstract base class for each independent Machine Learning gesture recognition module."""

    model_key: str = ""
    model_display_name: str = ""

    def __init__(self, model_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir) / self.model_key
        self.model = None
        self.scaler = None

    @abstractmethod
    def build_estimator(self):
        """Construct and return the scikit-learn classifier instance."""
        raise NotImplementedError

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.model_key}_model.pkl"

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / f"{self.model_key}_scaler.pkl"

    def train_and_evaluate(
        self,
        csv_path: Path = DATASET_CSV_PATH,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> dict:
        """Train this independent model, compute Accuracy/Precision/Recall/F1/CM, and save error analysis."""
        df = pd.read_csv(csv_path)
        feature_cols = get_feature_names()
        X = df[feature_cols].to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = self.build_estimator()

        t0 = time.perf_counter()
        self.model.fit(X_train_scaled, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        y_pred = self.model.predict(X_test_scaled)
        infer_total_ms = (time.perf_counter() - t1) * 1000.0
        infer_per_sample_ms = infer_total_ms / max(len(X_test_scaled), 1)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=list(POSE_CLASSES.keys()))

        cls_report_dict = classification_report(
            y_test,
            y_pred,
            labels=list(POSE_CLASSES.keys()),
            target_names=[POSE_CLASSES[i] for i in POSE_CLASSES],
            output_dict=True,
            zero_division=0,
        )

        # Save trained model & scaler
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metrics, confusion matrix plot, and error analysis
        self.results_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "model_key": self.model_key,
            "model_name": self.model_display_name,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "train_time_ms": round(train_time_ms, 2),
            "inference_per_sample_ms": round(infer_per_sample_ms, 4),
            "confusion_matrix": cm.tolist(),
            "per_class_report": cls_report_dict,
        }

        (self.results_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._plot_confusion_matrix(cm, acc, f1_macro)
        self._write_error_analysis(y_test, y_pred, cm, metrics)

        print(
            f"[{self.model_display_name}] Acc={acc*100:.2f}% | "
            f"Prec={prec_macro*100:.2f}% | Rec={rec_macro*100:.2f}% | "
            f"F1={f1_macro*100:.2f}% | Infer={infer_per_sample_ms:.4f}ms/sample"
        )
        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray, acc: float, f1_macro: float):
        labels = [f"{k}: {v.replace('_POSE', '')}" for k, v in POSE_CLASSES.items()]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(int(cm[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8.5,
                    fontweight="bold" if i == j else "normal",
                )

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            f"Confusion Matrix - {self.model_display_name}\n"
            f"(Accuracy: {acc*100:.2f}%, Macro F1: {f1_macro*100:.2f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Gesture Class", fontsize=10)
        ax.set_ylabel("True Gesture Class", fontsize=10)
        plt.tight_layout()
        fig.savefig(self.results_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)

    def _write_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, cm: np.ndarray, metrics: dict):
        mis_pairs = []
        n_classes = len(POSE_CLASSES)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    mis_pairs.append((int(cm[i, j]), i, j))
        mis_pairs.sort(reverse=True)

        total_errors = int(np.sum(y_true != y_pred))
        lines = [
            f"# Phân tích Kết quả & Trường hợp Dự đoán sai - {self.model_display_name}",
            "",
            "## 1. Chỉ số Đánh giá Tổng quát (Trên tập Kiểm thử 20%)",
            f"- **Accuracy:** `{metrics['accuracy']*100:.2f}%`",
            f"- **Precision (Macro):** `{metrics['precision_macro']*100:.2f}%`",
            f"- **Recall (Macro):** `{metrics['recall_macro']*100:.2f}%`",
            f"- **F1-Score (Macro):** `{metrics['f1_macro']*100:.2f}%`",
            f"- **Tổng số mẫu kiểm thử:** `{metrics['test_samples']}` mẫu (`{total_errors}` mẫu dự đoán sai)",
            f"- **Thời gian suy luận trung bình:** `{metrics['inference_per_sample_ms']:.4f} ms/frame`",
            "",
            "## 2. Top các cặp Cử chỉ bị Nhầm lẫn nhiều nhất (Confusion Matrix Off-Diagonal)",
            "| Hạng | Nhãn Thực tế (True) | Nhãn Dự đoán Nhầm (Predicted) | Số mẫu nhầm | Nguyên nhân Hình học & Giải thích |",
            "| :--- | :--- | :--- | :---: | :--- |",
        ]

        for rank, (cnt, true_i, pred_j) in enumerate(mis_pairs[:8], start=1):
            t_name = POSE_CLASSES[true_i]
            p_name = POSE_CLASSES[pred_j]
            reason = self._explain_confusion_pair(true_i, pred_j)
            lines.append(f"| {rank} | `{t_name}` | `{p_name}` | **{cnt}** | {reason} |")

        (self.results_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _explain_confusion_pair(true_i: int, pred_j: int) -> str:
        pair = {true_i, pred_j}
        if pair == {0, 2}:
            return "Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes)."
        if pair == {0, 4}:
            return "Cả `POINT_POSE` `[0,1,0,0,0]` và `ZOOM_L_POSE` `[1,1,0,0,0]` đều duỗi ngón trỏ; chỉ khác nhau ở độ mở rộng của ngón cái theo trục ngang."
        if pair == {1, 2}:
            return "Cả `PINCH_INDEX_POSE` và `PINCH_MIDDLE_POSE` đều thực hiện thao tác chụm ngón cái với một ngón dài ở vùng chuyển tiếp."
        if pair == {1, 9}:
            return "Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp."
        if pair == {5, 6} or pair == {6, 7}:
            return "Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết."
        return "Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn."

    def load_trained(self):
        """Load persisted model and scaler from disk."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            self.train_and_evaluate()
        else:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

    def predict_pose(self, features: np.ndarray) -> tuple[int, float]:
        """Predict static pose class ID (0..9) and confidence score (0.0..1.0) for a single 54-D feature vector."""
        if self.model is None or self.scaler is None:
            self.load_trained()
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred_id = int(self.model.predict(x_scaled)[0])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_scaled)[0]
            conf = float(np.max(probs))
        elif hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(x_scaled)).ravel()
            exp_s = np.exp(scores - np.max(scores))
            conf = float(np.max(exp_s / np.sum(exp_s)))
        else:
            conf = 1.0
        return pred_id, conf
