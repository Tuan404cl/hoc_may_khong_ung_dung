"""Step 4 & 5: Run all 5 independent ML model modules and generate comparative charts & error analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATASET_CSV_PATH, RESULTS_DIR, POSE_CLASSES
from src.models import MODEL_REGISTRY


def evaluate_all_independent_models() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_metrics = {}

    print("=" * 80)
    print("TRAINING & EVALUATING ALL 5 INDEPENDENT MACHINE LEARNING MODULES")
    print("=" * 80)

    for key, cls in MODEL_REGISTRY.items():
        mod = cls()
        m = mod.train_and_evaluate(csv_path=DATASET_CSV_PATH)
        all_metrics[key] = m
        summary_rows.append({
            "Model Key": key.upper(),
            "Model Name": m["model_name"],
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision Macro (%)": round(m["precision_macro"] * 100, 2),
            "Recall Macro (%)": round(m["recall_macro"] * 100, 2),
            "F1-Score Macro (%)": round(m["f1_macro"] * 100, 2),
            "Train Time (ms)": m["train_time_ms"],
            "Inference (ms/sample)": m["inference_per_sample_ms"],
        })

    comp_df = pd.DataFrame(summary_rows)
    comp_csv = RESULTS_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print("\n[Comparison Table Saved]:", comp_csv)
    print(comp_df.to_string(index=False))

    _plot_comparison_bar_chart(comp_df)
    _plot_combined_confusion_matrices(all_metrics)
    return comp_df


def _plot_comparison_bar_chart(comp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(comp_df))
    width = 0.20

    metrics_cols = ["Accuracy (%)", "Precision Macro (%)", "Recall Macro (%)", "F1-Score Macro (%)"]
    colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]

    for idx, (col, color) in enumerate(zip(metrics_cols, colors)):
        vals = comp_df[col].to_numpy()
        bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=col.replace(" (%)", ""), color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model Key"] + "\n(" + comp_df["Model Name"].str.split("(").str[0].str.strip() + ")")
    ax.set_ylim(70, 103)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        f"Comparison of 5 Independent Machine Learning Models ({len(POSE_CLASSES)} Natural Gesture Poses)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_chart.png", dpi=180)
    plt.close(fig)


def _plot_combined_confusion_matrices(all_metrics: dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    short_labels = [str(k) for k in POSE_CLASSES.keys()]

    for idx, (key, m) in enumerate(all_metrics.items()):
        ax = axes_flat[idx]
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
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
                    fontsize=7.5,
                )
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(
            f"{key.upper()}: Acc={m['accuracy']*100:.2f}% | F1={m['f1_macro']*100:.2f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Class ID")
        ax.set_ylabel("True Class ID")

    # Legend of class IDs in the 6th subplot
    ax_legend = axes_flat[5]
    ax_legend.axis("off")
    legend_text = f"{len(POSE_CLASSES)} Natural Ergonomic Hand Pose Classes:\n\n" + "\n".join(
        f"Class {k}: {v}" for k, v in POSE_CLASSES.items()
    )
    ax_legend.text(0.05, 0.92, legend_text, va="top", ha="left", fontsize=10.5, family="monospace")

    plt.suptitle("Confusion Matrices of All 5 Independent Machine Learning Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "all_confusion_matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    evaluate_all_independent_models()
