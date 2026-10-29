"""
Visualization and Evaluation Plotting Module.

Generates publication-quality figures:
- Training and validation loss/accuracy curves.
- Annotated confusion matrices.
- ROC Curves with Area Under Curve (AUC).
- Model comparison charts (Custom CNN vs EfficientNet-B0).
- Robustness degradation curves (Clean vs JPEG vs Blur vs Downscale).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers/headless
import matplotlib.pyplot as plt
import seaborn as sns

import config


def setup_plot_style():
    """Sets modern clean aesthetic for research figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_training_curves(history: Dict[str, Any], model_name: str, save_path: Optional[Path] = None) -> Path:
    """
    Plots dual-axis or subplots of Training & Validation Loss and Accuracy across epochs.
    """
    setup_plot_style()
    epochs = history.get("epochs", list(range(1, len(history["train_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3b82f6", linewidth=2.2, marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2.2, linestyle="--", marker="s", markersize=4)
    ax1.set_title(f"{model_name}: Loss vs Epochs", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Accuracy Curve
    train_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in history["train_acc"]]
    val_acc_pct = [acc * 100 if acc <= 1.0 else acc for acc in history["val_acc"]]

    ax2.plot(epochs, train_acc_pct, label="Train Accuracy", color="#10b981", linewidth=2.2, marker="o", markersize=4)
    ax2.plot(epochs, val_acc_pct, label="Val Accuracy", color="#8b5cf6", linewidth=2.2, linestyle="--", marker="s", markersize=4)
    ax2.set_title(f"{model_name}: Accuracy vs Epochs", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_training_curves.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    model_name: str = "EfficientNet-B0",
    save_path: Optional[Path] = None
) -> Path:
    """Plots formatted normalized confusion matrix heatmap."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=1.2,
        linecolor="#CBD5E1",
        annot_kws={"fontsize": 11, "fontweight": "medium"}
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="semibold")

    plt.tight_layout()

    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    model_name: str = "EfficientNet-B0",
    save_path: Optional[Path] = None
) -> Path:
    """Plots Receiver Operating Characteristic (ROC) curve."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)

    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(f"ROC Curve: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_model_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots a multi-bar comparison across Custom CNN and EfficientNet-B0
    for Accuracy, Precision, Recall, and F1-Score.
    """
    setup_plot_style()
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    models = list(metrics_summary.keys())
    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    colors = ["#64748b", "#3b82f6"]

    for i, model in enumerate(models):
        values = [metrics_summary[model].get(k, 0.0) * 100 for k in metric_keys]
        offset = (i - len(models)/2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)], edgecolor="#1e293b", linewidth=0.8)
        # Value labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="semibold")
    ax.set_title("Model Performance Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="semibold")
    ax.set_ylim([0, 110])
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is None:
        save_path = config.REPORTS_DIR / "model_comparison_chart.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_robustness_experiment(
    robustness_data: Dict[str, float],
    model_name: str = "EfficientNet-B0",
    save_path: Optional[Path] = None
) -> Path:
    """Plots robustness degradation chart under perturbations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    conditions = list(robustness_data.keys())
    accuracies = [robustness_data[k] * 100 for k in conditions]
    bar_colors = ["#10b981" if "Clean" in c else "#f59e0b" for c in conditions]

    bars = ax.bar(conditions, accuracies, color=bar_colors, edgecolor="#1e293b", linewidth=0.8, width=0.55)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Robustness Under Perturbations: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim([0, 110])
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is None:
        save_path = config.REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_robustness_chart.png"

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path

