"""
Model Evaluation and Robustness Experimentation Pipeline.

Performs:
1. Held-out test set evaluation for Custom CNN and EfficientNet-B0.
2. Comprehensive metric computation:
   - Overall Accuracy
   - Macro & Weighted Precision, Recall, F1-Score
   - Per-class metrics
   - Confusion Matrix
   - ROC Curve and AUC Score
3. Robustness Perturbation Testing:
   - Clean Test Set (baseline)
   - JPEG Compression (Quality = 50)
   - Gaussian Defocus Blur (Radius = 2.0)
   - Downscale & Upsample (0.5x resolution)
4. Saves high-res plots and JSON metrics to `reports/`.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)

import config
from models.cnn_model import build_custom_cnn
from models.efficientnet_model import build_efficientnet
from utils.preprocessing import (
    get_val_transforms, apply_jpeg_compression,
    apply_gaussian_blur, apply_resize_downscale
)
from utils.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, plot_model_comparison, plot_robustness_experiment
)


def load_checkpoint_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Safely loads model weights onto device."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on a DataLoader.
    Returns (y_true, y_pred, y_probs).
    """
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            y_prob_list.extend(probs[:, 1].cpu().numpy())  # Probability of positive class (AI-Generated)

    return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)


def run_robustness_eval(
    model: torch.nn.Module,
    test_dir: Path,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model accuracy under 3 simulated real-world image perturbations:
    1. Clean Test Images
    2. JPEG Compression (Q=50)
    3. Gaussian Defocus Blur (r=2.0)
    4. Downscale & Upsample (0.5x)
    """
    from PIL import Image

    conditions = {
        "Clean Baseline": lambda img: img,
        "JPEG (Q=50)": lambda img: apply_jpeg_compression(img, quality=50),
        "Gaussian Blur": lambda img: apply_gaussian_blur(img, radius=2.0),
        "Downscale (0.5x)": lambda img: apply_resize_downscale(img, scale=0.5)
    }

    results = {}
    base_transform = get_val_transforms()

    # Iterate over conditions
    for condition_name, perturb_func in conditions.items():
        correct = 0
        total = 0

        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            class_folder = test_dir / class_name
            if not class_folder.exists():
                continue

            for ext in config.ALLOWED_EXTENSIONS:
                for img_path in class_folder.glob(f"*{ext}"):
                    try:
                        with Image.open(img_path) as raw_img:
                            img = raw_img.convert("RGB")
                        perturbed = perturb_func(img)
                        tensor = base_transform(perturbed).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(tensor)
                            pred = torch.argmax(outputs, dim=1).item()

                        if pred == class_idx:
                            correct += 1
                        total += 1
                    except Exception:
                        continue

        accuracy = correct / max(1, total)
        results[condition_name] = round(accuracy, 4)

    return results


def run_full_evaluation() -> Dict[str, Any]:
    """
    Main evaluation routine:
    - Evaluates Custom CNN and EfficientNet-B0 on held-out test split.
    - Generates all plots and saves metrics to reports/.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Comprehensive Model Evaluation on {device} ===")

    if not config.TEST_DIR.exists() or not (config.TEST_DIR / "Real").exists():
        raise FileNotFoundError(
            f"Test dataset not found at {config.TEST_DIR}. Run training/prepare_data.py first."
        )

    test_dataset = ImageFolder(root=str(config.TEST_DIR), transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DEFAULT_NUM_WORKERS
    )
    print(f"Held-out test set size: {len(test_dataset)} images.")

    metrics_output = {
        "test_dataset_size": len(test_dataset),
        "class_names": config.CLASS_NAMES,
        "models": {}
    }

    # 1. Custom CNN Evaluation
    if config.CNN_WEIGHTS_PATH.exists():
        print("\n--- Evaluating Baseline Custom CNN ---")
        cnn_model = build_custom_cnn(num_classes=config.NUM_CLASSES)
        load_checkpoint_weights(cnn_model, config.CNN_WEIGHTS_PATH, device)

        y_true, y_pred, y_probs = evaluate_dataset(cnn_model, test_loader, device)

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = float(auc(fpr, tpr))
        report_str = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0)

        # Robustness
        robustness_scores = run_robustness_eval(cnn_model, config.TEST_DIR, device)

        # Plots
        plot_confusion_matrix(cm, config.CLASS_NAMES, "Custom CNN")
        plot_roc_curve(fpr, tpr, roc_auc, "Custom CNN")
        plot_robustness_experiment(robustness_scores, "Custom CNN")

        # Check for training history to plot curves
        cnn_hist_path = config.ARTIFACTS_DIR / "cnn_training_history.json"
        if cnn_hist_path.exists():
            with open(cnn_hist_path, "r") as f:
                cnn_hist = json.load(f)
            plot_training_curves(cnn_hist, "Custom CNN")

        metrics_output["models"]["Custom CNN"] = {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
            "robustness": robustness_scores
        }
        print(f"Custom CNN Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    else:
        print(f"Custom CNN weights not found at {config.CNN_WEIGHTS_PATH}. Skipping.")

    # 2. EfficientNet-B0 Evaluation
    if config.EFFICIENTNET_WEIGHTS_PATH.exists():
        print("\n--- Evaluating EfficientNet-B0 (Transfer Learning) ---")
        eff_model = build_efficientnet(num_classes=config.NUM_CLASSES, pretrained=False)
        load_checkpoint_weights(eff_model, config.EFFICIENTNET_WEIGHTS_PATH, device)

        y_true, y_pred, y_probs = evaluate_dataset(eff_model, test_loader, device)

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = float(auc(fpr, tpr))
        report_str = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0)

        # Robustness
        robustness_scores = run_robustness_eval(eff_model, config.TEST_DIR, device)

        # Plots
        plot_confusion_matrix(cm, config.CLASS_NAMES, "EfficientNet-B0")
        plot_roc_curve(fpr, tpr, roc_auc, "EfficientNet-B0")
        plot_robustness_experiment(robustness_scores, "EfficientNet-B0")

        # Training curves
        eff_hist_path = config.ARTIFACTS_DIR / "efficientnet_training_history.json"
        if eff_hist_path.exists():
            with open(eff_hist_path, "r") as f:
                eff_hist = json.load(f)
            plot_training_curves(eff_hist, "EfficientNet-B0")

        metrics_output["models"]["EfficientNet-B0"] = {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
            "robustness": robustness_scores
        }
        print(f"EfficientNet-B0 Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    else:
        print(f"EfficientNet-B0 weights not found at {config.EFFICIENTNET_WEIGHTS_PATH}. Skipping.")

    # 3. Model Comparison Chart (if at least one model exists)
    if len(metrics_output["models"]) > 0:
        comparison_dict = {
            m_name: {
                "accuracy": m_data["accuracy"],
                "precision": m_data["precision"],
                "recall": m_data["recall"],
                "f1_score": m_data["f1_score"]
            }
            for m_name, m_data in metrics_output["models"].items()
        }
        plot_model_comparison(comparison_dict)

    # Save metrics JSON
    with open(config.EVALUATION_METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)

    print(f"\nEvaluation complete! Report saved to {config.EVALUATION_METRICS_PATH}")
    return metrics_output


if __name__ == "__main__":
    run_full_evaluation()

