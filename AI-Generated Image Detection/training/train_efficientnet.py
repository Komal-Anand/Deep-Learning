"""
Two-Stage Transfer Learning Training Pipeline for EfficientNet-B0.

Stage 1: Feature Extraction / Head Warmup
- Backbone weights frozen
- Classifier head trained with AdamW (lr=1e-3)

Stage 2: Top-Block Fine-Tuning
- Top inverted residual blocks unfrozen
- Entire classifier head fine-tuned with reduced lr (lr=1e-4) and Cosine Annealing
- Checkpoints saved based on validation loss/accuracy.
"""

import os
import sys
import json
import random
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from models.efficientnet_model import build_efficientnet
from utils.preprocessing import get_train_transforms, get_val_transforms


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Sets deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_efficientnet_b0(
    head_epochs: int = 3,
    finetune_epochs: int = 5,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    head_lr: float = config.DEFAULT_LR_HEAD,
    finetune_lr: float = config.DEFAULT_LR_FINE_TUNE,
    patience: int = config.PATIENCE
) -> Dict[str, Any]:
    """
    Executes two-stage transfer learning pipeline for EfficientNet-B0.
    """
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting EfficientNet-B0 Transfer Learning on {device} ===")

    # Check that splits exist
    if not (config.TRAIN_DIR / "Real").exists() or not (config.VAL_DIR / "Real").exists():
        raise FileNotFoundError(
            f"Dataset splits not found in {config.SPLITS_DIR}. "
            "Please run training/prepare_data.py first."
        )

    # DataLoaders
    train_dataset = ImageFolder(root=str(config.TRAIN_DIR), transform=get_train_transforms())
    val_dataset = ImageFolder(root=str(config.VAL_DIR), transform=get_val_transforms())

    print(f"Loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.DEFAULT_NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.DEFAULT_NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # Instantiate model
    model = build_efficientnet(num_classes=config.NUM_CLASSES, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs": [],
        "stages": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    global_epoch = 0

    # ==========================================
    # STAGE 1: Classifier Head Warmup (Backbone Frozen)
    # ==========================================
    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    model.freeze_backbone()
    print(f"Parameters in Stage 1: {model.count_parameters()}")

    optimizer_head = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr,
        weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(1, head_epochs + 1):
        global_epoch += 1
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer_head.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_head.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        history["epochs"].append(global_epoch)
        history["stages"].append("Head Warmup")
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(
            f"[Stage 1] Epoch [{epoch:02d}/{head_epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            history["best_epoch"] = global_epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            history["best_val_acc"] = round(best_val_acc, 4)

            torch.save({
                "epoch": global_epoch,
                "stage": "head_warmup",
                "model_architecture": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_to_idx": train_dataset.class_to_idx
            }, config.EFFICIENTNET_WEIGHTS_PATH)

    # ==========================================
    # STAGE 2: Fine-Tuning Top Backbone Blocks
    # ==========================================
    print("\n--- STAGE 2: Fine-Tuning Top Backbone Blocks ---")
    model.unfreeze_top_blocks(num_blocks=3)
    print(f"Parameters in Stage 2: {model.count_parameters()}")

    optimizer_fine = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=finetune_lr,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_fine,
        T_max=finetune_epochs,
        eta_min=1e-6
    )

    patience_counter = 0

    for epoch in range(1, finetune_epochs + 1):
        global_epoch += 1
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer_fine.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_fine.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        scheduler_fine.step()

        history["epochs"].append(global_epoch)
        history["stages"].append("Fine-Tuning")
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(
            f"[Stage 2] Epoch [{epoch:02d}/{finetune_epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            history["best_epoch"] = global_epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            history["best_val_acc"] = round(best_val_acc, 4)
            patience_counter = 0

            torch.save({
                "epoch": global_epoch,
                "stage": "fine_tuning",
                "model_architecture": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_to_idx": train_dataset.class_to_idx
            }, config.EFFICIENTNET_WEIGHTS_PATH)
            print(f"  --> Saved new best checkpoint to {config.EFFICIENTNET_WEIGHTS_PATH.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping in fine-tuning phase at epoch {epoch}.")
                break

    # Save history
    history_file = config.ARTIFACTS_DIR / "efficientnet_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nEfficientNet-B0 training complete! Best Val Acc: {best_val_acc * 100:.2f}%. Checkpoint saved.")
    return history


if __name__ == "__main__":
    train_efficientnet_b0()

