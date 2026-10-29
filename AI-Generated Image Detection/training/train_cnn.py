"""
Training Pipeline for Model A: Custom Convolutional Neural Network (CNN).

Features:
- Reproducible random seeding across PyTorch, NumPy, and random.
- Data loading with standard augmentations on training set and deterministic validation.
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Real-time epoch metric tracking (loss, accuracy, learning rate).
- Best checkpoint saving and early stopping based on validation loss.
- Serializes training history to JSON for downstream plotting and analysis.
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

from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from models.cnn_model import build_custom_cnn
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


def train_custom_cnn(
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    learning_rate: float = config.DEFAULT_LR_HEAD,
    patience: int = config.PATIENCE
) -> Dict[str, Any]:
    """
    Executes training loop for the Custom CNN baseline model.
    """
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Custom CNN Training on {device} ===")

    # Check that splits exist
    if not (config.TRAIN_DIR / "Real").exists() or not (config.VAL_DIR / "Real").exists():
        raise FileNotFoundError(
            f"Dataset splits not found in {config.SPLITS_DIR}. "
            "Please run training/prepare_data.py first."
        )

    # Dataset and DataLoaders
    train_dataset = ImageFolder(root=str(config.TRAIN_DIR), transform=get_train_transforms())
    val_dataset = ImageFolder(root=str(config.VAL_DIR), transform=get_val_transforms())

    print(f"Loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    print(f"Class mapping: {train_dataset.class_to_idx}")

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

    # Initialize Model
    model = build_custom_cnn(num_classes=config.NUM_CLASSES)
    model.to(device)
    print(f"Model Parameters: {model.count_parameters()}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation Phase
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

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Track history
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            history["best_epoch"] = epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            history["best_val_acc"] = round(best_val_acc, 4)
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_architecture": "CustomCNN",
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_to_idx": train_dataset.class_to_idx
            }
            torch.save(checkpoint, config.CNN_WEIGHTS_PATH)
            print(f"  --> Saved new best checkpoint to {config.CNN_WEIGHTS_PATH.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    total_time = time.time() - start_time
    history["training_time_seconds"] = round(total_time, 2)

    # Save training history
    history_file = config.ARTIFACTS_DIR / "cnn_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc * 100:.2f}%.")
    return history


if __name__ == "__main__":
    train_custom_cnn()

