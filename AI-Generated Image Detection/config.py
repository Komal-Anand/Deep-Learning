"""
Configuration module for AI-Generated Image Detection.
Defines all project paths, hyperparameters, preprocessing constants, and hardware configurations.
"""

from pathlib import Path
import os

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_REAL_DIR = DATA_DIR / "real"
RAW_AI_DIR = DATA_DIR / "ai_generated"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "val"
TEST_DIR = SPLITS_DIR / "test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Model Checkpoint Paths
EFFICIENTNET_WEIGHTS_PATH = ARTIFACTS_DIR / "best_efficientnet_b0.pth"
CNN_WEIGHTS_PATH = ARTIFACTS_DIR / "best_custom_cnn.pth"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

# Dataset and Classes (Alphabetically sorted to match torchvision.datasets.ImageFolder)
CLASS_NAMES = ["AI-Generated", "Real"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_NAME = {0: "AI-Generated", 1: "Real"}
NAME_TO_LABEL = {"AI-Generated": 0, "Real": 1}

# Input Image Specifications (Standard EfficientNet-B0 Resolution)
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3

# Preprocessing Normalization (ImageNet Standard)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Reproducibility Seed
RANDOM_SEED = 42

# Training Hyperparameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0  # 0 for maximum stability on Windows
DEFAULT_EPOCHS = 10
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_FINE_TUNE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4  # Early stopping patience

# Security & Validation
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit

# Ensure critical directories exist
for directory in [DATA_DIR, RAW_REAL_DIR, RAW_AI_DIR, SPLITS_DIR, ARTIFACTS_DIR, REPORTS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

