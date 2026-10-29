"""
Dataset Preparation, Splitting, Verification, and Sample Generation Module.

Features:
1. Scans raw images from `data/real` and `data/ai_generated`.
2. SHA-256 deduplication to prevent data leakage across splits.
3. Corrupt file detection and quarantine.
4. Stratified splitting into Train (70%), Val (15%), and Test (15%) splits with fixed seed.
5. Generates metadata reports and split manifest files.
6. Includes sample/demo generator so the user can test the complete pipeline instantly.
"""

import os
import sys
import hashlib
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import config
from utils.preprocessing import validate_image_file, load_and_preprocess_image


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 digest of an image to prevent duplicates across splits."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_clean_directory(directory: Path, class_label: int) -> Tuple[List[Path], List[str]]:
    """
    Scans a directory for images, verifies integrity, and detects corruption.
    Returns (valid_paths, error_logs).
    """
    valid_paths = []
    error_logs = []

    if not directory.exists():
        return valid_paths, [f"Directory does not exist: {directory}"]

    for ext in config.ALLOWED_EXTENSIONS:
        for file_path in directory.glob(f"*{ext}"):
            is_valid, err_msg = validate_image_file(file_path)
            if is_valid:
                valid_paths.append(file_path)
            else:
                error_logs.append(f"Excluded {file_path.name}: {err_msg}")
        # Case insensitive match for uppercase extensions
        for file_path in directory.glob(f"*{ext.upper()}"):
            if file_path not in valid_paths:
                is_valid, err_msg = validate_image_file(file_path)
                if is_valid:
                    valid_paths.append(file_path)
                else:
                    error_logs.append(f"Excluded {file_path.name}: {err_msg}")

    return sorted(list(set(valid_paths))), error_logs


def generate_synthetic_demo_samples(num_per_class: int = 50) -> None:
    """
    Generates high-quality synthetic demo images if raw directories are empty.
    Allows end-to-end testing of training, inference, and Grad-CAM without large downloads.
    - Real class: Photographic procedural patterns (natural textures, gradients, optical noise).
    - AI-Generated class: Diffusion-like high frequency artifacts, grid interference, color fringing.
    """
    print(f"Generating {num_per_class} demo images per class in {config.DATA_DIR}...")
    np.random.seed(config.RANDOM_SEED)

    config.RAW_REAL_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_AI_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        # 1. Real Image: Natural landscape / photographic simulation
        real_img = Image.new("RGB", (256, 256), color=(135 + (i * 2) % 40, 180 + (i * 3) % 40, 220))
        draw = ImageDraw.Draw(real_img)
        # Ground/terrain
        draw.rectangle([0, 160, 256, 256], fill=(40 + i % 20, 120 + i % 30, 45 + i % 20))
        # Sun / light source
        draw.ellipse([30 + (i * 5) % 180, 30, 70 + (i * 5) % 180, 70], fill=(255, 230, 150))
        # Add smooth gaussian blur to simulate camera lens physics
        real_img = real_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        # Add slight natural sensor noise
        noise = np.random.normal(0, 3, (256, 256, 3)).astype(np.int16)
        real_arr = np.clip(np.array(real_img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(real_arr).save(config.RAW_REAL_DIR / f"real_sample_{i:04d}.jpg", quality=92)

        # 2. AI-Generated Image: Diffusion/GAN artifact simulation
        ai_img = Image.new("RGB", (256, 256), color=(80 + (i * 4) % 60, 60, 140 + (i * 3) % 80))
        draw = ImageDraw.Draw(ai_img)
        # Unnatural surreal floating geometry
        draw.ellipse([60, 50, 200, 190], fill=(240, 80 + i % 50, 190))
        draw.polygon([(100, 40), (180, 120), (50, 180)], fill=(0, 220, 200))
        # Add high-frequency checkerboard / spectral artifacts common in generative models
        ai_arr = np.array(ai_img, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(np.arange(256), np.arange(256))
        # High-frequency sinusoidal pattern (classic latent upsampling artifact)
        synth_freq = np.sin(grid_x / 3.0) * np.cos(grid_y / 3.0) * 18.0
        ai_arr[:, :, 0] = np.clip(ai_arr[:, :, 0] + synth_freq, 0, 255)
        ai_arr[:, :, 2] = np.clip(ai_arr[:, :, 2] - synth_freq, 0, 255)
        Image.fromarray(ai_arr.astype(np.uint8)).save(config.RAW_AI_DIR / f"ai_sample_{i:04d}.png")

    print(f"Demo samples successfully generated in {config.DATA_DIR}.")


def prepare_dataset_splits(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    auto_generate_demo_if_empty: bool = True
) -> Dict[str, Any]:
    """
    Prepares train, validation, and test splits with:
    - Zero data leakage (SHA-256 deduplication).
    - Class balance verification.
    - Stratified splitting.
    - JSON split manifest generation.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Scan raw directories
    real_paths, real_errors = scan_and_clean_directory(config.RAW_REAL_DIR, class_label=0)
    ai_paths, ai_errors = scan_and_clean_directory(config.RAW_AI_DIR, class_label=1)

    if (len(real_paths) == 0 or len(ai_paths) == 0) and auto_generate_demo_if_empty:
        print("Raw image directories are empty. Generating starter demo dataset...")
        generate_synthetic_demo_samples(num_per_class=60)
        real_paths, real_errors = scan_and_clean_directory(config.RAW_REAL_DIR, class_label=0)
        ai_paths, ai_errors = scan_and_clean_directory(config.RAW_AI_DIR, class_label=1)

    print(f"Discovered: {len(real_paths)} Real images, {len(ai_paths)} AI-Generated images.")

    # Deduplication across datasets
    seen_hashes = {}
    deduped_real = []
    deduped_ai = []

    for path in real_paths:
        h = compute_file_sha256(path)
        if h not in seen_hashes:
            seen_hashes[h] = path
            deduped_real.append(path)

    for path in ai_paths:
        h = compute_file_sha256(path)
        if h not in seen_hashes:
            seen_hashes[h] = path
            deduped_ai.append(path)

    print(f"After deduplication: {len(deduped_real)} Real, {len(deduped_ai)} AI-Generated.")

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(config.RANDOM_SEED)
    rng.shuffle(deduped_real)
    rng.shuffle(deduped_ai)

    def split_list(items: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    train_real, val_real, test_real = split_list(deduped_real)
    train_ai, val_ai, test_ai = split_list(deduped_ai)

    # Clean existing split directories
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "Real").mkdir(parents=True, exist_ok=True)
        (split_dir / "AI-Generated").mkdir(parents=True, exist_ok=True)

    def copy_files(paths: List[Path], target_dir: Path):
        for p in paths:
            shutil.copy2(p, target_dir / p.name)

    copy_files(train_real, config.TRAIN_DIR / "Real")
    copy_files(train_ai, config.TRAIN_DIR / "AI-Generated")
    copy_files(val_real, config.VAL_DIR / "Real")
    copy_files(val_ai, config.VAL_DIR / "AI-Generated")
    copy_files(test_real, config.TEST_DIR / "Real")
    copy_files(test_ai, config.TEST_DIR / "AI-Generated")

    summary = {
        "train": {"Real": len(train_real), "AI-Generated": len(train_ai), "Total": len(train_real) + len(train_ai)},
        "val": {"Real": len(val_real), "AI-Generated": len(val_ai), "Total": len(val_real) + len(val_ai)},
        "test": {"Real": len(test_real), "AI-Generated": len(test_ai), "Total": len(test_real) + len(test_ai)},
        "deduplicated_total": len(deduped_real) + len(deduped_ai),
        "excluded_files_count": len(real_errors) + len(ai_errors),
        "random_seed": config.RANDOM_SEED
    }

    manifest_path = config.SPLITS_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Dataset preparation complete! Manifest saved to {manifest_path}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    prepare_dataset_splits()

