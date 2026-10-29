"""
Image Preprocessing, Augmentation Pipelines, and Dataset Management.

Provides:
- Strict file validation (size, MIME/format, image integrity).
- Standardized train, validation, and inference PyTorch transforms.
- Robustness perturbation utilities (JPEG compression, Gaussian blur, resizing).
- Fault-tolerant PyTorch Dataset class.
"""

import io
from pathlib import Path
from typing import Tuple, Optional, Callable, List
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config


def validate_image_file(file_path_or_bytes, max_size_bytes: int = config.MAX_IMAGE_SIZE_BYTES) -> Tuple[bool, str]:
    """
    Validates that the given file or bytes is a legitimate, uncorrupted, and safely sized image.
    Returns (is_valid, error_message).
    """
    try:
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                return False, f"File does not exist: {path.name}"
            
            ext = path.suffix.lower()
            if ext not in config.ALLOWED_EXTENSIONS:
                return False, f"Unsupported format '{ext}'. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
            
            size = path.stat().st_size
            if size > max_size_bytes:
                return False, f"File exceeds maximum size of {max_size_bytes / (1024*1024):.1f} MB."
            if size == 0:
                return False, "File is empty (0 bytes)."
                
            with open(path, "rb") as f:
                img_bytes = f.read()
        else:
            img_bytes = file_path_or_bytes
            if len(img_bytes) > max_size_bytes:
                return False, f"File exceeds maximum size of {max_size_bytes / (1024*1024):.1f} MB."
            if len(img_bytes) == 0:
                return False, "Uploaded file is empty (0 bytes)."

        # Test PIL opening and verification
        with Image.open(io.BytesIO(img_bytes)) as img:
            img.verify()

        # Reopen to ensure it can be decoded into pixels
        with Image.open(io.BytesIO(img_bytes)) as img:
            img.load()
            
        return True, ""
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"


def load_and_preprocess_image(file_path_or_bytes) -> Image.Image:
    """
    Safely loads an image and converts it to standard 3-channel RGB.
    """
    if isinstance(file_path_or_bytes, (str, Path)):
        img = Image.open(file_path_or_bytes)
    else:
        img = Image.open(io.BytesIO(file_path_or_bytes))

    # Convert grayscale, RGBA, palette, or CMYK to standard RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    return img


def get_train_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """
    Training pipeline with data augmentation:
    - Random Horizontal Flip (common real-world invariant)
    - Mild Color Jitter (robustness to lighting/tone shifts)
    - Random Affine/Rotation (slight orientation invariance)
    - Resizing to target size
    - Standard ImageNet Normalization
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


def get_val_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """Validation/Evaluation transform pipeline: deterministic resizing and normalization."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


def get_inference_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """Inference transform pipeline identical to validation."""
    return get_val_transforms(image_size=image_size)


# Robustness Perturbation Functions
def apply_jpeg_compression(image: Image.Image, quality: int = 50) -> Image.Image:
    """Simulates social media or network transfer JPEG compression artifacting."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Simulates camera defocus, motion blur, or anti-aliasing filter."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resize_downscale(image: Image.Image, scale: float = 0.5) -> Image.Image:
    """Simulates thumbnail downsampling and subsequent upscaling."""
    w, h = image.size
    downscaled = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    return downscaled.resize((w, h), Image.Resampling.BILINEAR)


class RobustImageDataset(Dataset):
    """
    Custom PyTorch Dataset with fault tolerance.
    Silently handles unreadable images by logging or falling back.
    """

    def __init__(self, samples: List[Tuple[Path, int]], transform: Optional[Callable] = None):
        """
        Args:
            samples: List of (file_path, class_label) tuples.
            transform: Optional torchvision transform callable.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = load_and_preprocess_image(img_path)
        except Exception:
            # Fallback to black dummy image if file became corrupted on disk
            image = Image.new("RGB", config.IMAGE_SIZE, (0, 0, 0))

        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        return tensor, label

