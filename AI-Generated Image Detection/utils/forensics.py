"""
Advanced Forensic Analysis Utilities:
1. 2D Fast Fourier Transform (FFT) Magnitude Spectrum Analysis.
2. Error Level Analysis (ELA) for compression artifact detection.
3. Metadata & AI Generation Prompt Inspector.
4. Forensic Audit Certificate Generator.
"""

import io
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_fft_spectrum(pil_image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """
    Computes the 2D Fast Fourier Transform (FFT) log magnitude spectrum.
    Authentic camera images exhibit smooth radial decay (1/f power law).
    Generative models (GANs, Diffusion) exhibit high-frequency checkerboard peaks
    and azimuthal frequency spikes due to upsampling / deconvolution layers.
    """
    # Convert to grayscale
    gray = np.array(pil_image.convert("L"), dtype=np.float32)

    # 2D Fast Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    # Compute Log Magnitude Spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1.0)

    # Normalize to [0, 255] uint8
    norm_spectrum = magnitude_spectrum - np.min(magnitude_spectrum)
    max_val = np.max(norm_spectrum)
    if max_val > 0:
        norm_spectrum = (norm_spectrum / max_val) * 255.0
    norm_spectrum = np.uint8(norm_spectrum)

    # Apply pseudo-color map (Inferno or Magma style) using matplotlib
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(norm_spectrum, cmap="inferno")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    spectrum_img = Image.open(buf).convert("RGB")

    return spectrum_img, magnitude_spectrum


def compute_error_level_analysis(pil_image: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Performs Error Level Analysis (ELA).
    Saves image at a known lossy compression quality and calculates absolute
    pixel differences with original.
    - Uniform noise = consistent camera compression.
    - Localized hyper-intense regions = synthetic generation or digital splicing.
    """
    rgb_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Compute absolute difference
    diff = ImageChops.difference(rgb_image, resaved)

    # Amplify difference for human visualization
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff if max_diff < 255 else 1.0

    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(min(scale, scale_factor))
    return ela_image


def extract_image_metadata(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Extracts forensic metadata:
    - Camera EXIF tags (Make, Model, Software, Lens, Exposure)
    - AI Generation prompts embedded in PNG text chunks (Stable Diffusion, Midjourney, ComfyUI)
    """
    metadata = {
        "format": pil_image.format or "Unknown",
        "dimensions": f"{pil_image.width} x {pil_image.height}",
        "mode": pil_image.mode,
        "is_ai_metadata_detected": False,
        "ai_prompt_info": {},
        "camera_exif": {}
    }

    # 1. Check for AI Generation Prompt chunks in PNG/WebP info
    info = pil_image.info or {}
    ai_keys = ["parameters", "prompt", "workflow", "Comment", "Generation details"]

    for k in ai_keys:
        if k in info:
            metadata["is_ai_metadata_detected"] = True
            metadata["ai_prompt_info"][k] = str(info[k])[:800]  # Cap length for safety

    # 2. Check EXIF tags for genuine camera metadata
    try:
        exif_data = pil_image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in ["Make", "Model", "Software", "DateTime", "ExposureTime", "FNumber", "ISOSpeedRatings"]:
                    metadata["camera_exif"][tag_name] = str(value)
    except Exception:
        pass

    return metadata


def generate_forensic_certificate(
    result: Dict[str, Any],
    file_name: str,
    file_bytes: bytes
) -> str:
    """
    Generates a structured, formal Digital Forensics Audit Report in Markdown format.
    """
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    pred_class = result.get("predicted_class", "Unknown")
    conf = result.get("confidence", 0.0)
    model = result.get("model_used", "EfficientNet-B0")
    probs = result.get("probabilities", {})

    prob_lines = "\n".join([f"  - **{k}**: {v*100:.2f}%" for k, v in probs.items()])

    report = f"""# DIGITAL FORENSIC IMAGE AUTHENTICITY CERTIFICATE
**Document Ref:** AIS-{sha256[:12].upper()}  
**Issuing Authority:** AI Image Sentinel • Forensic Computer Vision Laboratory  
**Timestamp:** {now}  

---

### 1. Evidence File Identification
- **File Name:** `{file_name}`
- **SHA-256 Hash:** `{sha256}`
- **Dimensions:** {result.get('original_image', Image.new('RGB', (1,1))).size[0]} x {result.get('original_image', Image.new('RGB', (1,1))).size[1]} px
- **Input Channels:** 3 (RGB Standardized)

---

### 2. Deep Learning Classification Verdict
- **Classification Result:** **{pred_class.upper()}**
- **Confidence Rating:** **{conf*100:.2f}%**
- **Inference Engine:** {model}
- **Softmax Probability Distribution:**
{prob_lines}

---

### 3. Explainable AI (Grad-CAM) Attribution
- **Target Layer:** Final Deep Convolutional Receptive Fields
- **Summary:** {result.get('explanation', {}).get('technical_summary', 'N/A')}

---

### 4. Forensic Compliance & Disclaimer
This certificate represents probabilistic mathematical classification derived from deep convolutional 
feature representations. While high confidence indicates strong pattern correlation with known 
synthetic artifacts, it should be used in conjunction with cryptographic watermarks and human forensic review.

*Generated automatically by AI Image Sentinel Final-Year B.Tech Research Platform.*
"""
    return report

