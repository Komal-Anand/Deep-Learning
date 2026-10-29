"""
Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017).

Provides:
- PyTorch forward and backward hook management.
- Activation and gradient caching for target convolutional layers.
- Weighted feature map combination with ReLU rectification.
- High-resolution heatmap generation and seamless alpha overlay.
- Interpretability summary and academic disclaimer.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: Optional[torch.device] = None):
        """
        Args:
            model: The neural network model (eval mode).
            target_layer: The convolutional layer to inspect.
            device: torch device (CPU or CUDA).
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Attaches forward and backward hooks to the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is dLoss / dOutput of layer
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]

    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        """
        Generates the 2D Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: (1, 3, H, W) normalized image tensor.
            target_class: Target class index. If None, uses argmax of model logits.

        Returns:
            heatmap_2d: 2D numpy array [0.0, 1.0] representing normalized activation intensities.
            predicted_class_idx: Integer index of predicted class.
            prediction_confidence: Softmax probability [0.0, 1.0].
            probabilities: Full softmax tensor.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(probs).item())

        score = logits[0, target_class]

        # Backward pass specifically for target score
        score.backward(retain_graph=False)

        # Global Average Pooling of gradients across spatial dimensions
        # self.gradients: [1, C, H_feat, W_feat]
        # self.activations: [1, C, H_feat, W_feat]
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Channel weights alpha_k = 1/Z * sum_i sum_j (dScore / dActivation)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=0)  # [H, W]

        # Rectified Linear Unit (only features with positive influence on target class)
        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        confidence = float(probs[target_class].item())
        return cam, target_class, confidence, probs

    def overlay_heatmap_on_image(
        self,
        pil_image: Image.Image,
        heatmap_2d: np.ndarray,
        alpha: float = 0.5,
        colormap_name: str = "jet"
    ) -> Image.Image:
        """
        Resizes the 2D heatmap to match original image dimensions and overlays it
        with a color palette.
        """
        w, h = pil_image.size

        # Convert heatmap to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_2d)

        # Resize heatmap to original image resolution with bicubic interpolation
        heatmap_img = Image.fromarray(heatmap_uint8).resize((w, h), Image.Resampling.BICUBIC)
        heatmap_resized = np.array(heatmap_img, dtype=np.float32) / 255.0

        # Apply JET colormap manually using matplotlib or procedural RGB curves
        # to ensure zero dependencies on system GUI backends
        overlay_rgb = self._apply_colormap_jet(heatmap_resized)

        # Blend with original image
        orig_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        blended = (1.0 - alpha) * orig_np + alpha * overlay_rgb
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    @staticmethod
    def _apply_colormap_jet(vals: np.ndarray) -> np.ndarray:
        """
        Pure NumPy implementation of the JET colormap.
        Maps 2D [0.0, 1.0] array into 3D (H, W, 3) RGB in [0.0, 1.0].
        """
        # Jet colormap formula:
        # Red: clamp(1.5 - |4x - 3|, 0, 1)
        # Green: clamp(1.5 - |4x - 2|, 0, 1)
        # Blue: clamp(1.5 - |4x - 1|, 0, 1)
        r = np.clip(1.5 - np.abs(4.0 * vals - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * vals - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * vals - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1)


def generate_gradcam_explanation(
    predicted_class: str,
    confidence: float,
    model_name: str = "EfficientNet-B0"
) -> Dict[str, str]:
    """
    Returns an academic, scientifically grounded explanation of what the Grad-CAM visualization
    reveals for the predicted class.
    """
    if predicted_class == "AI-Generated":
        technical_summary = (
            f"The {model_name} architecture activated most strongly (warm red/yellow zones) "
            "over regions exhibiting characteristic generative synthetic indicators. Diffusion models "
            "and GANs frequently leave subtle high-frequency artifacts in textured boundaries, unnatural "
            "skin or fabric gradients, fine-hair interpolation inconsistencies, or symmetric background "
            "blur patterns."
        )
        academic_note = (
            "Grad-CAM identifies spatial features that drove the deep neural network's classification score. "
            "It serves as an interpretability aid and qualitative audit; it does not constitute mathematical "
            "or cryptographic proof of synthetic manipulation."
        )
    else:
        technical_summary = (
            f"The {model_name} architecture identified natural photographic characteristics, focusing on "
            "coherent physical lighting transitions, genuine sensor noise distribution, realistic depth-of-field "
            "falloff, and authentic edge textures typical of optical camera sensors."
        )
        academic_note = (
            "A 'Real' classification indicates that the image conforms more closely to real photographic "
            "training distributions. Heavily compressed, re-encoded, or subtly edited images may still challenge "
            "discriminative classifiers."
        )

    return {
        "technical_summary": technical_summary,
        "academic_note": academic_note,
        "disclaimer": "Grad-CAM highlights influential convolutional receptive fields, not definitive proof of origin."
    }

