"""
Inference Engine for AI-Generated Image Detection.

Provides:
- Checkpoint loading and validation for both EfficientNet-B0 and Custom CNN.
- High-level inference API with end-to-end preprocessing, inference, and Grad-CAM generation.
- Strict error handling for corrupt files, missing weights, or hardware failures.
- Non-mock behavior: clearly flags when models are untrained.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F

import config
from models.cnn_model import build_custom_cnn, CustomCNN
from models.efficientnet_model import build_efficientnet, EfficientNetB0Detector
from utils.preprocessing import validate_image_file, load_and_preprocess_image, get_inference_transforms
from explainability.gradcam import GradCAM, generate_gradcam_explanation


class InferenceEngine:
    """
    Unified inference manager for Custom CNN and EfficientNet-B0 architectures.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.models: Dict[str, torch.nn.Module] = {}
        self.transforms = get_inference_transforms()

    def is_model_trained(self, model_name: str) -> bool:
        """Checks if checkpoint exists on disk for the given model architecture."""
        if model_name.lower() in ["efficientnet", "efficientnet-b0", "efficientnet_b0"]:
            return config.EFFICIENTNET_WEIGHTS_PATH.exists()
        elif model_name.lower() in ["cnn", "custom_cnn", "custom cnn"]:
            return config.CNN_WEIGHTS_PATH.exists()
        return False

    def get_model_status(self) -> Dict[str, Any]:
        """Returns the training status and file details for all supported models."""
        return {
            "EfficientNet-B0": {
                "trained": config.EFFICIENTNET_WEIGHTS_PATH.exists(),
                "path": str(config.EFFICIENTNET_WEIGHTS_PATH),
                "size_mb": round(config.EFFICIENTNET_WEIGHTS_PATH.stat().st_size / (1024 * 1024), 2)
                if config.EFFICIENTNET_WEIGHTS_PATH.exists() else 0.0
            },
            "Custom CNN": {
                "trained": config.CNN_WEIGHTS_PATH.exists(),
                "path": str(config.CNN_WEIGHTS_PATH),
                "size_mb": round(config.CNN_WEIGHTS_PATH.stat().st_size / (1024 * 1024), 2)
                if config.CNN_WEIGHTS_PATH.exists() else 0.0
            }
        }

    def load_model(self, model_name: str, force_reload: bool = False) -> torch.nn.Module:
        """
        Loads a trained model checkpoint from disk.
        Raises FileNotFoundError if weights do not exist.
        """
        normalized_name = "efficientnet" if "efficient" in model_name.lower() else "cnn"

        if not force_reload and normalized_name in self.models:
            return self.models[normalized_name]

        if normalized_name == "efficientnet":
            weights_path = config.EFFICIENTNET_WEIGHTS_PATH
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"EfficientNet-B0 weights not found at: {weights_path}. "
                    "Please run training/train_efficientnet.py first."
                )

            model = build_efficientnet(num_classes=config.NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(weights_path, map_location=self.device)
            # Support both direct state_dict and dict with metadata
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)

        else:
            weights_path = config.CNN_WEIGHTS_PATH
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Custom CNN weights not found at: {weights_path}. "
                    "Please run training/train_cnn.py first."
                )

            model = build_custom_cnn(num_classes=config.NUM_CLASSES)
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.models[normalized_name] = model
        return model

    def analyze_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        model_name: str = "EfficientNet-B0",
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Performs end-to-end detection and optional Grad-CAM explainability analysis.

        Returns:
            Dict containing:
                - success: bool
                - error: str (if any)
                - predicted_class: str ('Real' or 'AI-Generated')
                - confidence: float [0.0 - 1.0]
                - probabilities: Dict[str, float]
                - original_image: PIL.Image
                - gradcam_overlay: PIL.Image (if generate_cam is True)
                - gradcam_heatmap: np.ndarray (if generate_cam is True)
                - explanation: Dict[str, str]
                - model_used: str
        """
        # 1. Validation
        if not isinstance(image_input, Image.Image):
            is_valid, err_msg = validate_image_file(image_input)
            if not is_valid:
                return {"success": False, "error": f"Image validation failed: {err_msg}"}
            try:
                pil_img = load_and_preprocess_image(image_input)
            except Exception as e:
                return {"success": False, "error": f"Error decoding image: {str(e)}"}
        else:
            pil_img = image_input.convert("RGB")

        # 2. Check model availability
        normalized_name = "efficientnet" if "efficient" in model_name.lower() else "cnn"
        if not self.is_model_trained(normalized_name):
            return {
                "success": False,
                "error": f"Model weights for '{model_name}' were not found. Please train the model before running inference.",
                "model_status": self.get_model_status()
            }

        try:
            model = self.load_model(normalized_name)
        except Exception as e:
            return {"success": False, "error": f"Failed to load model weights: {str(e)}"}

        # 3. Preprocess Tensor
        tensor_img = self.transforms(pil_img).unsqueeze(0).to(self.device)

        # 4. Inference & Grad-CAM
        try:
            target_layer = model.get_gradcam_target_layer()

            if generate_cam:
                gradcam = GradCAM(model=model, target_layer=target_layer, device=self.device)
                heatmap_2d, pred_idx, confidence, probs_tensor = gradcam.generate_heatmap(tensor_img)
                overlay_img = gradcam.overlay_heatmap_on_image(pil_img, heatmap_2d)
                gradcam.remove_hooks()
            else:
                with torch.no_grad():
                    logits = model(tensor_img)
                    probs_tensor = F.softmax(logits, dim=1).squeeze(0)
                    pred_idx = int(torch.argmax(probs_tensor).item())
                    confidence = float(probs_tensor[pred_idx].item())
                    heatmap_2d = None
                    overlay_img = None

            predicted_class = config.LABEL_TO_NAME[pred_idx]
            probs_dict = {
                config.LABEL_TO_NAME[i]: float(probs_tensor[i].item())
                for i in range(len(config.CLASS_NAMES))
            }

            explanation = generate_gradcam_explanation(
                predicted_class=predicted_class,
                confidence=confidence,
                model_name=model_name
            )

            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probs_dict,
                "original_image": pil_img,
                "gradcam_overlay": overlay_img,
                "gradcam_heatmap": heatmap_2d,
                "explanation": explanation,
                "model_used": model_name,
                "device_used": str(self.device)
            }

        except Exception as e:
            return {"success": False, "error": f"Inference pipeline execution error: {str(e)}"}

