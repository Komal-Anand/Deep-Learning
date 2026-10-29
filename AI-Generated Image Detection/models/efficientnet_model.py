"""
EfficientNet-B0 Transfer Learning Architecture for AI-Generated Image Detection.

Utilizes PyTorch's torchvision.models.efficientnet_b0 with ImageNet pretrained weights.
Supports:
1. Feature extraction mode (frozen backbone, trained classification head)
2. Fine-tuning mode (unfrozen top inverted residual blocks or entire network)
3. Direct target layer hook access for Grad-CAM explainability.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Dict, Any, Optional


class EfficientNetB0Detector(nn.Module):
    """
    EfficientNet-B0 transfer learning architecture for AI-Generated image detection.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone_initially: bool = False
    ):
        super(EfficientNetB0Detector, self).__init__()

        # Load weights
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = efficientnet_b0(weights=weights)

        # Retrieve input dimension of classifier (1280 for EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features

        # Replace classification head with custom binary classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

        if freeze_backbone_initially:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet-B0.
        Returns raw logits for [Real, AI-Generated].
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Freezes all parameters in the feature extraction backbone.
        Only the classifier parameters will receive gradient updates.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """
        Unfreezes all parameters across the entire network for full fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, num_blocks: int = 3) -> None:
        """
        Unfreezes the top `num_blocks` stages of the backbone while keeping early low-level
        feature extractors frozen to prevent catastrophic forgetting.
        EfficientNet-B0 features has 9 blocks (0 to 8).
        """
        # First freeze all
        self.freeze_backbone()
        # Unfreeze classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        total_blocks = len(self.backbone.features)
        start_idx = max(0, total_blocks - num_blocks)

        for i in range(start_idx, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the final convolutional block in EfficientNet-B0 features (features[8]),
        which contains the rich 1280-channel high-level semantic feature representations.
        """
        return self.backbone.features[-1]

    def count_parameters(self) -> Dict[str, int]:
        """
        Counts total and trainable parameters in the model.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def build_efficientnet(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone_initially: bool = False
) -> EfficientNetB0Detector:
    """Factory function for EfficientNetB0Detector."""
    return EfficientNetB0Detector(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone_initially=freeze_backbone_initially
    )

