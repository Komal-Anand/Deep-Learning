"""
Baseline Custom Convolutional Neural Network (CNN) for AI-Generated Image Detection.

Architecture Overview:
- 4 Feature Extraction Blocks:
    * Conv2D (3x3 kernel, padding=1)
    * Batch Normalization
    * ReLU Activation
    * Max Pooling (2x2)
    * Dropout (regularization)
- Adaptive Average Pooling
- Dense Classification Head with Dropout
- Target layer for Grad-CAM: final convolutional layer (conv4)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class CustomCNN(nn.Module):
    """
    Custom 4-layer Convolutional Neural Network baseline for binary classification
    (Real vs AI-Generated).
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(CustomCNN, self).__init__()

        # Block 1: 3 -> 32 channels (224x224 -> 112x112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )

        # Block 2: 32 -> 64 channels (112x112 -> 56x56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )

        # Block 3: 64 -> 128 channels (56x56 -> 28x28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )

        # Block 4: 128 -> 256 channels (28x28 -> 14x14) - Target for Grad-CAM
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )

        # Global Spatial Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CustomCNN.
        Returns raw unnormalized logits for [Real, AI-Generated].
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the final convolutional layer of block 4 for Grad-CAM feature map extraction.
        """
        return self.block4[0]

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates total and trainable parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def build_custom_cnn(num_classes: int = 2) -> CustomCNN:
    """Factory function to instantiate the Custom CNN baseline."""
    return CustomCNN(num_classes=num_classes)

