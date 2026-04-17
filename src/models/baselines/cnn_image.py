"""CNN + Sky Image deterministic baseline using ResNet-18 encoder."""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNImageForecaster(nn.Module):
    """ResNet-18 encoder on sky image → Linear → GHI prediction.

    Deterministic point forecast baseline testing whether images alone
    contain predictive information.
    """

    def __init__(self, num_horizons: int = 7, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_horizons)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Sky image, shape (B, 3, 256, 256).

        Returns:
            GHI predictions, shape (B, num_horizons).
        """
        features = self.backbone(image).flatten(1)  # (B, 512)
        return self.fc(features)
