"""Deep Ensemble of LSTMs for probabilistic forecasting."""

import torch
import torch.nn as nn
from src.models.baselines.lstm import LSTMForecaster


class DeepEnsemble(nn.Module):
    """Ensemble of independently trained LSTMs.

    Each member is initialized with a different random seed and trained independently.
    The ensemble's predictions form a probabilistic forecast.
    """

    def __init__(
        self,
        num_members: int = 5,
        input_dim: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_horizons: int = 7,
    ):
        super().__init__()
        self.members = nn.ModuleList([
            LSTMForecaster(input_dim, hidden_size, num_layers, num_horizons)
            for _ in range(num_members)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean prediction across ensemble."""
        preds = torch.stack([m(x) for m in self.members], dim=1)
        return preds.mean(dim=1)

    @torch.no_grad()
    def predict_probabilistic(self, x: torch.Tensor) -> torch.Tensor:
        """Return all ensemble member predictions.

        Returns:
            Shape (B, num_members, num_horizons).
        """
        self.eval()
        return torch.stack([m(x) for m in self.members], dim=1)
