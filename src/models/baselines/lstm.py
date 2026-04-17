"""Deterministic LSTM baseline for irradiance forecasting."""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """2-layer LSTM for deterministic irradiance point forecasting.

    Input: sequence of (GHI, covariates) → Output: GHI at each forecast horizon.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_horizons: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence, shape (B, seq_len, input_dim).

        Returns:
            Predictions, shape (B, num_horizons).
        """
        output, (h_n, _) = self.lstm(x)
        # Use final hidden state
        h_final = h_n[-1]  # (B, hidden_size)
        return self.fc(h_final)
