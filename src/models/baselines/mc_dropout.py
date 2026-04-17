"""LSTM + Monte Carlo Dropout for probabilistic forecasting."""

import torch
import torch.nn as nn


class MCDropoutLSTM(nn.Module):
    """LSTM with MC Dropout for uncertainty quantification.

    At inference, run multiple forward passes with dropout enabled
    to produce an ensemble of predictions.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_horizons: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass (dropout active in train mode)."""
        output, (h_n, _) = self.lstm(x)
        h_final = self.drop(h_n[-1])
        return self.fc(h_final)

    @torch.no_grad()
    def predict_probabilistic(
        self,
        x: torch.Tensor,
        num_forward_passes: int = 100,
    ) -> torch.Tensor:
        """Run MC Dropout inference.

        Args:
            x: Input sequence, shape (B, seq_len, input_dim).
            num_forward_passes: Number of stochastic forward passes.

        Returns:
            Ensemble predictions, shape (B, num_forward_passes, num_horizons).
        """
        self.train()  # Keep dropout active
        predictions = []
        for _ in range(num_forward_passes):
            pred = self.forward(x)
            predictions.append(pred)
        self.eval()
        return torch.stack(predictions, dim=1)
