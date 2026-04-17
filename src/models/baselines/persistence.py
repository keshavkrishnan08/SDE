"""Persistence baseline: GHI_{t+h} = GHI_t.

Probabilistic version adds Gaussian noise calibrated from training set residuals.
"""

import numpy as np
import torch


class PersistenceModel:
    """Naive persistence baseline."""

    def __init__(self):
        self.residual_std = None

    def fit(self, ghi_train: np.ndarray, horizons: list[int]) -> None:
        """Calibrate noise from training set residual statistics."""
        self.residual_std = {}
        for h in horizons:
            residuals = ghi_train[h:] - ghi_train[:-h]
            self.residual_std[h] = float(np.std(residuals))

    def predict(
        self,
        ghi_current: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Generate probabilistic persistence forecast.

        Args:
            ghi_current: Current GHI values, shape (B,).
            horizon: Forecast horizon in steps.
            num_samples: Number of MC samples.

        Returns:
            Forecast samples, shape (B, num_samples).
        """
        B = len(ghi_current)
        point_forecast = ghi_current[:, np.newaxis].repeat(num_samples, axis=1)

        if self.residual_std is not None and horizon in self.residual_std:
            noise = np.random.randn(B, num_samples) * self.residual_std[horizon]
            point_forecast = point_forecast + noise

        return np.clip(point_forecast, 0, None)

    def predict_point(self, ghi_current: np.ndarray) -> np.ndarray:
        """Point forecast: persist current value."""
        return ghi_current.copy()
