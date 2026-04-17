"""Smart Persistence baseline: persist clear-sky index, then multiply by clear-sky model.

k_{t+h} = k_t → GHI_{t+h} = k_t * GHI_clearsky_{t+h}
"""

import numpy as np


class SmartPersistenceModel:
    """Smart persistence using clear-sky index."""

    def __init__(self):
        self.residual_std = None

    def fit(
        self,
        kt_train: np.ndarray,
        ghi_clearsky_train: np.ndarray,
        ghi_train: np.ndarray,
        horizons: list[int],
    ) -> None:
        """Calibrate noise from training set clear-sky index residuals."""
        self.residual_std = {}
        for h in horizons:
            predicted = kt_train[:-h] * ghi_clearsky_train[h:]
            actual = ghi_train[h:]
            residuals = actual - predicted
            self.residual_std[h] = float(np.std(residuals))

    def predict(
        self,
        kt_current: np.ndarray,
        ghi_clearsky_future: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Generate probabilistic smart persistence forecast.

        Args:
            kt_current: Current clear-sky index, shape (B,).
            ghi_clearsky_future: Clear-sky GHI at future time, shape (B,).
            horizon: Forecast horizon in steps.
            num_samples: Number of MC samples.

        Returns:
            Forecast samples, shape (B, num_samples).
        """
        B = len(kt_current)
        point_forecast = (kt_current * ghi_clearsky_future)[:, np.newaxis]
        point_forecast = point_forecast.repeat(num_samples, axis=1)

        if self.residual_std is not None and horizon in self.residual_std:
            noise = np.random.randn(B, num_samples) * self.residual_std[horizon]
            point_forecast = point_forecast + noise

        return np.clip(point_forecast, 0, None)
