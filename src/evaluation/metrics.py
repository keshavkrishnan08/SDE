"""Probabilistic and deterministic forecast evaluation metrics.

Implements CRPS, PICP, PINAW, RMSE, MAE, Ramp Score, and Skill Score.
"""

import numpy as np
from scipy import stats


def crps_empirical(y_true: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    """Compute Continuous Ranked Probability Score from empirical samples.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M) where M is number of MC samples.

    Returns:
        CRPS for each observation, shape (N,).
    """
    N, M = y_samples.shape

    # E|X - y|: mean absolute difference between samples and observation
    term1 = np.mean(np.abs(y_samples - y_true[:, np.newaxis]), axis=1)

    # E|X - X'|: mean absolute difference between pairs of samples
    # Efficient computation using sorted samples
    y_sorted = np.sort(y_samples, axis=1)
    # Use the formula: E|X-X'| = (2/M^2) * sum_{i=1}^{M} (2i - M - 1) * x_{(i)}
    weights = 2 * np.arange(1, M + 1) - M - 1
    term2 = np.sum(weights[np.newaxis, :] * y_sorted, axis=1) / (M * M)

    return term1 - term2


def picp(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    alpha: float = 0.90,
) -> float:
    """Prediction Interval Coverage Probability.

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        alpha: Confidence level (e.g., 0.90 for 90% PI).

    Returns:
        Coverage fraction (target: alpha).
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = np.quantile(y_samples, lower_q, axis=1)
    upper = np.quantile(y_samples, upper_q, axis=1)
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    return float(covered)


def pinaw(
    y_samples: np.ndarray,
    y_range: float,
    alpha: float = 0.90,
) -> float:
    """Prediction Interval Normalized Average Width.

    Args:
        y_samples: Forecast samples, shape (N, M).
        y_range: Range of observed values for normalization.
        alpha: Confidence level.

    Returns:
        Normalized average width (lower = sharper).
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = np.quantile(y_samples, lower_q, axis=1)
    upper = np.quantile(y_samples, upper_q, axis=1)
    avg_width = np.mean(upper - lower)
    return float(avg_width / y_range) if y_range > 0 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def ramp_score(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    is_ramp: np.ndarray,
) -> float:
    """CRPS conditional on ramp events.

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        is_ramp: Boolean ramp event mask, shape (N,).

    Returns:
        Mean CRPS during ramp events.
    """
    if is_ramp.sum() == 0:
        return 0.0
    crps_all = crps_empirical(y_true, y_samples)
    return float(crps_all[is_ramp].mean())


def skill_score(crps_model: float, crps_reference: float) -> float:
    """Forecast skill score: 1 - CRPS_model / CRPS_reference."""
    if crps_reference == 0:
        return 0.0
    return 1.0 - crps_model / crps_reference


def compute_all_metrics(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    crps_persistence: float | None = None,
    is_ramp: np.ndarray | None = None,
    pi_level: float = 0.90,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        crps_persistence: CRPS of persistence baseline for skill score.
        is_ramp: Boolean ramp event mask.
        pi_level: Prediction interval confidence level.

    Returns:
        Dictionary of all metrics.
    """
    # Point forecast = median of samples
    y_median = np.median(y_samples, axis=1)
    y_range = y_true.max() - y_true.min()

    crps_vals = crps_empirical(y_true, y_samples)
    mean_crps = float(crps_vals.mean())

    results = {
        "crps": mean_crps,
        "picp": picp(y_true, y_samples, pi_level),
        "pinaw": pinaw(y_samples, y_range, pi_level),
        "rmse": rmse(y_true, y_median),
        "mae": mae(y_true, y_median),
    }

    if crps_persistence is not None:
        results["skill_score"] = skill_score(mean_crps, crps_persistence)

    if is_ramp is not None:
        results["ramp_crps"] = ramp_score(y_true, y_samples, is_ramp)

    return results
