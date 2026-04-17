"""Calibration assessment: PIT histograms and reliability diagrams."""

import numpy as np
from typing import Optional


def probability_integral_transform(
    y_true: np.ndarray,
    y_samples: np.ndarray,
) -> np.ndarray:
    """Compute PIT values.

    PIT_i = F_hat(y_i) = fraction of samples <= y_i.
    Should be Uniform(0,1) if the model is calibrated.

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).

    Returns:
        PIT values, shape (N,).
    """
    N, M = y_samples.shape
    pit = np.mean(y_samples <= y_true[:, np.newaxis], axis=1)
    return pit


def reliability_data(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    levels: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Compute data for reliability diagram.

    For each nominal coverage level, compute the observed coverage.

    Args:
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        levels: Nominal coverage levels. Default: 10%, 20%, ..., 90%.

    Returns:
        Dict with 'nominal' and 'observed' arrays.
    """
    if levels is None:
        levels = np.arange(0.1, 1.0, 0.1)

    observed = []
    for level in levels:
        lower_q = (1 - level) / 2
        upper_q = 1 - lower_q
        lower = np.quantile(y_samples, lower_q, axis=1)
        upper = np.quantile(y_samples, upper_q, axis=1)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        observed.append(coverage)

    return {
        "nominal": levels,
        "observed": np.array(observed),
    }


def pit_histogram_data(
    pit_values: np.ndarray,
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute histogram data for PIT values.

    Returns:
        Dict with 'bin_edges', 'counts', and 'expected' (uniform).
    """
    counts, bin_edges = np.histogram(pit_values, bins=num_bins, range=(0, 1))
    expected = len(pit_values) / num_bins
    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "expected": expected,
    }


def conditional_reliability(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    condition_values: np.ndarray,
    num_bins: int = 4,
    levels: Optional[np.ndarray] = None,
) -> dict[str, list]:
    """Compute reliability diagrams stratified by a conditioning variable (e.g., CTI).

    Returns:
        Dict with 'bin_labels' and 'reliability' (list of reliability dicts per bin).
    """
    if levels is None:
        levels = np.arange(0.1, 1.0, 0.1)

    quantiles = np.quantile(condition_values, np.linspace(0, 1, num_bins + 1))
    results = {"bin_labels": [], "reliability": []}

    for i in range(num_bins):
        mask = (condition_values >= quantiles[i]) & (condition_values < quantiles[i + 1])
        if i == num_bins - 1:
            mask = (condition_values >= quantiles[i]) & (condition_values <= quantiles[i + 1])
        if mask.sum() == 0:
            continue

        label = f"Q{i+1} [{quantiles[i]:.3f}, {quantiles[i+1]:.3f}]"
        rel = reliability_data(y_true[mask], y_samples[mask], levels)
        results["bin_labels"].append(label)
        results["reliability"].append(rel)

    return results
