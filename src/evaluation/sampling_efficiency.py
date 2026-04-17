"""Sampling efficiency experiment: how many MC samples are needed for converged forecasts."""

import numpy as np
from src.evaluation.metrics import crps_empirical, picp


def evaluate_sample_convergence(
    y_true: np.ndarray,
    y_samples_full: np.ndarray,
    sample_counts: list[int] | None = None,
    pi_level: float = 0.90,
    seed: int = 42,
) -> dict[str, list]:
    """Evaluate metric convergence as a function of sample count N.

    Args:
        y_true: Observed values, shape (T,).
        y_samples_full: Full set of samples, shape (T, M_max).
        sample_counts: List of N values to test. Default: [10, 25, 50, 100, 200, 500, 1000].
        pi_level: Prediction interval confidence level.
        seed: Random seed for subsampling.

    Returns:
        Dict with 'n_samples', 'crps', 'picp', 'crps_relative'.
    """
    if sample_counts is None:
        sample_counts = [10, 25, 50, 100, 200, 500, 1000]

    rng = np.random.RandomState(seed)
    M_max = y_samples_full.shape[1]

    # Reference CRPS with all samples
    crps_ref = crps_empirical(y_true, y_samples_full).mean()

    results = {"n_samples": [], "crps": [], "picp": [], "crps_relative": []}

    for n in sample_counts:
        if n > M_max:
            continue

        # Randomly subsample
        idx = rng.choice(M_max, size=n, replace=False)
        y_sub = y_samples_full[:, idx]

        crps_val = crps_empirical(y_true, y_sub).mean()
        picp_val = picp(y_true, y_sub, pi_level)

        results["n_samples"].append(n)
        results["crps"].append(float(crps_val))
        results["picp"].append(float(picp_val))
        results["crps_relative"].append(float((crps_val - crps_ref) / crps_ref * 100))

    return results


def find_minimum_samples(
    convergence_results: dict[str, list],
    crps_threshold: float = 1.0,
) -> int | None:
    """Find minimum N for <threshold% CRPS degradation vs. full sample set.

    Args:
        convergence_results: Output from evaluate_sample_convergence.
        crps_threshold: Maximum acceptable CRPS degradation in percent.

    Returns:
        Minimum N, or None if no value meets the threshold.
    """
    for n, rel in zip(convergence_results["n_samples"], convergence_results["crps_relative"]):
        if abs(rel) < crps_threshold:
            return n
    return None
