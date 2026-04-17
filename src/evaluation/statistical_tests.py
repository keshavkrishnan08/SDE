"""Statistical significance tests for forecast comparison."""

import numpy as np
from scipy import stats


def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_baseline: np.ndarray,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Uses Newey-West HAC standard errors to account for serial correlation.

    Args:
        errors_model: Squared errors from the model, shape (N,).
        errors_baseline: Squared errors from the baseline, shape (N,).
        horizon: Forecast horizon (for HAC bandwidth selection).
        alternative: "two-sided", "less", or "greater".

    Returns:
        Dict with 'statistic', 'p_value'.
    """
    d = errors_baseline - errors_model  # Positive means model is better
    n = len(d)
    d_mean = d.mean()

    # Newey-West HAC variance estimate
    bandwidth = max(1, horizon - 1)
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0
    for k in range(1, bandwidth + 1):
        weight = 1 - k / (bandwidth + 1)
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * weight * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return {"statistic": 0.0, "p_value": 1.0}

    dm_stat = d_mean / np.sqrt(var_d)

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:
        p_value = 1 - stats.norm.cdf(dm_stat)

    return {"statistic": float(dm_stat), "p_value": float(p_value)}


def bootstrap_confidence_interval(
    metric_fn,
    y_true: np.ndarray,
    y_samples: np.ndarray,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function(y_true, y_samples) -> scalar metric.
        y_true: Observed values, shape (N,).
        y_samples: Forecast samples, shape (N, M).
        num_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Dict with 'mean', 'lower', 'upper', 'std'.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrap_metrics = []

    for _ in range(num_bootstrap):
        idx = rng.randint(0, n, size=n)
        metric = metric_fn(y_true[idx], y_samples[idx])
        bootstrap_metrics.append(metric)

    bootstrap_metrics = np.array(bootstrap_metrics)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return {
        "mean": float(bootstrap_metrics.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "std": float(bootstrap_metrics.std()),
    }


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[dict[str, float | bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Family-wise error rate.

    Returns:
        List of dicts with 'original_p', 'adjusted_p', 'significant'.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    results = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(p * (n - rank), 1.0)
        results[orig_idx] = {
            "original_p": p,
            "adjusted_p": adjusted_p,
            "significant": adjusted_p < alpha,
        }

    # Enforce monotonicity
    prev_max = 0
    for rank, (orig_idx, _) in enumerate(indexed):
        results[orig_idx]["adjusted_p"] = max(results[orig_idx]["adjusted_p"], prev_max)
        prev_max = results[orig_idx]["adjusted_p"]
        results[orig_idx]["significant"] = results[orig_idx]["adjusted_p"] < alpha

    return results
