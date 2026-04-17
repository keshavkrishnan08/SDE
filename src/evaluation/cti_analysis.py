"""CTI Analysis: validate that the learned CTI is physically meaningful."""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


def cti_cloud_cover_correlation(
    cti: np.ndarray,
    cloud_cover: np.ndarray,
) -> dict[str, float]:
    """Compute Spearman correlation between CTI and observed cloud cover.

    Gate G2: expects ρ > 0.5.
    """
    mask = np.isfinite(cti) & np.isfinite(cloud_cover)
    rho, p_value = stats.spearmanr(cti[mask], cloud_cover[mask])
    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def cti_irradiance_variability_correlation(
    cti: np.ndarray,
    ghi: np.ndarray,
    window: int = 6,
) -> dict[str, float]:
    """Compute correlation between CTI and rolling standard deviation of GHI.

    Window of 6 steps × 10s = 1 minute.
    """
    # Rolling std of GHI
    ghi_std = np.zeros_like(ghi)
    for i in range(window, len(ghi)):
        ghi_std[i] = np.std(ghi[i - window : i])

    mask = (cti > 0) & np.isfinite(ghi_std)
    rho, p_value = stats.spearmanr(cti[mask], ghi_std[mask])
    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def cti_forecast_error_bins(
    cti: np.ndarray,
    crps_values: np.ndarray,
    num_bins: int = 4,
) -> dict[str, np.ndarray]:
    """Compute CRPS as a function of CTI quantile bins.

    Expected: CRPS increases monotonically with CTI.
    """
    quantiles = np.quantile(cti[cti > 0], np.linspace(0, 1, num_bins + 1))
    bin_means = []
    bin_crps = []

    for i in range(num_bins):
        if i < num_bins - 1:
            mask = (cti >= quantiles[i]) & (cti < quantiles[i + 1])
        else:
            mask = (cti >= quantiles[i]) & (cti <= quantiles[i + 1])

        if mask.sum() > 0:
            bin_means.append(cti[mask].mean())
            bin_crps.append(crps_values[mask].mean())

    return {
        "cti_bin_means": np.array(bin_means),
        "crps_bin_means": np.array(bin_crps),
        "quantile_edges": quantiles,
    }


def cti_regime_clustering(
    cti: np.ndarray,
    ghi: np.ndarray,
    n_clusters: int = 4,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """K-means cluster CTI values into weather regimes.

    Expected regimes: clear, thin cloud, broken cloud, overcast.

    Returns:
        Dict with 'labels', 'centers', 'regime_stats' (mean/std GHI per cluster).
    """
    valid = cti > 0
    cti_valid = cti[valid].reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels_valid = kmeans.fit_predict(cti_valid)

    # Full labels (invalid timesteps get -1)
    labels = np.full(len(cti), -1, dtype=int)
    labels[valid] = labels_valid

    # Regime statistics
    regime_stats = []
    for c in range(n_clusters):
        mask = labels == c
        regime_stats.append({
            "cluster": c,
            "cti_mean": float(cti[mask].mean()),
            "cti_std": float(cti[mask].std()),
            "ghi_mean": float(ghi[mask].mean()),
            "ghi_std": float(ghi[mask].std()),
            "count": int(mask.sum()),
        })

    # Sort by CTI mean (ascending: clear → overcast with turbulent broken in between)
    regime_stats.sort(key=lambda x: x["cti_mean"])

    return {
        "labels": labels,
        "centers": kmeans.cluster_centers_.flatten(),
        "regime_stats": regime_stats,
    }
