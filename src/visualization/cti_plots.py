"""Figure 3: CTI analysis plots."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_cti_scatter(
    cti: np.ndarray,
    ghi_variability: np.ndarray,
    output_path: str = "outputs/figures/fig3a_cti_scatter.pdf",
) -> None:
    """Scatter plot: CTI vs. observed irradiance variability."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Subsample for readability
    n = min(5000, len(cti))
    idx = np.random.choice(len(cti), n, replace=False)

    ax.scatter(cti[idx], ghi_variability[idx], alpha=0.3, s=5, c="#3498db")
    ax.set_xlabel("Cloud Turbulence Index (CTI)", fontsize=12)
    ax.set_ylabel("GHI Variability (W/m² std, 1-min)", fontsize=12)
    ax.set_title("CTI vs. Irradiance Variability", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_crps_by_cti_quartile(
    cti_bin_means: np.ndarray,
    crps_bin_means: np.ndarray,
    output_path: str = "outputs/figures/fig3b_crps_by_cti.pdf",
) -> None:
    """Bar plot: CRPS by CTI quartile."""
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = [f"Q{i+1}\n({m:.3f})" for i, m in enumerate(cti_bin_means)]
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(labels)))

    ax.bar(labels, crps_bin_means, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("CTI Quartile (mean value)", fontsize=12)
    ax.set_ylabel("Mean CRPS (W/m²)", fontsize=12)
    ax.set_title("Forecast Error by Cloud Turbulence", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cti_regime_distributions(
    ghi_by_regime: dict[str, np.ndarray],
    output_path: str = "outputs/figures/fig3c_cti_regimes.pdf",
) -> None:
    """Show irradiance distribution for each CTI regime."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    regime_names = ["Clear", "Thin Cloud", "Broken Cloud", "Overcast"]
    colors = ["#f39c12", "#e74c3c", "#9b59b6", "#3498db"]

    for i, (ax, name) in enumerate(zip(axes, regime_names)):
        key = list(ghi_by_regime.keys())[i] if i < len(ghi_by_regime) else None
        if key is not None:
            data = ghi_by_regime[key]
            ax.hist(data, bins=50, color=colors[i], alpha=0.7, density=True)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("GHI (W/m²)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Density", fontsize=10)

    plt.suptitle("Irradiance Distribution by CTI Regime", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
