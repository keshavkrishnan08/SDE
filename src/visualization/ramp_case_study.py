"""Figure 4: Ramp event case study visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_ramp_case_study(
    timestamps: np.ndarray,
    ghi_actual: np.ndarray,
    ghi_median: np.ndarray,
    ghi_lower: np.ndarray,
    ghi_upper: np.ndarray,
    cti: np.ndarray,
    output_path: str = "outputs/figures/fig4_ramp_case_study.pdf",
) -> None:
    """Time series plot showing a ramp event with SolarSDE forecast.

    Shows actual GHI, median forecast, 90% PI, and CTI on secondary axis.
    Highlights: CTI spikes before the ramp, PI widens in advance.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # GHI actual
    ax1.plot(timestamps, ghi_actual, "k-", linewidth=1.5, label="Observed GHI", zorder=5)

    # Median forecast
    ax1.plot(timestamps, ghi_median, "b-", linewidth=1.2, label="SolarSDE Median", zorder=4)

    # 90% prediction interval
    ax1.fill_between(
        timestamps, ghi_lower, ghi_upper,
        alpha=0.25, color="#3498db", label="90% PI", zorder=3,
    )

    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("GHI (W/m²)", fontsize=12, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # CTI on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(timestamps, cti, "r-", linewidth=1, alpha=0.7, label="CTI")
    ax2.set_ylabel("Cloud Turbulence Index", fontsize=12, color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("Ramp Event: CTI-Conditioned Uncertainty", fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
