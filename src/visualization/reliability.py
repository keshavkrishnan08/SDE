"""Figure 5: Reliability diagram."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_reliability_diagram(
    reliability_data: dict[str, dict[str, np.ndarray]],
    output_path: str = "outputs/figures/fig5_reliability.pdf",
) -> None:
    """Reliability diagram: observed vs. nominal coverage.

    Args:
        reliability_data: Dict mapping model_name -> {'nominal': array, 'observed': array}.
        output_path: Where to save.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)

    colors = {
        "SolarSDE (ours)": "#e74c3c",
        "Deep Ensemble": "#1abc9c",
        "TimeGrad": "#9b59b6",
        "CSDI": "#e67e22",
        "MC-Dropout": "#2980b9",
    }

    for model_name, data in reliability_data.items():
        color = colors.get(model_name, "#333333")
        lw = 2.5 if "SolarSDE" in model_name else 1.5
        ax.plot(
            data["nominal"], data["observed"],
            "o-", color=color, linewidth=lw, markersize=5,
            label=model_name,
        )

    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Observed Coverage", fontsize=12)
    ax.set_title("Calibration Reliability Diagram", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pit_histogram(
    pit_values: np.ndarray,
    model_name: str = "SolarSDE",
    num_bins: int = 10,
    output_path: str = "outputs/figures/pit_histogram.pdf",
) -> None:
    """PIT histogram — should be uniform if calibrated."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(pit_values, bins=num_bins, range=(0, 1), density=True,
            color="#3498db", edgecolor="white", alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Uniform reference")

    ax.set_xlabel("PIT Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"PIT Histogram — {model_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
