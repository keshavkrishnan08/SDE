"""Figure 2: Main results — CRPS vs. forecast horizon for all models."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_crps_vs_horizon(
    results: dict[str, dict[int, float]],
    horizons_minutes: list[float],
    output_path: str = "outputs/figures/fig2_crps_vs_horizon.pdf",
) -> None:
    """Plot CRPS vs. forecast horizon for all models.

    Args:
        results: Dict mapping model_name -> {horizon_steps: crps_value}.
        horizons_minutes: Horizon values in minutes for x-axis.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Style for each model
    styles = {
        "SolarSDE (ours)": {"color": "#e74c3c", "linewidth": 2.5, "marker": "o", "zorder": 10},
        "Persistence": {"color": "#95a5a6", "linewidth": 1, "marker": "x", "linestyle": "--"},
        "Smart Persistence": {"color": "#7f8c8d", "linewidth": 1, "marker": "+", "linestyle": "--"},
        "LSTM": {"color": "#3498db", "linewidth": 1.5, "marker": "s"},
        "MC-Dropout": {"color": "#2980b9", "linewidth": 1.5, "marker": "D"},
        "Deep Ensemble": {"color": "#1abc9c", "linewidth": 1.5, "marker": "^"},
        "TimeGrad": {"color": "#9b59b6", "linewidth": 1.5, "marker": "v"},
        "CSDI": {"color": "#e67e22", "linewidth": 1.5, "marker": "<"},
        "CNN+Image": {"color": "#34495e", "linewidth": 1, "marker": ">", "linestyle": "-."},
    }

    for model_name, crps_by_horizon in results.items():
        horizons = sorted(crps_by_horizon.keys())
        crps_values = [crps_by_horizon[h] for h in horizons]
        style = styles.get(model_name, {"linewidth": 1})
        ax.plot(horizons_minutes[:len(crps_values)], crps_values, label=model_name, **style)

    ax.set_xlabel("Forecast Horizon (minutes)", fontsize=12)
    ax.set_ylabel("CRPS (W/m²)", fontsize=12)
    ax.set_title("Probabilistic Forecast Performance", fontsize=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(horizons_minutes) + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_skill_score_bars(
    skill_scores: dict[str, float],
    output_path: str = "outputs/figures/fig2b_skill_scores.pdf",
) -> None:
    """Bar chart of skill scores relative to persistence."""
    fig, ax = plt.subplots(figsize=(8, 4))

    models = list(skill_scores.keys())
    scores = list(skill_scores.values())
    colors = ["#e74c3c" if m == "SolarSDE (ours)" else "#3498db" for m in models]

    bars = ax.barh(models, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Skill Score (vs. Persistence)", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
