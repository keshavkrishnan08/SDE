"""Figure 6: Economic value bar chart."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_economic_value(
    annual_costs: dict[str, float],
    output_path: str = "outputs/figures/fig6_economic_value.pdf",
) -> None:
    """Bar chart: annual reserve cost ($/GW) for each forecasting model.

    Args:
        annual_costs: Dict mapping model_name -> annual cost per GW.
        output_path: Where to save.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(annual_costs.keys())
    costs = list(annual_costs.values())

    colors = []
    for m in models:
        if "SolarSDE" in m:
            colors.append("#e74c3c")
        elif "Persistence" in m:
            colors.append("#95a5a6")
        else:
            colors.append("#3498db")

    bars = ax.bar(models, costs, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate savings vs. persistence
    if "Persistence" in annual_costs and "SolarSDE (ours)" in annual_costs:
        savings = annual_costs["Persistence"] - annual_costs["SolarSDE (ours)"]
        ax.annotate(
            f"Savings: ${savings/1e6:.1f}M/GW/yr",
            xy=(models.index("SolarSDE (ours)"), annual_costs["SolarSDE (ours)"]),
            xytext=(0, 20), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        )

    ax.set_ylabel("Annual Reserve Cost ($/GW)", fontsize=12)
    ax.set_title("Economic Value of Probabilistic Forecasting", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
