"""Figure 1: Architecture overview diagram (programmatic generation)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")


def plot_architecture_diagram(
    output_path: str = "outputs/figures/fig1_architecture.pdf",
) -> None:
    """Generate a three-panel architecture overview diagram.

    Left: CS-VAE encoding sky images into latent space
    Center: Neural SDE with CTI-conditioned diffusion
    Right: Score-matching decoder producing irradiance distributions
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # === Panel 1: CS-VAE ===
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Cloud-State VAE", fontsize=13, fontweight="bold")

    # Sky image box
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 6), 3, 3, boxstyle="round,pad=0.2",
                 facecolor="#AED6F1", edgecolor="#2980B9", linewidth=2))
    ax.text(2, 7.5, "Sky Image\nx_t", ha="center", va="center", fontsize=10)

    # Encoder arrow
    ax.annotate("", xy=(5, 7.5), xytext=(3.7, 7.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
    ax.text(4.3, 8, "E_φ", ha="center", fontsize=9, style="italic")

    # Latent space
    ax.add_patch(mpatches.Ellipse((6.5, 7.5), 2.5, 2, facecolor="#FADBD8",
                 edgecolor="#E74C3C", linewidth=2))
    ax.text(6.5, 7.5, "z_t ∈ ℝ^d", ha="center", va="center", fontsize=10)

    # Decoder arrow
    ax.annotate("", xy=(6.5, 5.5), xytext=(6.5, 6.3),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
    ax.text(7.2, 5.9, "D_ψ", ha="center", fontsize=9, style="italic")

    # Reconstruction
    ax.add_patch(mpatches.FancyBboxPatch((5, 3.5), 3, 1.5, boxstyle="round,pad=0.2",
                 facecolor="#AED6F1", edgecolor="#2980B9", linewidth=1, linestyle="--"))
    ax.text(6.5, 4.25, "x̂_t", ha="center", va="center", fontsize=10)

    # CTI extraction
    ax.add_patch(mpatches.FancyBboxPatch((1, 1), 4, 1.5, boxstyle="round,pad=0.2",
                 facecolor="#F9E79F", edgecolor="#F39C12", linewidth=2))
    ax.text(3, 1.75, "CTI = ‖Var(Δz)‖₂", ha="center", va="center", fontsize=9)

    ax.axis("off")

    # === Panel 2: Neural SDE ===
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Latent Neural SDE", fontsize=13, fontweight="bold")

    # SDE equation
    ax.text(5, 9, "dz = μ_θ(z,t,c)dt + σ_θ(z,CTI)dW", ha="center",
            fontsize=10, style="italic", bbox=dict(boxstyle="round", facecolor="#E8DAEF"))

    # Sample paths (fan out)
    import numpy as np
    np.random.seed(42)
    t_vals = np.linspace(0, 1, 50)
    for _ in range(8):
        path = 5 + np.cumsum(np.random.randn(50) * 0.15)
        color_intensity = np.random.uniform(0.3, 0.8)
        ax.plot(t_vals * 8 + 1, path * 0.4 + 3, alpha=0.5,
                color=plt.cm.Reds(color_intensity), linewidth=1)

    ax.text(1, 7.5, "z_t", fontsize=11, fontweight="bold")
    ax.text(9, 7.5, "z_{t+h}", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(8.5, 7.5), xytext=(1.5, 7.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#7F8C8D", linestyle="--"))

    # CTI gate
    ax.add_patch(mpatches.FancyBboxPatch((3, 0.5), 4, 1.5, boxstyle="round,pad=0.2",
                 facecolor="#F9E79F", edgecolor="#F39C12", linewidth=2))
    ax.text(5, 1.25, "CTI gates σ_θ", ha="center", va="center", fontsize=10)

    ax.axis("off")

    # === Panel 3: Score Decoder ===
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Score-Matching Decoder", fontsize=13, fontweight="bold")

    # Latent input
    ax.add_patch(mpatches.Ellipse((2, 7.5), 2, 1.5, facecolor="#FADBD8",
                 edgecolor="#E74C3C", linewidth=2))
    ax.text(2, 7.5, "z_{t+h}", ha="center", va="center", fontsize=10)

    # Arrow
    ax.annotate("", xy=(5, 7.5), xytext=(3.2, 7.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
    ax.text(4.1, 8, "s_ω", ha="center", fontsize=9, style="italic")

    # Distribution
    x = np.linspace(4.5, 9.5, 100)
    y1 = 3 * np.exp(-0.5 * ((x - 7) / 0.8) ** 2) + 7
    ax.fill_between(x, 7, y1, alpha=0.4, color="#27AE60")
    ax.plot(x, y1, color="#27AE60", linewidth=2)
    ax.text(7, 6.3, "p(GHI_{t+h} | z, CTI, c)", ha="center", fontsize=9, style="italic")

    # Output label
    ax.add_patch(mpatches.FancyBboxPatch((4, 2), 5, 2, boxstyle="round,pad=0.3",
                 facecolor="#D5F5E3", edgecolor="#27AE60", linewidth=2))
    ax.text(6.5, 3, "Probabilistic\nIrradiance Forecast", ha="center", va="center", fontsize=10)

    ax.axis("off")

    plt.suptitle("SolarSDE Architecture", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
