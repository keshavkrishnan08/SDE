"""Latent space visualization: t-SNE embeddings and latent traversals."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE


def plot_latent_tsne(
    latents: np.ndarray,
    color_values: np.ndarray,
    color_label: str = "Clear-Sky Index",
    output_path: str = "outputs/figures/latent_tsne.pdf",
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    """t-SNE visualization of latent embeddings colored by a variable.

    Args:
        latents: Latent vectors, shape (N, d_z).
        color_values: Values to color-code, shape (N,).
        color_label: Label for the colorbar.
        output_path: Where to save.
        max_points: Max points to embed (t-SNE is O(N²)).
        seed: Random seed.
    """
    n = min(max_points, len(latents))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(latents), n, replace=False)

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    z_2d = tsne.fit_transform(latents[idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=color_values[idx],
                    cmap="viridis", s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label=color_label)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(f"Latent Space Colored by {color_label}", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latent_tsne_multipanel(
    latents: np.ndarray,
    variables: dict[str, np.ndarray],
    output_path: str = "outputs/figures/latent_tsne_multi.pdf",
    max_points: int = 5000,
    seed: int = 42,
) -> None:
    """Multi-panel t-SNE colored by different variables.

    Args:
        latents: Shape (N, d_z).
        variables: Dict mapping variable_name -> values of shape (N,).
        output_path: Where to save.
    """
    n = min(max_points, len(latents))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(latents), n, replace=False)

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    z_2d = tsne.fit_transform(latents[idx])

    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    if n_vars == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, variables.items()):
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=values[idx],
                        cmap="viridis", s=3, alpha=0.6)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Latent Space Organization", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_reconstruction_examples(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    labels: list[str] | None = None,
    output_path: str = "outputs/figures/vae_reconstructions.pdf",
) -> None:
    """Show input vs. reconstructed sky images.

    Args:
        originals: Shape (K, H, W, 3) or (K, 3, H, W).
        reconstructions: Same shape as originals.
        labels: Optional labels for each example.
        output_path: Where to save.
    """
    K = len(originals)
    fig, axes = plt.subplots(2, K, figsize=(3 * K, 6))

    for i in range(K):
        orig = originals[i]
        recon = reconstructions[i]
        if orig.shape[0] == 3:  # CHW -> HWC
            orig = orig.transpose(1, 2, 0)
            recon = recon.transpose(1, 2, 0)

        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].set_title(labels[i] if labels else f"#{i+1}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)

    plt.suptitle("CS-VAE Reconstruction Quality", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
