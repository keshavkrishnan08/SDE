"""Stage 2: Extract latent trajectories and compute CTI from frozen CS-VAE."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cs_vae import CloudStateVAE
from src.models.cti import compute_cti_from_numpy
from src.data.dataset import SkyImageDataset
from src.utils.io import load_checkpoint, get_device

logger = logging.getLogger(__name__)


def extract_latents(config: dict, vae_checkpoint: str | Path | None = None) -> dict[str, Path]:
    """Extract latent representations and CTI from all splits using frozen VAE.

    Stage 2 of the SolarSDE pipeline.

    Args:
        config: Full configuration dictionary.
        vae_checkpoint: Path to trained VAE checkpoint. Defaults to best checkpoint.

    Returns:
        Dictionary mapping split names to output paths.
    """
    device = get_device()
    vae_cfg = config["vae"]

    # Load frozen VAE
    model = CloudStateVAE(
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta"],
        encoder_channels=vae_cfg.get("encoder_channels"),
    ).to(device)

    if vae_checkpoint is None:
        vae_checkpoint = Path("outputs/checkpoints/vae_best.pt")
    load_checkpoint(vae_checkpoint, model)
    model.eval()

    splits_dir = Path(config["data"]["processed_dir"]) / "splits"
    output_dir = Path(config["data"]["processed_dir"]) / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"{split}.parquet"
        if not split_path.exists():
            logger.warning(f"Split file not found: {split_path}")
            continue

        dataset = SkyImageDataset(split_path)
        loader = DataLoader(
            dataset, batch_size=vae_cfg["batch_size"], shuffle=False,
            num_workers=4, pin_memory=True,
        )

        all_latents = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f"Extracting latents ({split})"):
                images = images.to(device)
                mu = model.encode_to_latent(images)
                all_latents.append(mu.cpu().numpy())

        latents = np.concatenate(all_latents, axis=0)  # (N, d_z)
        logger.info(f"{split}: extracted {latents.shape[0]} latent vectors of dim {latents.shape[1]}")

        # Compute CTI
        cti = compute_cti_from_numpy(latents, window_size=config["cti"]["window_size"])
        logger.info(f"{split}: CTI range [{cti.min():.4f}, {cti.max():.4f}], mean={cti.mean():.4f}")

        # Save
        latent_path = output_dir / f"{split}_latents.npy"
        cti_path = output_dir / f"{split}_cti.npy"
        np.save(latent_path, latents)
        np.save(cti_path, cti)

        output_paths[f"{split}_latents"] = latent_path
        output_paths[f"{split}_cti"] = cti_path

    logger.info("Latent extraction and CTI computation complete.")
    return output_paths
