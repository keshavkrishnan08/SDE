"""Stage 1: CS-VAE pre-training on individual sky images."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cs_vae import CloudStateVAE
from src.data.dataset import SkyImageDataset
from src.utils.io import save_checkpoint, get_device
from src.utils.logging import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def train_vae(config: dict) -> CloudStateVAE:
    """Train the Cloud-State VAE.

    Stage 1 of the SolarSDE pipeline. Trains the VAE on individual sky images
    using a β-VAE objective (reconstruction + β * KL).

    Args:
        config: Full configuration dictionary.

    Returns:
        Trained CloudStateVAE model.
    """
    seed_everything(config["evaluation"]["seeds"][0])
    device = get_device()
    vae_cfg = config["vae"]

    # Data
    splits_dir = Path(config["data"]["processed_dir"]) / "splits"
    train_dataset = SkyImageDataset(splits_dir / "train.parquet")
    val_dataset = SkyImageDataset(splits_dir / "val.parquet")

    train_loader = DataLoader(
        train_dataset, batch_size=vae_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=vae_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = CloudStateVAE(
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta"],
        encoder_channels=vae_cfg.get("encoder_channels"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vae_cfg["learning_rate"])
    exp_logger = ExperimentLogger(
        project=config["logging"]["project_name"],
        run_name="stage1_vae",
        use_wandb=config["logging"]["use_wandb"],
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path("outputs/checkpoints")

    for epoch in range(vae_cfg["epochs"]):
        # Training
        model.train()
        train_metrics = {"recon_loss": 0, "kl_loss": 0, "loss": 0}
        for batch_idx, images in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images = images.to(device)
            recon, mu, logvar = model(images)
            losses = model.loss(images, recon, mu, logvar)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            for k in train_metrics:
                train_metrics[k] += losses[k].item()

        n_batches = len(train_loader)
        train_metrics = {f"train/{k}": v / n_batches for k, v in train_metrics.items()}

        # Validation
        model.eval()
        val_metrics = {"recon_loss": 0, "kl_loss": 0, "loss": 0}
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                losses = model.loss(images, recon, mu, logvar)
                for k in val_metrics:
                    val_metrics[k] += losses[k].item()

        n_val = len(val_loader)
        val_metrics = {f"val/{k}": v / n_val for k, v in val_metrics.items()}

        # Log
        all_metrics = {**train_metrics, **val_metrics}
        exp_logger.log(all_metrics, step=epoch)
        logger.info(
            f"Epoch {epoch+1}/{vae_cfg['epochs']} | "
            f"Train loss: {train_metrics['train/loss']:.4f} | "
            f"Val loss: {val_metrics['val/loss']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / "vae_best.pt",
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, vae_cfg["epochs"] - 1, val_metrics,
        checkpoint_dir / "vae_final.pt",
    )
    exp_logger.finish()
    logger.info(f"VAE training complete. Best val loss: {best_val_loss:.4f}")
    return model
