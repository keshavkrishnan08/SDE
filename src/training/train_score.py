"""Stage 4: Train the Conditional Score-Matching Irradiance Decoder (CSMID)."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.score_decoder import ConditionalScoreDecoder
from src.data.dataset import LatentSequenceDataset
from src.utils.io import save_checkpoint, get_device
from src.utils.logging import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def train_score_decoder(config: dict) -> ConditionalScoreDecoder:
    """Train the Conditional Score-Matching Irradiance Decoder.

    Stage 4 of the SolarSDE pipeline. Trains a small 1D diffusion model
    to map latent states → calibrated irradiance distributions.

    Args:
        config: Full configuration dictionary.

    Returns:
        Trained ConditionalScoreDecoder model.
    """
    seed_everything(config["evaluation"]["seeds"][0])
    device = get_device()
    score_cfg = config["score"]
    vae_cfg = config["vae"]
    sde_cfg = config["sde"]

    # Data
    latent_dir = Path(config["data"]["processed_dir"]) / "latents"

    train_dataset = LatentSequenceDataset(
        latents_path=latent_dir / "train_latents.npy",
        cti_path=latent_dir / "train_cti.npy",
        ghi_path=latent_dir / "train_ghi.npy",
        covariates_path=latent_dir / "train_covariates.npy" if (latent_dir / "train_covariates.npy").exists() else None,
        seq_len=config["data"]["sequence_length"],
    )

    val_dataset = LatentSequenceDataset(
        latents_path=latent_dir / "val_latents.npy",
        cti_path=latent_dir / "val_cti.npy",
        ghi_path=latent_dir / "val_ghi.npy",
        covariates_path=latent_dir / "val_covariates.npy" if (latent_dir / "val_covariates.npy").exists() else None,
        seq_len=config["data"]["sequence_length"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=score_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=score_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    covariate_dim = sde_cfg["covariate_dim"]
    model = ConditionalScoreDecoder(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=covariate_dim,
        hidden_dim=score_cfg["hidden_dim"],
        num_res_blocks=score_cfg["num_res_blocks"],
        diffusion_steps=score_cfg["diffusion_steps"],
        beta_start=score_cfg["beta_start"],
        beta_end=score_cfg["beta_end"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=score_cfg["learning_rate"])
    exp_logger = ExperimentLogger(
        project=config["logging"]["project_name"],
        run_name="stage4_score",
        use_wandb=config["logging"]["use_wandb"],
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path("outputs/checkpoints")

    for epoch in range(score_cfg["epochs"]):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Score Epoch {epoch+1}"):
            z_t = batch["z_t"].to(device)
            cti_t = batch["cti_t"].to(device).unsqueeze(-1)
            ghi_t = batch["ghi_t"].to(device).unsqueeze(-1)

            if "covariates" in batch:
                c_t = batch["covariates"].to(device)
            else:
                c_t = torch.zeros(z_t.shape[0], covariate_dim, device=device)

            losses = model.training_loss(ghi_t, z_t, cti_t, c_t)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            train_loss += losses["loss"].item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                cti_t = batch["cti_t"].to(device).unsqueeze(-1)
                ghi_t = batch["ghi_t"].to(device).unsqueeze(-1)
                c_t = batch.get("covariates", torch.zeros(z_t.shape[0], covariate_dim)).to(device)

                losses = model.training_loss(ghi_t, z_t, cti_t, c_t)
                val_loss += losses["loss"].item()

        val_loss /= len(val_loader)

        exp_logger.log({"train/loss": train_loss, "val/loss": val_loss}, step=epoch)
        logger.info(f"Epoch {epoch+1}/{score_cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {"val_loss": val_loss}, checkpoint_dir / "score_best.pt")

    save_checkpoint(model, optimizer, score_cfg["epochs"] - 1, {"val_loss": val_loss}, checkpoint_dir / "score_final.pt")
    exp_logger.finish()
    logger.info(f"Score decoder training complete. Best val loss: {best_val_loss:.4f}")
    return model
