"""Stage 3: Train the Latent Neural SDE using SDE Matching."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.neural_sde import LatentNeuralSDE
from src.data.dataset import LatentSequenceDataset
from src.utils.io import save_checkpoint, get_device
from src.utils.logging import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def train_sde(config: dict) -> LatentNeuralSDE:
    """Train the Latent Neural SDE via SDE Matching.

    Stage 3 of the SolarSDE pipeline. Uses simulation-free training
    (drift matching + diffusion matching on observed finite differences).

    Args:
        config: Full configuration dictionary.

    Returns:
        Trained LatentNeuralSDE model.
    """
    seed_everything(config["evaluation"]["seeds"][0])
    device = get_device()
    sde_cfg = config["sde"]
    vae_cfg = config["vae"]

    # Data
    latent_dir = Path(config["data"]["processed_dir"]) / "latents"
    splits_dir = Path(config["data"]["processed_dir"]) / "splits"

    # Load GHI for covariate extraction
    import pandas as pd
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")

    # Save GHI arrays for the latent dataset
    ghi_dir = latent_dir
    for split, df in [("train", train_df), ("val", val_df)]:
        ghi_path = ghi_dir / f"{split}_ghi.npy"
        if not ghi_path.exists() and "ghi" in df.columns:
            np.save(ghi_path, df["ghi"].values.astype(np.float32))

        # Save covariates
        cov_cols = [c for c in ["solar_zenith", "clear_sky_index", "temperature", "humidity", "wind_speed"]
                    if c in df.columns]
        if cov_cols:
            cov_path = ghi_dir / f"{split}_covariates.npy"
            if not cov_path.exists():
                np.save(cov_path, df[cov_cols].values.astype(np.float32))

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
        train_dataset, batch_size=sde_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=sde_cfg["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    covariate_dim = sde_cfg["covariate_dim"]
    model = LatentNeuralSDE(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=covariate_dim,
        drift_hidden=sde_cfg["drift_hidden"],
        diffusion_hidden=sde_cfg["diffusion_hidden"],
        lambda_sigma=sde_cfg["lambda_sigma"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=sde_cfg["learning_rate"])
    exp_logger = ExperimentLogger(
        project=config["logging"]["project_name"],
        run_name="stage3_sde",
        use_wandb=config["logging"]["use_wandb"],
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path("outputs/checkpoints")

    for epoch in range(sde_cfg["epochs"]):
        model.train()
        train_metrics = {"loss": 0, "drift_loss": 0, "diffusion_loss": 0}

        for batch in tqdm(train_loader, desc=f"SDE Epoch {epoch+1}"):
            z_t = batch["z_t"].to(device)
            z_next = batch["z_next"].to(device)
            cti_t = batch["cti_t"].to(device).unsqueeze(-1)

            # Create time and covariate tensors
            t = torch.zeros(z_t.shape[0], 1, device=device)
            if "covariates" in batch:
                c_t = batch["covariates"].to(device)
            else:
                c_t = torch.zeros(z_t.shape[0], covariate_dim, device=device)

            losses = model.sde_matching_loss(z_t, z_next, t, c_t, cti_t)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in train_metrics:
                train_metrics[k] += losses[k].item()

        n_train = len(train_loader)
        train_metrics = {f"train/{k}": v / n_train for k, v in train_metrics.items()}

        # Validation
        model.eval()
        val_metrics = {"loss": 0, "drift_loss": 0, "diffusion_loss": 0}
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                z_next = batch["z_next"].to(device)
                cti_t = batch["cti_t"].to(device).unsqueeze(-1)
                t = torch.zeros(z_t.shape[0], 1, device=device)
                c_t = batch.get("covariates", torch.zeros(z_t.shape[0], covariate_dim)).to(device)

                losses = model.sde_matching_loss(z_t, z_next, t, c_t, cti_t)
                for k in val_metrics:
                    val_metrics[k] += losses[k].item()

        n_val = len(val_loader)
        val_metrics = {f"val/{k}": v / n_val for k, v in val_metrics.items()}

        all_metrics = {**train_metrics, **val_metrics}
        exp_logger.log(all_metrics, step=epoch)
        logger.info(
            f"Epoch {epoch+1}/{sde_cfg['epochs']} | "
            f"Train: {train_metrics['train/loss']:.4f} | "
            f"Val: {val_metrics['val/loss']:.4f}"
        )

        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir / "sde_best.pt")

    save_checkpoint(model, optimizer, sde_cfg["epochs"] - 1, val_metrics, checkpoint_dir / "sde_final.pt")
    exp_logger.finish()
    logger.info(f"SDE training complete. Best val loss: {best_val_loss:.4f}")
    return model
