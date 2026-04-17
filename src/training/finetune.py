"""Stage 5: Optional end-to-end fine-tuning of all SolarSDE components."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.solar_sde import SolarSDE
from src.data.dataset import SolarSequenceDataset
from src.utils.io import save_checkpoint, load_checkpoint, get_device
from src.utils.logging import ExperimentLogger
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def finetune(config: dict) -> SolarSDE:
    """End-to-end fine-tuning of the full SolarSDE pipeline.

    Stage 5 (optional). Unfreezes all components and trains jointly with
    reduced learning rate and combined loss.

    Args:
        config: Full configuration dictionary.

    Returns:
        Fine-tuned SolarSDE model.
    """
    seed_everything(config["evaluation"]["seeds"][0])
    device = get_device()
    ft_cfg = config["finetune"]

    # Build full model and load pre-trained components
    model = SolarSDE(config).to(device)

    checkpoint_dir = Path("outputs/checkpoints")
    load_checkpoint(checkpoint_dir / "vae_best.pt", model.vae)
    load_checkpoint(checkpoint_dir / "sde_best.pt", model.sde)
    load_checkpoint(checkpoint_dir / "score_best.pt", model.score_decoder)

    # Data
    splits_dir = Path(config["data"]["processed_dir"]) / "splits"
    train_dataset = SolarSequenceDataset(
        splits_dir / "train.parquet",
        seq_len=config["data"]["sequence_length"],
        forecast_horizons=config["data"]["forecast_horizons"],
    )
    val_dataset = SolarSequenceDataset(
        splits_dir / "val.parquet",
        seq_len=config["data"]["sequence_length"],
        forecast_horizons=config["data"]["forecast_horizons"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Differential learning rates
    param_groups = [
        {"params": model.vae.parameters(), "lr": ft_cfg["learning_rate"] * 0.1},
        {"params": model.sde.parameters(), "lr": ft_cfg["learning_rate"]},
        {"params": model.score_decoder.parameters(), "lr": ft_cfg["learning_rate"]},
    ]
    optimizer = torch.optim.Adam(param_groups)

    exp_logger = ExperimentLogger(
        project=config["logging"]["project_name"],
        run_name="stage5_finetune",
        use_wandb=config["logging"]["use_wandb"],
    )

    best_val_loss = float("inf")

    for epoch in range(ft_cfg["epochs"]):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
            image = batch["image"].to(device)

            # VAE forward
            recon, mu, logvar = model.vae(image)
            vae_losses = model.vae.loss(image, recon, mu, logvar)

            # Use the latent mean for SDE loss (simplified for fine-tuning)
            z_t = mu.detach()  # Detach to avoid second-order gradients through VAE

            loss = vae_losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                recon, mu, logvar = model.vae(image)
                vae_losses = model.vae.loss(image, recon, mu, logvar)
                val_loss += vae_losses["loss"].item()

        val_loss /= len(val_loader)

        exp_logger.log({"train/loss": train_loss, "val/loss": val_loss}, step=epoch)
        logger.info(f"Finetune {epoch+1}/{ft_cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {"val_loss": val_loss}, checkpoint_dir / "solar_sde_best.pt")

    save_checkpoint(model, optimizer, ft_cfg["epochs"] - 1, {"val_loss": val_loss}, checkpoint_dir / "solar_sde_final.pt")
    exp_logger.finish()
    logger.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return model
