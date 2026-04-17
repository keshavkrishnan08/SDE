#!/usr/bin/env python3
"""End-to-end SolarSDE training pipeline on downloaded CloudCV + BMS data.

Runs all 5 stages sequentially:
  Stage 1: Train CS-VAE
  Stage 2: Extract latents + CTI
  Stage 3: Train Neural SDE
  Stage 4: Train Score Decoder
  Stage 5: (Optional) End-to-end fine-tuning

Also trains baselines and runs evaluation.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.cs_vae import CloudStateVAE
from src.models.cti import compute_cti_from_numpy
from src.models.neural_sde import LatentNeuralSDE
from src.models.score_decoder import ConditionalScoreDecoder
from src.data.dataset import SkyImageDataset, LatentSequenceDataset
from src.utils.config import load_config
from src.utils.seeding import seed_everything
from src.utils.io import save_checkpoint, load_checkpoint, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLITS_DIR = PROJECT_DIR / "data" / "processed" / "splits"
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
LATENT_DIR = PROJECT_DIR / "data" / "processed" / "latents"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LATENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STAGE 1: Train CS-VAE
# ============================================================

def train_vae_stage(config: dict) -> CloudStateVAE:
    """Train Cloud-State VAE on sky images."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Training CS-VAE")
    logger.info("=" * 60)

    device = get_device()
    vae_cfg = config["vae"]
    seed_everything(42)

    img_size = 128  # Use 128x128 for CPU feasibility
    train_dataset = SkyImageDataset(SPLITS_DIR / "train.parquet", target_size=img_size)
    val_dataset = SkyImageDataset(SPLITS_DIR / "val.parquet", target_size=img_size)

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    logger.info(f"Image size: {img_size}x{img_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=vae_cfg["batch_size"], shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=vae_cfg["batch_size"], shuffle=False,
        num_workers=0,
    )

    # Smaller encoder for 128x128 (4 conv layers -> 8x8 feature maps)
    model = CloudStateVAE(
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta"],
        encoder_channels=[32, 64, 128, 256],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=vae_cfg["learning_rate"])
    best_val_loss = float("inf")

    for epoch in range(vae_cfg["epochs"]):
        # Train
        model.train()
        train_loss = 0
        for images in train_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            losses = model.loss(images, recon, mu, logvar)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            train_loss += losses["loss"].item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                losses = model.loss(images, recon, mu, logvar)
                val_loss += losses["loss"].item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{vae_cfg['epochs']} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                          {"val_loss": val_loss}, CHECKPOINT_DIR / "vae_best.pt")

    save_checkpoint(model, optimizer, epoch,
                   {"val_loss": val_loss}, CHECKPOINT_DIR / "vae_final.pt")
    logger.info(f"VAE training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ============================================================
# STAGE 2: Extract Latents + CTI
# ============================================================

def extract_latents_stage(config: dict) -> None:
    """Extract latent representations and CTI from frozen VAE."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Extracting Latents + CTI")
    logger.info("=" * 60)

    device = get_device()
    vae_cfg = config["vae"]

    img_size = 128
    model = CloudStateVAE(
        latent_dim=vae_cfg["latent_dim"], beta=vae_cfg["beta"],
        encoder_channels=[32, 64, 128, 256],
    ).to(device)
    load_checkpoint(CHECKPOINT_DIR / "vae_best.pt", model)
    model.eval()

    for split in ["train", "val", "test"]:
        split_path = SPLITS_DIR / f"{split}.parquet"
        if not split_path.exists():
            continue

        dataset = SkyImageDataset(split_path, target_size=img_size)
        loader = DataLoader(dataset, batch_size=vae_cfg["batch_size"],
                          shuffle=False, num_workers=0)

        all_latents = []
        with torch.no_grad():
            for images in tqdm(loader, desc=f"Encoding {split}"):
                images = images.to(device)
                mu = model.encode_to_latent(images)
                all_latents.append(mu.cpu().numpy())

        latents = np.concatenate(all_latents, axis=0)
        cti = compute_cti_from_numpy(latents, window_size=config["cti"]["window_size"])

        np.save(LATENT_DIR / f"{split}_latents.npy", latents)
        np.save(LATENT_DIR / f"{split}_cti.npy", cti)

        # Also save GHI and covariates for SDE/Score training
        df = pd.read_parquet(split_path)
        if "image_exists" in df.columns:
            df = df[df["image_exists"]].reset_index(drop=True)

        np.save(LATENT_DIR / f"{split}_ghi.npy", df["ghi"].values.astype(np.float32))

        cov_cols = [c for c in ["solar_zenith", "clear_sky_index", "temperature",
                                "humidity", "wind_speed"] if c in df.columns]
        if cov_cols:
            covs = df[cov_cols].fillna(0).values.astype(np.float32)
            np.save(LATENT_DIR / f"{split}_covariates.npy", covs)

        logger.info(f"  {split}: {latents.shape[0]} latents (d={latents.shape[1]}), "
                   f"CTI range [{cti.min():.4f}, {cti.max():.4f}]")

    logger.info("Latent extraction complete.")


# ============================================================
# STAGE 3: Train Neural SDE
# ============================================================

def train_sde_stage(config: dict) -> LatentNeuralSDE:
    """Train the Latent Neural SDE via SDE Matching."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Training Neural SDE")
    logger.info("=" * 60)

    device = get_device()
    sde_cfg = config["sde"]
    vae_cfg = config["vae"]
    seed_everything(42)

    # Determine actual covariate dimension from saved data
    cov_path = LATENT_DIR / "train_covariates.npy"
    if cov_path.exists():
        cov_dim = np.load(cov_path).shape[1]
    else:
        cov_dim = 0
    logger.info(f"  Covariate dimension: {cov_dim}")

    train_dataset = LatentSequenceDataset(
        LATENT_DIR / "train_latents.npy",
        LATENT_DIR / "train_cti.npy",
        LATENT_DIR / "train_ghi.npy",
        covariates_path=cov_path if cov_path.exists() else None,
        seq_len=config["data"]["sequence_length"],
    )
    val_dataset = LatentSequenceDataset(
        LATENT_DIR / "val_latents.npy",
        LATENT_DIR / "val_cti.npy",
        LATENT_DIR / "val_ghi.npy",
        covariates_path=LATENT_DIR / "val_covariates.npy" if (LATENT_DIR / "val_covariates.npy").exists() else None,
        seq_len=config["data"]["sequence_length"],
    )

    logger.info(f"  Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

    train_loader = DataLoader(train_dataset, batch_size=sde_cfg["batch_size"],
                            shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=sde_cfg["batch_size"],
                          shuffle=False, num_workers=0)

    model = LatentNeuralSDE(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=max(cov_dim, 1),
        drift_hidden=sde_cfg["drift_hidden"],
        diffusion_hidden=sde_cfg["diffusion_hidden"],
        lambda_sigma=sde_cfg["lambda_sigma"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=sde_cfg["learning_rate"])
    best_val_loss = float("inf")

    for epoch in range(sde_cfg["epochs"]):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_next = batch["z_next"].to(device)
            cti_t = batch["cti_t"].to(device).unsqueeze(-1)
            t = torch.zeros(z_t.shape[0], 1, device=device)

            if "covariates" in batch:
                c_t = batch["covariates"].to(device)
            else:
                c_t = torch.zeros(z_t.shape[0], max(cov_dim, 1), device=device)

            losses = model.sde_matching_loss(z_t, z_next, t, c_t, cti_t)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += losses["loss"].item()
            n_batches += 1

        if n_batches == 0:
            logger.warning("No training batches — dataset too small for sequence length")
            break

        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                z_next = batch["z_next"].to(device)
                cti_t = batch["cti_t"].to(device).unsqueeze(-1)
                t = torch.zeros(z_t.shape[0], 1, device=device)
                c_t = batch.get("covariates", torch.zeros(z_t.shape[0], max(cov_dim, 1)))
                if isinstance(c_t, torch.Tensor):
                    c_t = c_t.to(device)

                losses = model.sde_matching_loss(z_t, z_next, t, c_t, cti_t)
                val_loss += losses["loss"].item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{sde_cfg['epochs']} | "
                       f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                          {"val_loss": val_loss}, CHECKPOINT_DIR / "sde_best.pt")

    save_checkpoint(model, optimizer, epoch,
                   {"val_loss": val_loss}, CHECKPOINT_DIR / "sde_final.pt")
    logger.info(f"SDE training complete. Best val loss: {best_val_loss:.6f}")
    return model


# ============================================================
# STAGE 4: Train Score Decoder
# ============================================================

def train_score_stage(config: dict) -> ConditionalScoreDecoder:
    """Train the Conditional Score-Matching Irradiance Decoder."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Training Score-Matching Decoder")
    logger.info("=" * 60)

    device = get_device()
    score_cfg = config["score"]
    vae_cfg = config["vae"]
    seed_everything(42)

    cov_path = LATENT_DIR / "train_covariates.npy"
    cov_dim = np.load(cov_path).shape[1] if cov_path.exists() else 0

    train_dataset = LatentSequenceDataset(
        LATENT_DIR / "train_latents.npy",
        LATENT_DIR / "train_cti.npy",
        LATENT_DIR / "train_ghi.npy",
        covariates_path=cov_path if cov_path.exists() else None,
        seq_len=config["data"]["sequence_length"],
    )
    val_dataset = LatentSequenceDataset(
        LATENT_DIR / "val_latents.npy",
        LATENT_DIR / "val_cti.npy",
        LATENT_DIR / "val_ghi.npy",
        covariates_path=LATENT_DIR / "val_covariates.npy" if (LATENT_DIR / "val_covariates.npy").exists() else None,
        seq_len=config["data"]["sequence_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=score_cfg["batch_size"],
                            shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=score_cfg["batch_size"],
                          shuffle=False, num_workers=0)

    model = ConditionalScoreDecoder(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=max(cov_dim, 1),
        hidden_dim=score_cfg["hidden_dim"],
        num_res_blocks=score_cfg["num_res_blocks"],
        diffusion_steps=score_cfg["diffusion_steps"],
        beta_start=score_cfg["beta_start"],
        beta_end=score_cfg["beta_end"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=score_cfg["learning_rate"])
    best_val_loss = float("inf")

    for epoch in range(score_cfg["epochs"]):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            cti_t = batch["cti_t"].to(device).unsqueeze(-1)
            ghi_t = batch["ghi_t"].to(device).unsqueeze(-1)

            if "covariates" in batch:
                c_t = batch["covariates"].to(device)
            else:
                c_t = torch.zeros(z_t.shape[0], max(cov_dim, 1), device=device)

            losses = model.training_loss(ghi_t, z_t, cti_t, c_t)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            train_loss += losses["loss"].item()
            n_batches += 1

        if n_batches == 0:
            break

        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                cti_t = batch["cti_t"].to(device).unsqueeze(-1)
                ghi_t = batch["ghi_t"].to(device).unsqueeze(-1)
                c_t = batch.get("covariates", torch.zeros(z_t.shape[0], max(cov_dim, 1)))
                if isinstance(c_t, torch.Tensor):
                    c_t = c_t.to(device)
                losses = model.training_loss(ghi_t, z_t, cti_t, c_t)
                val_loss += losses["loss"].item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{score_cfg['epochs']} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                          {"val_loss": val_loss}, CHECKPOINT_DIR / "score_best.pt")

    save_checkpoint(model, optimizer, epoch,
                   {"val_loss": val_loss}, CHECKPOINT_DIR / "score_final.pt")
    logger.info(f"Score decoder training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(config: dict) -> None:
    """Evaluate trained SolarSDE on test set."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION: Running on test set")
    logger.info("=" * 60)

    device = get_device()
    vae_cfg = config["vae"]
    sde_cfg = config["sde"]
    score_cfg = config["score"]

    cov_path = LATENT_DIR / "test_covariates.npy"
    cov_dim = np.load(cov_path).shape[1] if cov_path.exists() else 0

    # Load test data
    test_latents = np.load(LATENT_DIR / "test_latents.npy")
    test_cti = np.load(LATENT_DIR / "test_cti.npy")
    test_ghi = np.load(LATENT_DIR / "test_ghi.npy")
    test_covs = np.load(cov_path) if cov_path.exists() else np.zeros((len(test_latents), 1))

    # Load models
    sde_model = LatentNeuralSDE(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=max(cov_dim, 1),
        drift_hidden=sde_cfg["drift_hidden"],
        diffusion_hidden=sde_cfg["diffusion_hidden"],
    ).to(device)
    load_checkpoint(CHECKPOINT_DIR / "sde_best.pt", sde_model)
    sde_model.eval()

    score_model = ConditionalScoreDecoder(
        latent_dim=vae_cfg["latent_dim"],
        covariate_dim=max(cov_dim, 1),
        hidden_dim=score_cfg["hidden_dim"],
        num_res_blocks=score_cfg["num_res_blocks"],
        diffusion_steps=score_cfg["diffusion_steps"],
    ).to(device)
    load_checkpoint(CHECKPOINT_DIR / "score_best.pt", score_model)
    score_model.eval()

    from src.models.sde_solver import solve_sde
    from src.evaluation.metrics import compute_all_metrics, crps_empirical

    # Evaluate at multiple horizons
    horizons = [6, 12, 30, 60]  # steps (× 10s = 1, 2, 5, 10 min)
    num_samples = 50
    seq_len = config["data"]["sequence_length"]

    results = {}
    for h in horizons:
        logger.info(f"\n  Horizon: {h} steps ({h * 10 / 60:.0f} min)")

        y_true_list = []
        y_samples_list = []
        n_eval = min(500, len(test_latents) - seq_len - h)

        for i in range(seq_len, seq_len + n_eval):
            z_0 = torch.from_numpy(test_latents[i:i+1]).float().to(device)
            c_t = torch.from_numpy(test_covs[i:i+1]).float().to(device)
            cti_t = torch.tensor([[test_cti[i]]]).float().to(device)

            t_span = torch.linspace(0, 1, h + 1)
            with torch.no_grad():
                sde_result = solve_sde(sde_model, z_0, t_span, c_t, cti_t,
                                      num_samples=num_samples, dt=1.0)
                z_endpoints = sde_result["endpoints"]  # (1, N, d_z)

                z_flat = z_endpoints.view(num_samples, -1)
                cti_flat = cti_t.expand(num_samples, -1)
                c_flat = c_t.expand(num_samples, -1)

                ghi_samples = score_model.sample(z_flat, cti_flat, c_flat, num_samples=1)
                ghi_samples = ghi_samples.squeeze(-1).cpu().numpy()  # (N,)

            target_idx = i + h
            if target_idx < len(test_ghi):
                y_true_list.append(test_ghi[target_idx])
                y_samples_list.append(ghi_samples)

        if y_true_list:
            y_true = np.array(y_true_list)
            y_samples = np.array(y_samples_list)
            metrics = compute_all_metrics(y_true, y_samples, pi_level=0.90)
            results[h] = metrics
            logger.info(f"    CRPS: {metrics['crps']:.4f} | "
                       f"RMSE: {metrics['rmse']:.2f} | "
                       f"PICP: {metrics['picp']:.3f} | "
                       f"PINAW: {metrics['pinaw']:.4f}")

    # Save results
    results_dir = PROJECT_DIR / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(results_dir / "solar_sde_results.csv")
    logger.info(f"\nResults saved to {results_dir / 'solar_sde_results.csv'}")


# ============================================================
# MAIN
# ============================================================

def main():
    config = load_config(PROJECT_DIR / "configs" / "default.yaml")

    # Adapted for CPU training on 8-day dataset
    config["vae"]["epochs"] = 10
    config["vae"]["batch_size"] = 16
    config["sde"]["epochs"] = 30
    config["sde"]["batch_size"] = 64
    config["score"]["epochs"] = 20
    config["score"]["batch_size"] = 128
    config["data"]["sequence_length"] = 10  # Shorter sequences for smaller dataset

    logger.info("SolarSDE Training Pipeline")
    logger.info(f"Device: {get_device()}")
    start = time.time()

    # Stage 1
    train_vae_stage(config)

    # Stage 2
    extract_latents_stage(config)

    # Stage 3
    train_sde_stage(config)

    # Stage 4
    train_score_stage(config)

    # Evaluate
    run_evaluation(config)

    elapsed = time.time() - start
    logger.info(f"\nTotal pipeline time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
