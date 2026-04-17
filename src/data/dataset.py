"""PyTorch Dataset classes for SolarSDE training."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_and_preprocess_image(
    img_path: str, target_size: int = 256
) -> np.ndarray:
    """Load a JPEG sky image, crop center square, resize, normalize to [0,1]."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class SkyImageDataset(Dataset):
    """Dataset for CS-VAE pre-training on individual sky images."""

    def __init__(
        self,
        split_path: str | Path,
        image_col: str = "image_path",
        target_size: int = 256,
    ):
        self.df = pd.read_parquet(split_path)
        # Filter to rows with valid image paths
        if "image_exists" in self.df.columns:
            self.df = self.df[self.df["image_exists"]].reset_index(drop=True)
        self.image_col = image_col
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        if str(img_path).endswith(".npy"):
            image = np.load(img_path)
        else:
            image = load_and_preprocess_image(img_path, self.target_size)

        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        return image


class SolarSequenceDataset(Dataset):
    """Dataset for Neural SDE and full SolarSDE training.

    Each sample contains a sequence of past observations and future targets.
    """

    def __init__(
        self,
        split_path: str | Path,
        seq_len: int = 30,
        forecast_horizons: Optional[list[int]] = None,
        load_images: bool = True,
    ):
        self.df = pd.read_parquet(split_path)
        self.seq_len = seq_len
        self.forecast_horizons = forecast_horizons or [6, 12, 30, 60, 90, 120, 180]
        self.load_images = load_images
        self.max_horizon = max(self.forecast_horizons)

        # Extract numerical features as arrays for fast access
        self.ghi = self.df["ghi"].values.astype(np.float32) if "ghi" in self.df.columns else None
        self.clear_sky_index = (
            self.df["clear_sky_index"].values.astype(np.float32)
            if "clear_sky_index" in self.df.columns
            else None
        )

        # Meteorological covariates
        covariate_cols = ["solar_zenith", "clear_sky_index"]
        optional_cols = ["temperature", "humidity", "wind_speed"]
        self.covariate_cols = [c for c in covariate_cols + optional_cols if c in self.df.columns]
        if self.covariate_cols:
            self.covariates = self.df[self.covariate_cols].values.astype(np.float32)
        else:
            self.covariates = None

        # Image paths
        if load_images and "image_path" in self.df.columns:
            self.image_paths = self.df["image_path"].values
        else:
            self.image_paths = None

    def __len__(self) -> int:
        return max(0, len(self.df) - self.seq_len - self.max_horizon)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        sample = {}

        # Past GHI sequence
        if self.ghi is not None:
            sample["ghi_past"] = torch.from_numpy(self.ghi[start:end])
            # Future GHI targets at each horizon
            targets = np.array([self.ghi[end + h] for h in self.forecast_horizons])
            sample["ghi_targets"] = torch.from_numpy(targets)

        # Past clear-sky index
        if self.clear_sky_index is not None:
            sample["kt_past"] = torch.from_numpy(self.clear_sky_index[start:end])

        # Meteorological covariates at current time
        if self.covariates is not None:
            sample["covariates"] = torch.from_numpy(self.covariates[end - 1])

        # Load the most recent image in the sequence
        if self.image_paths is not None:
            img = np.load(self.image_paths[end - 1])
            sample["image"] = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return sample


class LatentSequenceDataset(Dataset):
    """Dataset for training on pre-extracted latent trajectories.

    Used for Neural SDE training (Stage 3) and score decoder training (Stage 4).
    """

    def __init__(
        self,
        latents_path: str | Path,
        cti_path: str | Path,
        ghi_path: str | Path,
        covariates_path: Optional[str | Path] = None,
        seq_len: int = 30,
    ):
        self.latents = np.load(latents_path).astype(np.float32)  # (N, d_z)
        self.cti = np.load(cti_path).astype(np.float32)  # (N,)
        self.ghi = np.load(ghi_path).astype(np.float32)  # (N,)
        self.covariates = (
            np.load(covariates_path).astype(np.float32)
            if covariates_path is not None
            else None
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.latents) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = idx + self.seq_len  # Current timestep
        sample = {
            "z_t": torch.from_numpy(self.latents[t]),
            "z_next": torch.from_numpy(self.latents[t + 1]),
            "cti_t": torch.tensor(self.cti[t]),
            "ghi_t": torch.tensor(self.ghi[t]),
        }
        if self.covariates is not None:
            sample["covariates"] = torch.from_numpy(self.covariates[t])
        return sample
