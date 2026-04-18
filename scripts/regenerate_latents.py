#!/usr/bin/env python3
"""Re-extract latents + CTI + BMS-interpolated fields using the corrected
timestamp-aligned dataset and the existing VAE checkpoint.

This is what Notebook 1's Stage 2 (latent extraction) does, but on CPU.
Produces the `colab_outputs/latents/*.npy` files so Notebooks 2-5 can
fast-start from the corrected data without re-running Notebook 1.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IMG_SIZE = 128
LATENT_DIM = 64
CTI_WINDOW = 10
BATCH_SIZE = 32

SPLITS_DIR = PROJECT_DIR / "data" / "processed" / "splits"
VAE_CKPT = PROJECT_DIR / "colab_outputs" / "checkpoints" / "vae_best.pt"
OUT_DIR = PROJECT_DIR / "colab_outputs" / "latents"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# VAE definition (must match colab_outputs checkpoint)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, in_ch = [], 3
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, 4, 2, 1),
                nn.GroupNorm(min(32, ch), ch),
                nn.SiLU(inplace=True),
            ])
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(channels[-1], latent_dim)
        self.fc_lv = nn.Linear(channels[-1], latent_dim)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, channels=(256, 128, 64, 32)):
        super().__init__()
        self.init_ch = channels[0]
        self.fc = nn.Linear(latent_dim, channels[0] * 8 * 8)
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, 2, 1),
                nn.GroupNorm(min(32, channels[i + 1]), channels[i + 1]),
                nn.SiLU(inplace=True),
            ])
        layers.extend([nn.ConvTranspose2d(channels[-1], 3, 4, 2, 1), nn.Sigmoid()])
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, self.init_ch, 8, 8))


class CloudStateVAE(nn.Module):
    def __init__(self, latent_dim=64, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @torch.no_grad()
    def encode_mu(self, x):
        mu, _ = self.encoder(x)
        return mu


# ============================================================================
# Dataset
# ============================================================================

def load_img(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    l, t = (w - side) // 2, (h - side) // 2
    img = img.crop((l, t, l + side, t + side)).resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class SkyDS(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        if "image_exists" in self.df.columns:
            self.df = self.df[self.df["image_exists"]].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        arr = load_img(self.df.iloc[i]["image_path"])
        return torch.from_numpy(arr).permute(2, 0, 1)


# ============================================================================
# CTI
# ============================================================================

def cti_from_latents(Z: np.ndarray, window: int = CTI_WINDOW) -> np.ndarray:
    T = Z.shape[0]
    cti = np.zeros(T, dtype=np.float32)
    for t in range(window, T):
        v = np.diff(Z[t - window:t], axis=0)
        cti[t] = np.linalg.norm(v.var(axis=0), ord=2)
    return cti


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading VAE checkpoint: {VAE_CKPT}")
    vae = CloudStateVAE(latent_dim=LATENT_DIM, beta=0.1).to(device)
    state = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    vae.load_state_dict(state)
    vae.eval()
    log.info(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()):,} params")

    for split in ["train", "val", "test"]:
        log.info(f"\n=== {split.upper()} ===")
        ds = SkyDS(SPLITS_DIR / f"{split}.parquet")
        log.info(f"  {len(ds)} images")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        t0 = time.time()
        all_mu = []
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            mu = vae.encode_mu(batch)
            all_mu.append(mu.numpy())
        Z = np.concatenate(all_mu, axis=0).astype(np.float32)
        log.info(f"  Encoded {Z.shape} in {time.time() - t0:.1f}s")

        cti = cti_from_latents(Z, window=CTI_WINDOW)
        log.info(f"  CTI range [{cti.min():.4f}, {cti.max():.4f}]  "
                 f"mean {cti.mean():.4f}")

        df = ds.df
        ghi = df["ghi"].values.astype(np.float32)
        cov_cols = [c for c in ["solar_zenith", "clear_sky_index", "temperature",
                                "humidity", "wind_speed"] if c in df.columns]
        cov = df[cov_cols].fillna(0).values.astype(np.float32) \
            if cov_cols else np.zeros((len(df), 0), np.float32)
        is_ramp = df["is_ramp"].values.astype(bool)

        np.save(OUT_DIR / f"{split}_latents.npy", Z)
        np.save(OUT_DIR / f"{split}_cti.npy", cti)
        np.save(OUT_DIR / f"{split}_ghi.npy", ghi)
        np.save(OUT_DIR / f"{split}_covariates.npy", cov)
        np.save(OUT_DIR / f"{split}_is_ramp.npy", is_ramp)
        log.info(f"  GHI range [{ghi.min():.1f}, {ghi.max():.1f}] W/m²  "
                 f"ramps: {int(is_ramp.sum())}  cov: {cov.shape}")

    log.info("\nAll latents regenerated.")


if __name__ == "__main__":
    main()
