#!/usr/bin/env python3
"""Pretrain CS-VAE on Stanford SKIPP'D (cloudy California), fine-tune on Golden CO.

Motivation: Golden CO has only 8 days (mostly clear sky). Stanford has 497 days
with more cloud diversity. Pretraining the VAE on Stanford gives it a more
expressive latent representation of cloud structures, which transfers to our
target domain via fine-tuning.

Pipeline:
  1. Pretrain VAE on Stanford 64x64 images (upsampled to 128x128 bilinear)
  2. Fine-tune on Golden CloudCV images for 10-20 epochs
  3. Save as vae_pretrained.pt (replaces the original vae_best.pt if desired)

Runtime: ~2-3 hours pretraining + ~30 min fine-tuning on GPU. Skip by default;
run manually when ready.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data" / "processed"
CKPT_DIR = PROJECT_DIR / "colab_outputs" / "checkpoints"


# ==== VAE architecture (matches the one in notebook) ====
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, in_ch = [], 3
        for ch in channels:
            layers.extend([nn.Conv2d(in_ch, ch, 4, 2, 1),
                          nn.GroupNorm(min(32, ch), ch), nn.SiLU(inplace=True)])
            in_ch = ch
        self.conv = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool2d(1)
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
            layers.extend([nn.ConvTranspose2d(channels[i], channels[i+1], 4, 2, 1),
                          nn.GroupNorm(min(32, channels[i+1]), channels[i+1]),
                          nn.SiLU(inplace=True)])
        layers.extend([nn.ConvTranspose2d(channels[-1], 3, 4, 2, 1), nn.Sigmoid()])
        self.deconv = nn.Sequential(*layers)
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, self.init_ch, 8, 8))

class CloudStateVAE(nn.Module):
    def __init__(self, latent_dim=64, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim; self.beta = beta
        self.encoder = Encoder(latent_dim); self.decoder = Decoder(latent_dim)
    def reparam(self, mu, lv): return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    def forward(self, x):
        mu, lv = self.encoder(x); z = self.reparam(mu, lv)
        return self.decoder(z), mu, lv
    def loss(self, x, recon, mu, lv):
        rec = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return {"loss": rec + self.beta * kl, "recon": rec, "kl": kl}


class StanfordImageDataset(Dataset):
    """Load Stanford 64x64 images, upsample to 128x128 via bilinear."""
    def __init__(self, images_path: Path, target_size: int = 128):
        self.imgs = np.load(images_path)   # (N, 64, 64, 3) uint8
        self.target_size = target_size
        print(f"  Loaded {self.imgs.shape} from {images_path.name}")
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i]).float() / 255.0   # (64, 64, 3)
        img = img.permute(2, 0, 1).unsqueeze(0)                # (1, 3, 64, 64)
        img = F.interpolate(img, size=self.target_size, mode="bilinear", align_corners=False)
        return img.squeeze(0)                                    # (3, 128, 128)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check Stanford data
    sf_imgs = DATA_DIR / "stanford_train_images.npy"
    if not sf_imgs.exists():
        print(f"ERROR: Stanford data not found. Run prepare_stanford_skippd.py first.")
        return

    # === Stage 1: Pretrain VAE on Stanford ===
    print("\n=== Pretraining VAE on Stanford SKIPP'D (cloudy California) ===")
    torch.manual_seed(42)
    sf_ds = StanfordImageDataset(sf_imgs)
    dl = DataLoader(sf_ds, batch_size=64, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)

    vae = CloudStateVAE(latent_dim=64, beta=0.1).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)

    EPOCHS_PRETRAIN = 30
    best_loss = float("inf")
    for ep in range(1, EPOCHS_PRETRAIN + 1):
        vae.train(); tl = tr = tk = 0; n = 0
        for img in dl:
            img = img.to(device, non_blocking=True)
            recon, mu, lv = vae(img)
            losses = vae.loss(img, recon, mu, lv)
            opt.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()
            tl += losses["loss"].item(); tr += losses["recon"].item(); tk += losses["kl"].item()
            n += 1
        tl /= n; tr /= n; tk /= n
        print(f"  Pretrain ep {ep}/{EPOCHS_PRETRAIN} | loss={tl:.4f} (recon={tr:.4f}, kl={tk:.4f})")
        if tl < best_loss:
            best_loss = tl
            torch.save(vae.state_dict(), CKPT_DIR / "vae_stanford_pretrained.pt")

    print(f"\nStanford pretraining done. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'vae_stanford_pretrained.pt'}")
    print("\nNext: fine-tune this VAE on Golden CloudCV images for 10-20 epochs.")


if __name__ == "__main__":
    main()
