#!/usr/bin/env python3
"""Extract image-based features for SolarSDE v4.

Designed to run inside the combined Kaggle/Colab notebook (where images are
already downloaded into data/raw/cloudcv/). Also works as a standalone script
if run locally after data/raw/cloudcv has been populated.

Features (per-timestep):
  Optical flow (between consecutive frames, Farneback):
    flow_mean_x, flow_mean_y   - mean horizontal / vertical motion (pixels/frame)
    flow_magnitude             - sqrt(mx² + my²)
    flow_direction_sin/cos     - cyclic direction encoding
    flow_variance              - spatial variance of flow magnitude (cloud field fragmentation)
  Sun-ROI features (ROI = 50-px radius around computed sun pixel):
    sun_roi_brightness         - mean intensity in ROI
    sun_roi_variance           - pixel variance in ROI (speckle/cloud edge)
    sun_roi_edge_density       - Sobel edge density in ROI
  Whole-image cloud features:
    cloud_fraction             - fraction of pixels above a brightness threshold
    sky_blueness               - (B - R) / (B + R), averaged (clear sky = positive)

Total: 10 image features. Concat with 15 physics + 5 original covariates -> 30-dim c_t.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ==== Try to import OpenCV (required for optical flow) ====
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    log.warning("OpenCV not available — optical flow features will be zero. "
                "Install with: pip install opencv-python-headless")
    HAVE_CV2 = False

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_DIR / "colab_outputs" / "splits"
LATENT_DIR = PROJECT_DIR / "colab_outputs" / "latents"
IMG_SIZE = 128   # match VAE processing size


def load_img_small(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load, center-crop, resize — matches SkyImageDataset preprocessing."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    l, t = (w - side) // 2, (h - side) // 2
    img = img.crop((l, t, l + side, t + side)).resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def sun_pixel_coords(zenith_deg: float, azimuth_deg: float, size: int = IMG_SIZE) -> tuple[int, int]:
    """Approximate sun location in the (center-cropped, resized) fisheye image.

    Fisheye maps zenith angle to radial distance from center.
    Azimuth maps to angular position. Camera oriented so up of image = east.
    """
    # Fisheye radial fraction: r/R = zenith / 90 (equidistant projection)
    r_frac = np.clip(zenith_deg / 90.0, 0, 1)
    radius_pixels = r_frac * (size / 2 - 5)

    # Azimuth 0 = north, 90 = east. Image top = east per CloudCV docs.
    # So east is -y direction (up in image), north is -x (left).
    az_rad = np.deg2rad(azimuth_deg)
    dx =  radius_pixels * np.sin(az_rad)    # east component -> +x (right in image)
    dy = -radius_pixels * np.cos(az_rad)    # north component -> -y

    cx, cy = size // 2, size // 2
    return int(cx + dx), int(cy + dy)


def sun_roi_features(img_gray: np.ndarray, sun_x: int, sun_y: int, r: int = 15) -> tuple[float, float, float]:
    """Extract brightness, variance, edge density in the solar ROI.

    img_gray: (H, W) uint8 grayscale
    Returns (brightness, variance, edge_density) in [0, 1] normalized units.
    """
    H, W = img_gray.shape
    x0, x1 = max(0, sun_x - r), min(W, sun_x + r)
    y0, y1 = max(0, sun_y - r), min(H, sun_y + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    roi = img_gray[y0:y1, x0:x1].astype(np.float32)
    brightness = roi.mean() / 255.0
    variance = roi.var() / (255.0 ** 2)
    if HAVE_CV2:
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
        edge_density = np.sqrt(gx ** 2 + gy ** 2).mean() / 255.0
    else:
        # Manual gradient
        gx = np.abs(np.diff(roi, axis=1, prepend=roi[:, :1])).mean()
        gy = np.abs(np.diff(roi, axis=0, prepend=roi[:1, :])).mean()
        edge_density = (gx + gy) / 255.0
    return float(brightness), float(variance), float(edge_density)


def cloud_fraction(img_rgb: np.ndarray, threshold: float = 0.75) -> float:
    """Fraction of pixels that are bright (whitish), i.e. cloud-covered."""
    img = img_rgb.astype(np.float32) / 255.0
    mean_channel = img.mean(axis=2)
    return float((mean_channel > threshold).mean())


def sky_blueness(img_rgb: np.ndarray) -> float:
    """(B - R) / (B + R + eps), averaged over non-bright pixels (actual sky)."""
    img = img_rgb.astype(np.float32) / 255.0
    mean_channel = img.mean(axis=2)
    sky_mask = mean_channel < 0.75     # exclude cloud pixels
    if sky_mask.sum() < 10:
        return 0.0
    R = img[..., 0][sky_mask]
    B = img[..., 2][sky_mask]
    return float(((B - R) / (B + R + 1e-6)).mean())


def extract_features_for_split(df: pd.DataFrame, split_name: str) -> np.ndarray:
    """Loop through all rows, compute features per image."""
    n = len(df)
    log.info(f"[{split_name}] Extracting image features for {n} rows ...")
    feats = np.zeros((n, 10), dtype=np.float32)   # 10 image features

    # Preload first image
    prev_gray = None

    for i in tqdm(range(n), desc=f"images ({split_name})"):
        row = df.iloc[i]
        img_path = row["image_path"]
        if not Path(img_path).exists():
            continue

        try:
            img_rgb = load_img_small(img_path)
            img_gray = np.mean(img_rgb, axis=2).astype(np.uint8)
        except Exception as e:
            continue

        # Optical flow with previous frame
        if prev_gray is not None and HAVE_CV2:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, img_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0,
                )
                fx = flow[..., 0].mean()
                fy = flow[..., 1].mean()
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                fmag = mag.mean()
                fvar = mag.var()
                fdir = np.arctan2(fy, fx)
                feats[i, 0] = fx / 10.0           # normalize pixel/frame to ~[-1, 1]
                feats[i, 1] = fy / 10.0
                feats[i, 2] = fmag / 10.0
                feats[i, 3] = np.sin(fdir)
                feats[i, 4] = np.cos(fdir)
                feats[i, 5] = np.tanh(fvar / 100.0)
            except Exception:
                pass

        # Sun ROI
        zen = row.get("solar_zenith", 90.0)
        az = row.get("solar_azimuth", 0.0)
        sun_x, sun_y = sun_pixel_coords(zen, az, IMG_SIZE)
        b, v, e = sun_roi_features(img_gray, sun_x, sun_y, r=12)
        feats[i, 6] = b
        feats[i, 7] = v
        feats[i, 8] = e

        # Whole-image features
        feats[i, 9] = cloud_fraction(img_rgb)
        # (10 features total; can extend with sky_blueness if we have room)

        prev_gray = img_gray

    return feats


def main():
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
        df = df[df["image_exists"]].reset_index(drop=True) if "image_exists" in df.columns else df

        feats = extract_features_for_split(df, split)
        out = LATENT_DIR / f"{split}_image_features.npy"
        np.save(out, feats)
        log.info(f"  [{split}] saved {feats.shape} to {out}")
        log.info(f"    per-col mean: {feats.mean(0)}")
        log.info(f"    per-col std:  {feats.std(0)}")


if __name__ == "__main__":
    main()
