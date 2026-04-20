#!/usr/bin/env python3
"""Extract physics + temporal features for SolarSDE v4.

Features (per-timestep, all purely closed-form / from existing parquets):
  Solar geometry:
    air_mass               - 1/cos(zenith); how much atmosphere sunlight traverses
    zenith_rate            - d(zenith)/dt; rapid change at dawn/dusk
    azimuth_sin, azimuth_cos - cyclic encoding of sun direction
  Temporal context:
    hour_sin, hour_cos     - cyclic local time
    doy_sin, doy_cos       - cyclic day-of-year
    time_since_sunrise     - normalized position in daylight
  kt trend / variability:
    kt_trend_1min          - (kt(t) - kt(t-6)) / 6   at 10s sampling
    kt_trend_5min          - (kt(t) - kt(t-30)) / 30
    kt_trend_10min         - (kt(t) - kt(t-60)) / 60
    kt_std_5min            - rolling std of kt over 30 steps
    ghi_std_1min           - rolling std of GHI over 6 steps (raw variability)
  Pyranometer:
    pyr_mv                 - raw LI-200 voltage (high-freq signal)

Total: 15 features. Concatenated with the existing 5 covariates -> 20-dim c_t.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_DIR / "colab_outputs" / "splits"
LATENT_DIR = PROJECT_DIR / "colab_outputs" / "latents"


def compute_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 15 physics/temporal features from a parquet split."""
    df = df.reset_index(drop=True).copy()
    n = len(df)

    # --- Solar geometry ---
    zenith = df["solar_zenith"].values.astype(np.float32)
    zenith_rad = np.deg2rad(np.clip(zenith, 0, 89.9))

    # Air mass — Kasten-Young approximation for stability at high zenith
    air_mass = 1.0 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    air_mass = np.clip(air_mass, 1.0, 40.0).astype(np.float32)

    # Zenith rate (degrees/second via finite difference, 10s spacing)
    zenith_rate = np.zeros_like(zenith)
    zenith_rate[1:] = (zenith[1:] - zenith[:-1]) / 10.0

    # Azimuth cyclic encoding (0..360 deg)
    az = df.get("solar_azimuth", pd.Series(np.zeros(n))).values.astype(np.float32)
    az_rad = np.deg2rad(az)
    az_sin = np.sin(az_rad).astype(np.float32)
    az_cos = np.cos(az_rad).astype(np.float32)

    # --- Temporal encodings ---
    ts = pd.to_datetime(df["timestamp"])
    hour_frac = (ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0).values
    hour_sin = np.sin(2 * np.pi * hour_frac / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour_frac / 24).astype(np.float32)

    doy = ts.dt.dayofyear.values
    doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # Time since sunrise (normalized): use zenith = 90 as sunrise/sunset proxy
    # For simplicity: use hour normalized within daytime (6 AM to 6 PM MST ~)
    time_frac = np.clip((hour_frac - 6.0) / 12.0, 0, 1).astype(np.float32)

    # --- kt trends + variability ---
    kt = df["clear_sky_index"].values.astype(np.float32)
    def trend(series, lag):
        out = np.zeros_like(series)
        out[lag:] = (series[lag:] - series[:-lag]) / lag
        return out
    kt_trend_1min = trend(kt, 6)        # 6 steps × 10s = 60s
    kt_trend_5min = trend(kt, 30)
    kt_trend_10min = trend(kt, 60)

    def rolling_std(series, window):
        out = np.zeros_like(series)
        for i in range(window, len(series)):
            out[i] = np.std(series[i - window:i])
        return out
    kt_std_5min  = rolling_std(kt, 30).astype(np.float32)
    ghi = df["ghi"].values.astype(np.float32)
    ghi_std_1min = rolling_std(ghi, 6).astype(np.float32) / 1200.0  # normalized

    # --- Raw pyranometer millivolts (if present) ---
    if "millivolts" in df.columns:
        pyr_mv = df["millivolts"].values.astype(np.float32)
    else:
        pyr_mv = np.zeros(n, dtype=np.float32)

    out = pd.DataFrame({
        "air_mass":          air_mass,
        "zenith_rate":       zenith_rate,
        "azimuth_sin":       az_sin,
        "azimuth_cos":       az_cos,
        "hour_sin":          hour_sin,
        "hour_cos":          hour_cos,
        "doy_sin":           doy_sin,
        "doy_cos":           doy_cos,
        "time_frac":         time_frac,
        "kt_trend_1min":     kt_trend_1min,
        "kt_trend_5min":     kt_trend_5min,
        "kt_trend_10min":    kt_trend_10min,
        "kt_std_5min":       kt_std_5min,
        "ghi_std_1min":      ghi_std_1min,
        "pyr_mv":            pyr_mv,
    })
    return out


def main():
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
        df = df[df["image_exists"]].reset_index(drop=True) if "image_exists" in df.columns else df

        feats = compute_physics(df)
        arr = feats.values.astype(np.float32)
        out = LATENT_DIR / f"{split}_physics_features.npy"
        np.save(out, arr)

        print(f"  {split}: {arr.shape}  columns={list(feats.columns)}")
        print(f"    per-feature [mean, std]:")
        for col in feats.columns:
            v = feats[col].values
            print(f"      {col:20s}  [{v.mean():.4f}, {v.std():.4f}]")


if __name__ == "__main__":
    main()
