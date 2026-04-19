#!/usr/bin/env python3
"""Add kt (clear-sky index) and ghi_clearsky arrays to the latents directory.
These are needed for the SolarSDE v2 that predicts kt instead of raw GHI.
Existing latent files are preserved; just adds two new npy files per split.
"""

import numpy as np, pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_DIR / "colab_outputs" / "splits"
LATENT_DIR = PROJECT_DIR / "colab_outputs" / "latents"

for split in ["train", "val", "test"]:
    df = pd.read_parquet(SPLITS_DIR / f"{split}.parquet")
    df = df[df["image_exists"]].reset_index(drop=True)
    kt  = df["clear_sky_index"].values.astype(np.float32)
    gcs = df["ghi_clearsky"].values.astype(np.float32)

    # Sanity: existing ghi array should match df["ghi"]
    existing_ghi = np.load(LATENT_DIR / f"{split}_ghi.npy")
    df_ghi = df["ghi"].values.astype(np.float32)
    max_err = float(np.abs(existing_ghi - df_ghi).max())

    np.save(LATENT_DIR / f"{split}_kt.npy", kt)
    np.save(LATENT_DIR / f"{split}_ghi_clearsky.npy", gcs)

    print(f"  {split}: kt shape={kt.shape}, range=[{kt.min():.3f}, {kt.max():.3f}]  "
          f"ghi_clearsky range=[{gcs.min():.1f}, {gcs.max():.1f}]  "
          f"consistency_check_err={max_err:.4f}")

print("Done.")
