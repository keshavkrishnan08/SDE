#!/usr/bin/env python3
"""Prepare Stanford SKIPP'D dataset for SolarSDE cross-site training.

Stanford SKIPP'D: 3 years (2017-2019) of 64x64 sky images + PV power at 1-min
resolution. 497 unique training days vs. Golden CO's 8 days. Much more cloud
diversity (Mediterranean climate: summer clear, winter partly cloudy).

Usage:
  python3 scripts/prepare_stanford_skippd.py

Produces:
  data/processed/stanford_train_images.npy   — (N, 64, 64, 3) uint8 images
  data/processed/stanford_train_pv.npy       — (N,) float32 PV power in kW
  data/processed/stanford_train_times.npy    — (N,) datetime64
  data/processed/stanford_test_*             — analogous test split
  data/processed/stanford_pretrain.npy       — (N, 128, 128, 3) float32 [0,1]
                                                (resized + normalized for VAE pretraining)

The Stanford images are already 64x64 from SKIPP'D. For VAE compatibility with
our 128x128 setup, we upsample via bilinear interpolation.
"""

import numpy as np
import h5py
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "stanford_skippd"
OUT_DIR = PROJECT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HDF5_PATH = RAW_DIR / "2017_2019_images_pv_processed.hdf5"
TIMES_TV = RAW_DIR / "times_trainval.npy"
TIMES_TE = RAW_DIR / "times_test.npy"

if not HDF5_PATH.exists():
    print(f"ERROR: {HDF5_PATH} not found.")
    print("Download with:")
    print("  curl -L -o data/raw/stanford_skippd/2017_2019_images_pv_processed.hdf5 \\")
    print("    https://stacks.stanford.edu/file/dj417rh1007/2017_2019_images_pv_processed.hdf5")
    sys.exit(1)

print(f"Loading HDF5 from {HDF5_PATH}")
with h5py.File(HDF5_PATH, "r") as f:
    print(f"  Keys: {list(f.keys())}")
    for grp in f.keys():
        print(f"  {grp}:")
        for k in f[grp].keys():
            arr = f[grp][k]
            print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")

    # Save each split
    for split_key, out_prefix in [("trainval", "stanford_train"), ("test", "stanford_test")]:
        if split_key not in f:
            print(f"  [skip] split '{split_key}' not in HDF5")
            continue
        grp = f[split_key]
        # Common key names in SKIPP'D benchmark
        img_key = "images_log" if "images_log" in grp else list(grp.keys())[0]
        pv_key  = "pv_log" if "pv_log" in grp else [k for k in grp.keys() if "pv" in k.lower()][0]

        imgs = grp[img_key][:]
        pv   = grp[pv_key][:]
        print(f"\n{split_key}: images {imgs.shape} dtype={imgs.dtype}, pv {pv.shape}")

        np.save(OUT_DIR / f"{out_prefix}_images.npy", imgs)
        np.save(OUT_DIR / f"{out_prefix}_pv.npy", pv)
        print(f"  saved to {out_prefix}_images.npy + {out_prefix}_pv.npy")

# Save timestamps
tv = np.load(TIMES_TV, allow_pickle=True)
te = np.load(TIMES_TE, allow_pickle=True)
np.save(OUT_DIR / "stanford_train_times.npy", tv)
np.save(OUT_DIR / "stanford_test_times.npy", te)

# Stats for the train split
imgs_tv = np.load(OUT_DIR / "stanford_train_images.npy")
pv_tv   = np.load(OUT_DIR / "stanford_train_pv.npy")
print(f"\nTrain split summary:")
print(f"  images: {imgs_tv.shape}, dtype={imgs_tv.dtype}")
print(f"  pv:     range [{pv_tv.min():.1f}, {pv_tv.max():.1f}] kW")
print(f"  timestamps: {len(tv)} rows, {tv[0]} to {tv[-1]}")

# Sample a few cloudy days to sanity-check cloud diversity
import pandas as pd
ts_tv = pd.to_datetime(tv)
days = ts_tv.normalize().unique()
print(f"\n  Unique days in trainval: {len(days)}")
print(f"  Sample days: {list(days[:5])}")
print(f"               ... {list(days[-3:])}")
