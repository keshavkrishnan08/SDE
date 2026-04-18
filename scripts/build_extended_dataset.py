#!/usr/bin/env python3
"""Build the complete preprocessed dataset for SolarSDE.

Produces two data products:

1. IMAGE DATASET (for SolarSDE pipeline, needs sky images):
   - 8 days of CloudCV image-irradiance pairs
   - Used by Notebook 1 (VAE training) and all downstream notebooks
   - Output: data/processed/splits/{train,val,test}.parquet

2. EXTENDED TIME-SERIES DATASET (for baselines, no images needed):
   - All 90 days of BMS meteorological data
   - 12x more training data for baselines (LSTM, MC-Dropout, CSDI, Persistence)
   - Used by Notebook 3 (baselines) for a fairer comparison
   - Output: data/processed/extended/{train,val,test}.parquet

Runs on CPU in ~2-3 minutes. Outputs ~10 MB total, pushed to GitHub for
zero-setup Colab use.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SRRL = Location(latitude=39.742, longitude=-105.18, tz="America/Denver",
                altitude=1829, name="NREL SRRL")

DATA_DIR = PROJECT_DIR / "data"
CLOUDCV_DIR = DATA_DIR / "raw" / "cloudcv"
BMS_PATH = DATA_DIR / "raw" / "bms" / "bms_srrl_2019.csv"
OUT_DIR = DATA_DIR / "processed"


# ============================================================================
# BMS parsing
# ============================================================================

def parse_bms_full() -> pd.DataFrame:
    """Parse the full 90-day BMS CSV into clean time-aligned DataFrame."""
    log.info("Loading BMS 1-minute data (90 days, Sep 5 – Dec 3 2019) ...")
    raw = pd.read_csv(BMS_PATH)
    log.info(f"  Raw shape: {raw.shape}")

    # Build timestamps: MST column is HHMM (e.g., 1435 = 14:35)
    ts = []
    for _, r in raw.iterrows():
        try:
            y, doy, mst = int(r["Year"]), int(r["DOY"]), int(r["MST"])
            h, mi = divmod(mst, 100)
            dt = datetime.strptime(f"{y}-{doy}", "%Y-%j").replace(hour=h, minute=mi)
            ts.append(dt)
        except (ValueError, OverflowError):
            ts.append(pd.NaT)
    raw["timestamp"] = pd.to_datetime(ts)
    raw = raw.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Extract key variables (rename to clean names)
    df = pd.DataFrame({
        "timestamp":          raw["timestamp"],
        "ghi":                raw.get("Global LI-200 [W/m^2]"),
        "dni":                raw.get("Direct NIP [W/m^2]"),
        "dhi":                raw.get("Diffuse CM22-1 (vent/cor) [W/m^2]"),
        "temperature":        raw.get("Deck Dry Bulb Temp [deg C]"),
        "humidity":           raw.get("Deck RH [%]"),
        "wind_speed":         raw.get("Avg Wind Speed @ 19ft [m/s]"),
        "wind_direction":     raw.get("Avg Wind Direction @ 19ft [deg from N]"),
        "pressure":           raw.get("Station Pressure [mBar]"),
        "cloud_cover_total":  raw.get("Total Cloud Cover [%]"),
        "cloud_cover_opaque": raw.get("Opaque Cloud Cover [%]"),
    })
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(
                [-7999, -6999, -9999], np.nan
            )
    # Clip GHI to physical range
    df["ghi"] = df["ghi"].clip(lower=0, upper=1500)

    log.info(f"  Valid GHI: {df['ghi'].notna().sum():,} / {len(df):,}")
    log.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log.info(f"  Unique days: {df['timestamp'].dt.date.nunique()}")
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add solar zenith/azimuth and clear-sky index."""
    log.info("Computing solar geometry + clear-sky irradiance ...")
    # BMS and CloudCV are labeled UTC-7 (fixed MST, no DST). Use fixed offset to avoid
    # ambiguity at the 2019-11-03 fall-back transition.
    tz_ts = pd.DatetimeIndex(df["timestamp"]).tz_localize("Etc/GMT+7")
    sp = SRRL.get_solarposition(tz_ts)
    cs = SRRL.get_clearsky(tz_ts, model="ineichen")
    df = df.copy()
    df["solar_zenith"] = sp["apparent_zenith"].values
    df["solar_azimuth"] = sp["azimuth"].values
    df["ghi_clearsky"] = cs["ghi"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = df["ghi"].values / df["ghi_clearsky"].values
        kt = np.where(df["ghi_clearsky"].values < 10, 0.0, kt)
    df["clear_sky_index"] = np.clip(kt, 0.0, 1.5)
    return df


def filter_daytime(df: pd.DataFrame, zenith_max: float = 85.0) -> pd.DataFrame:
    before = len(df)
    out = df[
        (df["solar_zenith"] <= zenith_max)
        & (df["ghi"] >= 0)
        & (df["ghi"].notna())
    ].reset_index(drop=True)
    log.info(f"  Daytime filter: {before:,} -> {len(out):,}")
    return out


def add_ramp_labels(df: pd.DataFrame, dt_seconds: int = 60, threshold: float = 50.0) -> pd.DataFrame:
    """Ramp events: |ΔGHI| > threshold W/m² in dt_seconds window."""
    steps = max(1, dt_seconds // 60)  # BMS is 1-minute data
    df = df.copy()
    df["is_ramp"] = (df["ghi"].diff(steps).abs() > threshold).fillna(False)
    log.info(f"  Ramp events: {df['is_ramp'].sum():,} ({df['is_ramp'].mean()*100:.1f}%)")
    return df


# ============================================================================
# CloudCV image-irradiance alignment
# ============================================================================

def parse_ts(s: str) -> datetime:
    s = s.strip()
    if s.startswith("UTC-7_"):
        s = s[6:]
    date_p, time_p = s.split("-")
    y, mo, d = date_p.split("_")
    tp = time_p.split("_")
    h, mi, sec = tp[0], tp[1], tp[2]
    us = tp[3] if len(tp) > 3 else "0"
    return datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec), int(us))


def load_cloudcv_day(day_dir: Path) -> pd.DataFrame:
    csv = day_dir / "pyranometer.csv"
    imgs = day_dir / "images"
    if not csv.exists():
        return pd.DataFrame()
    rows = []
    for line in open(csv):
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            ts = parse_ts(parts[0])
        except Exception:
            continue
        img_name = parts[0].strip() + ".jpg"
        img_path = imgs / img_name
        rows.append({
            "timestamp": ts,
            "image_path": str(img_path),
            "image_exists": img_path.exists(),
        })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def build_image_dataset(bms_full: pd.DataFrame) -> pd.DataFrame:
    """8 days of image-irradiance pairs, GHI interpolated from BMS."""
    log.info("Building image dataset from 8 CloudCV days ...")

    day_dirs = sorted([d for d in CLOUDCV_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("2019")])
    all_dfs = []
    for d in day_dirs:
        df = load_cloudcv_day(d)
        if len(df) > 0:
            all_dfs.append(df)
            log.info(f"  {d.name}: {len(df)} rows ({df['image_exists'].sum()} images)")

    cloudcv = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    log.info(f"  Total: {len(cloudcv)} CloudCV timestamps")

    # Interpolate BMS GHI + covariates to 10s CloudCV timestamps
    log.info("  Interpolating BMS to CloudCV timestamps (10s) ...")
    bms_idx = bms_full.set_index("timestamp").sort_index()
    cols_to_interp = ["ghi", "temperature", "humidity", "wind_speed",
                      "pressure", "cloud_cover_total"]
    bms_10s = bms_idx[cols_to_interp].resample("10s").interpolate(method="linear")

    cloudcv["ts_round"] = cloudcv["timestamp"].dt.round("10s")
    merged_vals = bms_10s.reindex(cloudcv["ts_round"].values, method="nearest")
    for col in cols_to_interp:
        cloudcv[col] = merged_vals[col].values

    # Add solar features from CloudCV timestamps (not BMS)
    tz_ts = pd.DatetimeIndex(cloudcv["timestamp"]).tz_localize("America/Denver")
    sp = SRRL.get_solarposition(tz_ts)
    cs = SRRL.get_clearsky(tz_ts, model="ineichen")
    cloudcv["solar_zenith"] = sp["apparent_zenith"].values
    cloudcv["solar_azimuth"] = sp["azimuth"].values
    cloudcv["ghi_clearsky"] = cs["ghi"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = cloudcv["ghi"].values / cloudcv["ghi_clearsky"].values
        kt = np.where(cloudcv["ghi_clearsky"].values < 10, 0.0, kt)
    cloudcv["clear_sky_index"] = np.clip(kt, 0.0, 1.5)

    # Quality filter
    before = len(cloudcv)
    cloudcv = cloudcv[
        (cloudcv["solar_zenith"] <= 85.0)
        & (cloudcv["ghi"] >= 0)
        & (cloudcv["ghi"].notna())
        & (cloudcv["image_exists"])
    ].reset_index(drop=True)
    log.info(f"  Filter: {before:,} -> {len(cloudcv):,}")

    # Ramp events (10s resolution → 6 steps = 60s window)
    cloudcv["is_ramp"] = (cloudcv["ghi"].diff(6).abs() > 50.0).fillna(False)
    log.info(f"  Ramp events: {cloudcv['is_ramp'].sum():,} ({cloudcv['is_ramp'].mean()*100:.2f}%)")

    return cloudcv


# ============================================================================
# Split builders
# ============================================================================

def chronological_split(df: pd.DataFrame, frac_train=0.625, frac_val=0.125):
    dates = sorted(df["timestamp"].dt.date.unique())
    n = len(dates)
    n_tr = max(1, int(n * frac_train))
    n_val = max(1, int(n * frac_val))
    tr = set(dates[:n_tr])
    vl = set(dates[n_tr:n_tr + n_val])
    te = set(dates[n_tr + n_val:])
    train = df[df["timestamp"].dt.date.isin(tr)].reset_index(drop=True)
    val = df[df["timestamp"].dt.date.isin(vl)].reset_index(drop=True)
    test = df[df["timestamp"].dt.date.isin(te)].reset_index(drop=True)
    return train, val, test, (len(tr), len(vl), len(te))


# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 70)
    log.info("Building extended SolarSDE dataset")
    log.info("=" * 70)

    # 1) Load and enrich BMS (90 days)
    bms = parse_bms_full()
    bms = add_solar_features(bms)
    bms_day = filter_daytime(bms)
    bms_day = add_ramp_labels(bms_day, dt_seconds=60, threshold=50.0)

    # 2) Chronological split for extended time-series dataset
    tr90, va90, te90, (dtr, dva, dte) = chronological_split(bms_day)
    ext_dir = OUT_DIR / "extended"
    ext_dir.mkdir(parents=True, exist_ok=True)
    tr90.to_parquet(ext_dir / "train.parquet")
    va90.to_parquet(ext_dir / "val.parquet")
    te90.to_parquet(ext_dir / "test.parquet")
    log.info(f"\nEXTENDED (90-day BMS, no images):")
    log.info(f"  train: {len(tr90):>6} rows ({dtr} days)")
    log.info(f"  val:   {len(va90):>6} rows ({dva} days)")
    log.info(f"  test:  {len(te90):>6} rows ({dte} days)")

    # 3) Build image dataset (8 days with CloudCV imagery)
    log.info("\n" + "=" * 70)
    log.info("Building IMAGE dataset (8 CloudCV days)")
    log.info("=" * 70)
    img = build_image_dataset(bms)
    tr8, va8, te8, (dtr, dva, dte) = chronological_split(img)
    img_dir = OUT_DIR / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    tr8.to_parquet(img_dir / "train.parquet")
    va8.to_parquet(img_dir / "val.parquet")
    te8.to_parquet(img_dir / "test.parquet")
    log.info(f"\nIMAGE (8-day CloudCV, with images):")
    log.info(f"  train: {len(tr8):>6} rows ({dtr} days)")
    log.info(f"  val:   {len(va8):>6} rows ({dva} days)")
    log.info(f"  test:  {len(te8):>6} rows ({dte} days)")

    # 4) Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  90-day BMS coverage: {len(bms_day):,} valid-daytime rows across "
             f"{bms_day['timestamp'].dt.date.nunique()} days")
    log.info(f"   8-day CloudCV coverage: {len(img):,} image-irradiance pairs")
    log.info(f"  Output dirs:")
    log.info(f"    data/processed/splits/    (image-based, for Notebook 1/2/4/5)")
    log.info(f"    data/processed/extended/  (90-day time-series, for Notebook 3 baselines)")
    log.info("Done.")


if __name__ == "__main__":
    main()
