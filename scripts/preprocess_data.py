#!/usr/bin/env python3
"""Preprocess downloaded CloudCV and BMS data for SolarSDE.

Handles:
1. Parsing CloudCV pyranometer CSVs (millivolt → W/m²)
2. Parsing BMS 1-minute meteorological data
3. Aligning image timestamps with irradiance
4. Computing clear-sky irradiance and clear-sky index
5. Solar zenith filtering
6. Creating train/val/test splits
7. Creating sliding window sequences
8. Ramp event labeling
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.clear_sky import compute_solar_position, compute_clear_sky_index, SRRL_LOCATION
from src.data.ramp_labels import detect_ramp_events, ramp_event_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_DIR / "data"

# LI-200 pyranometer calibration: sensitivity ~5 µV per W/m²
# Reading is in millivolts, so: GHI (W/m²) = mV / 0.005 = mV * 200
# Typical calibration constant for LI-200: ~80-100 µA per 1000 W/m²
# With a 604 ohm shunt: V = I * R, so 0.1mA * 604 = 0.0604V = 60.4 mV at 1000 W/m²
# So: GHI ≈ mV * (1000/60.4) ≈ mV * 16.56
# However, the ADS1115 ADC may output in different units.
# Let's calibrate against the BMS LI-200 readings for the overlapping period.
LI200_MV_TO_WM2_DEFAULT = 200.0  # Will be calibrated


def parse_cloudcv_timestamp(ts_str: str) -> datetime:
    """Parse CloudCV timestamp format: UTC-7_YYYY_MM_DD-HH_MM_SS_UUUUUU"""
    ts_str = ts_str.strip()
    # Remove UTC-7_ prefix
    if ts_str.startswith("UTC-7_"):
        ts_str = ts_str[6:]
    # Parse: YYYY_MM_DD-HH_MM_SS_UUUUUU
    parts = ts_str.split("-")
    date_part = parts[0]  # YYYY_MM_DD
    time_part = parts[1]  # HH_MM_SS_UUUUUU

    y, mo, d = date_part.split("_")
    time_fields = time_part.split("_")
    h, mi, s = time_fields[0], time_fields[1], time_fields[2]
    us = time_fields[3] if len(time_fields) > 3 else "0"

    return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int(us))


def load_cloudcv_day(day_dir: Path) -> pd.DataFrame:
    """Load one day of CloudCV data (images + pyranometer CSV)."""
    csv_path = day_dir / "pyranometer.csv"
    images_dir = day_dir / "images"

    if not csv_path.exists():
        logger.warning(f"No pyranometer.csv in {day_dir}")
        return pd.DataFrame()

    # Parse CSV: no header, columns are (timestamp_string, millivolts)
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            ts_str = parts[0].strip()
            mv = float(parts[1].strip())

            try:
                ts = parse_cloudcv_timestamp(ts_str)
            except (ValueError, IndexError):
                continue

            # Build image path
            img_name = ts_str.strip() + ".jpg"
            img_path = images_dir / img_name

            rows.append({
                "timestamp": ts,
                "millivolts": mv,
                "image_path": str(img_path) if img_path.exists() else None,
                "image_exists": img_path.exists(),
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_all_cloudcv(cloudcv_dir: Path) -> pd.DataFrame:
    """Load all available CloudCV days."""
    day_dirs = sorted([d for d in cloudcv_dir.iterdir()
                       if d.is_dir() and d.name.startswith("2019")])

    all_dfs = []
    for day_dir in day_dirs:
        df = load_cloudcv_day(day_dir)
        if len(df) > 0:
            logger.info(f"  {day_dir.name}: {len(df)} rows, "
                       f"{df['image_exists'].sum()} images found")
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No CloudCV data found")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Total CloudCV: {len(combined)} rows across {len(all_dfs)} days")
    return combined


def load_bms_data(bms_path: Path) -> pd.DataFrame:
    """Load and parse BMS 1-minute meteorological data."""
    df = pd.read_csv(bms_path)

    # Build timestamps from Year, DOY, MST columns
    # MST is minutes since midnight (0-1439)
    timestamps = []
    for _, row in df.iterrows():
        year = int(row["Year"])
        doy = int(row["DOY"])
        mst = int(row["MST"])
        hour = mst // 60
        minute = mst % 60
        try:
            dt = datetime.strptime(f"{year}-{doy}", "%Y-%j")
            dt = dt.replace(hour=hour, minute=minute)
            timestamps.append(dt)
        except ValueError:
            timestamps.append(pd.NaT)

    df["timestamp"] = pd.to_datetime(timestamps)

    # Extract key variables with clean column names
    bms_clean = pd.DataFrame({
        "timestamp": df["timestamp"],
        "ghi_bms": df.get("Global LI-200 [W/m^2]", pd.Series(dtype=float)),
        "dni_bms": df.get("Direct NIP [W/m^2]", pd.Series(dtype=float)),
        "dhi_bms": df.get("Diffuse CM22-1 (vent/cor) [W/m^2]", pd.Series(dtype=float)),
        "temperature": df.get("Deck Dry Bulb Temp [deg C]", pd.Series(dtype=float)),
        "humidity": df.get("Deck RH [%]", pd.Series(dtype=float)),
        "wind_speed": df.get("Avg Wind Speed @ 19ft [m/s]", pd.Series(dtype=float)),
        "wind_direction": df.get("Avg Wind Direction @ 19ft [deg from N]", pd.Series(dtype=float)),
        "pressure": df.get("Station Pressure [mBar]", pd.Series(dtype=float)),
        "cloud_cover_opaque": df.get("Opaque Cloud Cover [%]", pd.Series(dtype=float)),
        "cloud_cover_total": df.get("Total Cloud Cover [%]", pd.Series(dtype=float)),
        "zenith_bms": df.get("Zenith Angle [degrees]", pd.Series(dtype=float)),
    })

    # Replace missing value codes (-7999, -6999) with NaN
    for col in bms_clean.columns:
        if col != "timestamp":
            bms_clean[col] = bms_clean[col].replace([-7999, -6999, -9999], np.nan)

    logger.info(f"BMS data: {len(bms_clean)} rows, "
               f"GHI range: [{bms_clean['ghi_bms'].min():.1f}, {bms_clean['ghi_bms'].max():.1f}] W/m²")
    return bms_clean


def interpolate_bms_to_10sec(cloudcv_df: pd.DataFrame, bms_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate BMS 1-minute GHI to 10-second CloudCV timestamps.

    Uses BMS LI-200 GHI as calibrated ground truth instead of uncalibrated pyranometer mV.
    """
    bms_ghi = bms_df[["timestamp", "ghi_bms"]].dropna().copy()
    bms_ghi = bms_ghi.set_index("timestamp").sort_index()

    # Resample BMS to 10-second resolution via linear interpolation
    bms_10s = bms_ghi.resample("10s").interpolate(method="linear")

    # Match to CloudCV timestamps
    cloudcv_df = cloudcv_df.copy()
    cloudcv_df["ts_round"] = cloudcv_df["timestamp"].dt.round("10s")

    ghi_values = []
    for ts in cloudcv_df["ts_round"]:
        if ts in bms_10s.index:
            ghi_values.append(float(bms_10s.loc[ts, "ghi_bms"]))
        else:
            # Find nearest
            idx = bms_10s.index.get_indexer([ts], method="nearest")[0]
            if idx >= 0 and idx < len(bms_10s):
                ghi_values.append(float(bms_10s.iloc[idx]["ghi_bms"]))
            else:
                ghi_values.append(np.nan)

    cloudcv_df["ghi"] = ghi_values
    matched = cloudcv_df["ghi"].notna().sum()
    logger.info(f"  Interpolated BMS GHI to {matched}/{len(cloudcv_df)} CloudCV timestamps")
    logger.info(f"  GHI range: [{cloudcv_df['ghi'].min():.1f}, {cloudcv_df['ghi'].max():.1f}] W/m²")
    return cloudcv_df


def process_images(
    df: pd.DataFrame,
    output_dir: Path,
    target_size: int = 256,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Crop fisheye images, resize to target_size, save as numpy arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        if not row.get("image_exists", False) or row["image_path"] is None:
            processed_paths.append(None)
            continue

        out_name = Path(row["image_path"]).stem + ".npy"
        out_path = output_dir / out_name

        if skip_existing and out_path.exists():
            processed_paths.append(str(out_path))
            continue

        try:
            img = Image.open(row["image_path"]).convert("RGB")
            # The fisheye image is 1920x1080, crop to center square then resize
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            np.save(out_path, arr)
            processed_paths.append(str(out_path))
        except Exception as e:
            logger.warning(f"Error processing {row['image_path']}: {e}")
            processed_paths.append(None)

    df["processed_image_path"] = processed_paths
    valid = sum(1 for p in processed_paths if p is not None)
    logger.info(f"Processed {valid}/{len(df)} images -> {output_dir}")
    return df


def run_preprocessing():
    """Full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("SolarSDE Data Preprocessing Pipeline")
    logger.info("=" * 60)

    cloudcv_dir = DATA_DIR / "raw" / "cloudcv"
    bms_path = DATA_DIR / "raw" / "bms" / "bms_srrl_2019.csv"
    processed_dir = DATA_DIR / "processed"
    metadata_dir = DATA_DIR / "metadata"

    # Step 1: Load CloudCV data
    logger.info("\n[Step 1] Loading CloudCV data...")
    cloudcv_df = load_all_cloudcv(cloudcv_dir)

    # Step 2: Load BMS data
    logger.info("\n[Step 2] Loading BMS data...")
    bms_df = load_bms_data(bms_path)

    # Step 3: Interpolate BMS GHI to 10-second CloudCV timestamps
    logger.info("\n[Step 3] Interpolating BMS GHI to CloudCV timestamps...")
    cloudcv_df = interpolate_bms_to_10sec(cloudcv_df, bms_df)
    cloudcv_df["ghi"] = cloudcv_df["ghi"].clip(lower=0)

    # Step 4: Merge with BMS meteorological data
    logger.info("\n[Step 4] Merging with BMS meteorological data...")
    cloudcv_df["ts_minute"] = cloudcv_df["timestamp"].dt.floor("min")
    bms_df["ts_minute"] = bms_df["timestamp"].dt.floor("min")

    merged = cloudcv_df.merge(
        bms_df.drop(columns=["timestamp"]),
        on="ts_minute",
        how="left",
    )
    # Forward-fill BMS data (1-min resolution vs 10-sec CloudCV)
    bms_cols = ["temperature", "humidity", "wind_speed", "wind_direction",
                "pressure", "cloud_cover_opaque", "cloud_cover_total", "ghi_bms"]
    for col in bms_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    logger.info(f"  Merged: {len(merged)} rows with meteorological data")

    # Step 5: Compute solar position
    logger.info("\n[Step 5] Computing solar geometry...")
    timestamps_tz = pd.DatetimeIndex(merged["timestamp"]).tz_localize("America/Denver")
    solpos = compute_solar_position(timestamps_tz)
    merged["solar_zenith"] = solpos["apparent_zenith"].values
    merged["solar_azimuth"] = solpos["azimuth"].values

    # Step 6: Compute clear-sky irradiance and index
    logger.info("\n[Step 6] Computing clear-sky irradiance...")
    from src.data.clear_sky import compute_clear_sky
    clearsky = compute_clear_sky(timestamps_tz)
    merged["ghi_clearsky"] = clearsky["ghi"].values

    # Clear-sky index
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = merged["ghi"].values / merged["ghi_clearsky"].values
        kt = np.where(merged["ghi_clearsky"].values < 10, 0.0, kt)
        kt = np.clip(kt, 0.0, 1.5)
    merged["clear_sky_index"] = kt

    # Step 7: Filter quality
    logger.info("\n[Step 7] Quality filtering...")
    n_before = len(merged)
    mask = (
        (merged["solar_zenith"] <= 85.0) &  # Daytime only
        (merged["ghi"] >= 0) &               # Non-negative GHI
        (merged["ghi"].notna()) &            # No missing GHI
        (merged["image_exists"])              # Image exists
    )
    filtered = merged[mask].reset_index(drop=True)
    logger.info(f"  Filtered: {n_before} -> {len(filtered)} rows "
               f"(removed {n_before - len(filtered)} nighttime/missing)")

    # Step 8: Detect ramp events
    logger.info("\n[Step 8] Detecting ramp events...")
    filtered["is_ramp"] = detect_ramp_events(
        filtered["ghi"], threshold=50.0, window_seconds=60, dt_seconds=10
    ).values
    stats = ramp_event_statistics(filtered["ghi"], threshold=50.0)
    logger.info(f"  Ramp events: {stats['ramp_timesteps']}/{stats['total_timesteps']} "
               f"({stats['ramp_fraction']:.1%})")

    # Step 9: Store image paths (raw JPEGs, processed lazily at training time)
    logger.info("\n[Step 9] Verifying image paths...")
    filtered["processed_image_path"] = filtered["image_path"]
    valid_images = filtered["image_exists"].sum()
    logger.info(f"  {valid_images}/{len(filtered)} images available")

    # Step 10: Save solar geometry metadata
    logger.info("\n[Step 10] Saving metadata...")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    filtered[["timestamp", "solar_zenith", "solar_azimuth"]].to_csv(
        metadata_dir / "solar_geometry.csv", index=False
    )

    # Step 11: Create train/val/test splits
    logger.info("\n[Step 11] Creating chronological splits...")
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted(filtered["timestamp"].dt.date.unique())
    n_dates = len(dates)
    logger.info(f"  Total unique dates: {n_dates}")

    # For 8 days: 5 train / 1 val / 2 test
    train_days = max(1, int(n_dates * 0.625))
    val_days = max(1, int(n_dates * 0.125))
    test_days = n_dates - train_days - val_days

    train_dates = set(dates[:train_days])
    val_dates = set(dates[train_days:train_days + val_days])
    test_dates = set(dates[train_days + val_days:])

    train_df = filtered[filtered["timestamp"].dt.date.isin(train_dates)].reset_index(drop=True)
    val_df = filtered[filtered["timestamp"].dt.date.isin(val_dates)].reset_index(drop=True)
    test_df = filtered[filtered["timestamp"].dt.date.isin(test_dates)].reset_index(drop=True)

    # Save as parquet
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = splits_dir / f"{name}.parquet"
        split_df.to_parquet(path)
        logger.info(f"  {name}: {len(split_df)} rows ({len(split_df['timestamp'].dt.date.unique())} days) -> {path}")

    # Step 12: Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Summary")
    logger.info("=" * 60)
    logger.info(f"  Total samples: {len(filtered)}")
    logger.info(f"  Date range: {filtered['timestamp'].min()} to {filtered['timestamp'].max()}")
    logger.info(f"  GHI range: [{filtered['ghi'].min():.1f}, {filtered['ghi'].max():.1f}] W/m²")
    logger.info(f"  Clear-sky index range: [{filtered['clear_sky_index'].min():.2f}, "
               f"{filtered['clear_sky_index'].max():.2f}]")
    logger.info(f"  Ramp events: {stats['ramp_fraction']:.1%}")
    logger.info(f"  Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    logger.info(f"  Images processed: {(filtered['processed_image_path'].notna()).sum()}")
    logger.info("  Done!")


if __name__ == "__main__":
    run_preprocessing()
