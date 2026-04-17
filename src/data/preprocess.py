"""Data preprocessing: alignment, normalization, splitting, sequence creation."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.clear_sky import (
    compute_clear_sky,
    compute_clear_sky_index,
    compute_solar_position,
    SRRL_LOCATION,
)
from src.data.ramp_labels import detect_ramp_events

logger = logging.getLogger(__name__)


def load_cloudcv_irradiance(cloudcv_dir: Path) -> pd.DataFrame:
    """Load irradiance CSV from the CloudCV dataset.

    Searches for CSV files containing irradiance measurements and parses timestamps.
    """
    csv_files = list(cloudcv_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cloudcv_dir}")

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, parse_dates=[0])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not parse {csv_path}: {e}")

    if not dfs:
        raise ValueError("No valid irradiance CSVs found")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(combined.columns[0]).reset_index(drop=True)
    return combined


def crop_fisheye_circle(
    image: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """Crop a fisheye image to its circular region and resize.

    Masks pixels outside the inscribed circle to black, then resizes.
    """
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(center_x, center_y)

    # Create circular mask
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist <= radius

    # Apply mask
    if image.ndim == 3:
        masked = image * mask[:, :, np.newaxis]
    else:
        masked = image * mask

    # Crop to bounding box of circle
    y1, y2 = center_y - radius, center_y + radius
    x1, x2 = center_x - radius, center_x + radius
    cropped = masked[y1:y2, x1:x2]

    # Resize
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil_img)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] float32."""
    return image.astype(np.float32) / 255.0


def preprocess_images(
    image_dir: Path,
    output_dir: Path,
    target_size: int = 256,
) -> list[Path]:
    """Process all sky images: crop fisheye circle, resize, normalize, save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.jpeg"))
    image_paths += sorted(image_dir.rglob("*.png"))

    processed_paths = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = np.array(Image.open(img_path).convert("RGB"))
        img = crop_fisheye_circle(img, target_size)
        img = normalize_image(img)

        out_path = output_dir / f"{img_path.stem}.npy"
        np.save(out_path, img)
        processed_paths.append(out_path)

    logger.info(f"Processed {len(processed_paths)} images")
    return processed_paths


def align_images_irradiance(
    image_dir: Path,
    irradiance_df: pd.DataFrame,
    tolerance_seconds: int = 5,
) -> pd.DataFrame:
    """Align image timestamps with irradiance measurements.

    Returns a DataFrame with columns: timestamp, image_path, ghi, and metadata.
    """
    image_paths = sorted(image_dir.rglob("*.npy"))

    # Extract timestamps from filenames (format varies by dataset)
    records = []
    for img_path in image_paths:
        # Try to parse timestamp from filename
        stem = img_path.stem
        try:
            ts = pd.to_datetime(stem, format="mixed")
        except (ValueError, TypeError):
            # Fall back: use index ordering
            continue
        records.append({"timestamp": ts, "image_path": str(img_path)})

    if not records:
        # If timestamp parsing fails, create sequential alignment
        logger.warning("Could not parse timestamps from filenames. Using sequential alignment.")
        timestamps = irradiance_df.iloc[:, 0]
        n = min(len(image_paths), len(timestamps))
        for i in range(n):
            records.append({
                "timestamp": timestamps.iloc[i],
                "image_path": str(image_paths[i]),
            })

    aligned = pd.DataFrame(records)
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"])
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)
    return aligned


def compute_features(
    aligned_df: pd.DataFrame,
    location=SRRL_LOCATION,
) -> pd.DataFrame:
    """Add solar geometry, clear-sky, and clear-sky index features."""
    timestamps = pd.DatetimeIndex(aligned_df["timestamp"])

    # Solar position
    solpos = compute_solar_position(timestamps, location)
    aligned_df["solar_zenith"] = solpos["apparent_zenith"].values
    aligned_df["solar_azimuth"] = solpos["azimuth"].values

    # Clear-sky irradiance
    clearsky = compute_clear_sky(timestamps, location)
    aligned_df["ghi_clearsky"] = clearsky["ghi"].values

    # Clear-sky index
    if "ghi" in aligned_df.columns:
        aligned_df["clear_sky_index"] = compute_clear_sky_index(
            aligned_df["ghi"], timestamps, location
        ).values

    return aligned_df


def filter_quality(
    df: pd.DataFrame,
    zenith_max: float = 85.0,
) -> pd.DataFrame:
    """Remove nighttime data and quality-flagged rows."""
    mask = df["solar_zenith"] <= zenith_max
    # Remove rows with NaN GHI or negative GHI
    if "ghi" in df.columns:
        mask &= df["ghi"].notna() & (df["ghi"] >= 0)
    # Remove rows with extreme clear-sky index
    if "clear_sky_index" in df.columns:
        mask &= df["clear_sky_index"] <= 1.5
    filtered = df.loc[mask].reset_index(drop=True)
    logger.info(f"Quality filter: {len(df)} -> {len(filtered)} rows")
    return filtered


def create_chronological_splits(
    df: pd.DataFrame,
    train_days: int = 60,
    val_days: int = 15,
    test_days: int = 15,
    output_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test sets (no shuffling)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    dates = df["timestamp"].dt.date.unique()

    total_days = len(dates)
    if total_days < train_days + val_days + test_days:
        logger.warning(
            f"Only {total_days} days available, adjusting split proportions"
        )
        train_frac = train_days / (train_days + val_days + test_days)
        val_frac = val_days / (train_days + val_days + test_days)
        train_days = int(total_days * train_frac)
        val_days = int(total_days * val_frac)
        test_days = total_days - train_days - val_days

    train_dates = dates[:train_days]
    val_dates = dates[train_days : train_days + val_days]
    test_dates = dates[train_days + val_days : train_days + val_days + test_days]

    train_df = df[df["timestamp"].dt.date.isin(train_dates)].reset_index(drop=True)
    val_df = df[df["timestamp"].dt.date.isin(val_dates)].reset_index(drop=True)
    test_df = df[df["timestamp"].dt.date.isin(test_dates)].reset_index(drop=True)

    logger.info(
        f"Split: train={len(train_df)} ({train_days}d), "
        f"val={len(val_df)} ({val_days}d), "
        f"test={len(test_df)} ({test_days}d)"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(output_dir / "train.parquet")
        val_df.to_parquet(output_dir / "val.parquet")
        test_df.to_parquet(output_dir / "test.parquet")

    return train_df, val_df, test_df


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 30,
    forecast_horizons: list[int] | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Create sliding window sequences for training.

    Each sequence contains:
      - input: seq_len past timesteps of (image_path, ghi, covariates)
      - targets: GHI values at each forecast horizon

    Args:
        df: Aligned DataFrame with all features.
        seq_len: Number of past timesteps.
        forecast_horizons: List of future steps to predict (in 10s increments).
        output_dir: Optional directory to save sequences.

    Returns:
        List of sequence dictionaries.
    """
    if forecast_horizons is None:
        forecast_horizons = [6, 12, 30, 60, 90, 120, 180]

    max_horizon = max(forecast_horizons)
    sequences = []

    for i in range(seq_len, len(df) - max_horizon):
        seq = {
            "input_indices": list(range(i - seq_len, i)),
            "target_indices": {h: i + h for h in forecast_horizons},
            "input_start": i - seq_len,
            "input_end": i,
        }
        sequences.append(seq)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "sequences.npy", sequences, allow_pickle=True)

    logger.info(f"Created {len(sequences)} sequences (seq_len={seq_len})")
    return sequences


def run_full_preprocessing(config: dict) -> None:
    """Run the complete preprocessing pipeline from config."""
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    metadata_dir = Path(data_cfg["metadata_dir"])

    # Step 1: Load irradiance data
    logger.info("Step 1: Loading irradiance data")
    irradiance_df = load_cloudcv_irradiance(raw_dir / "cloudcv")

    # Step 2: Process images
    logger.info("Step 2: Processing images")
    preprocess_images(
        raw_dir / "cloudcv",
        processed_dir / "aligned",
        target_size=data_cfg["image_size"],
    )

    # Step 3: Align images and irradiance
    logger.info("Step 3: Aligning images and irradiance")
    aligned_df = align_images_irradiance(
        processed_dir / "aligned",
        irradiance_df,
    )

    # Step 4: Compute features (solar geometry, clear-sky, clear-sky index)
    logger.info("Step 4: Computing features")
    aligned_df = compute_features(aligned_df)

    # Step 5: Quality filtering
    logger.info("Step 5: Quality filtering")
    filtered_df = filter_quality(aligned_df, data_cfg["solar_zenith_max"])

    # Step 6: Detect ramp events
    logger.info("Step 6: Detecting ramp events")
    if "ghi" in filtered_df.columns:
        filtered_df["is_ramp"] = detect_ramp_events(
            filtered_df["ghi"],
            threshold=data_cfg["ramp_threshold"],
        ).values

    # Step 7: Save solar geometry metadata
    metadata_dir.mkdir(parents=True, exist_ok=True)
    filtered_df[["timestamp", "solar_zenith", "solar_azimuth"]].to_csv(
        metadata_dir / "solar_geometry.csv", index=False
    )

    # Step 8: Create train/val/test splits
    logger.info("Step 8: Creating splits")
    create_chronological_splits(
        filtered_df,
        train_days=data_cfg["train_days"],
        val_days=data_cfg["val_days"],
        test_days=data_cfg["test_days"],
        output_dir=processed_dir / "splits",
    )

    # Step 9: Create sequences
    logger.info("Step 9: Creating sequences")
    create_sequences(
        filtered_df,
        seq_len=data_cfg["sequence_length"],
        forecast_horizons=data_cfg["forecast_horizons"],
        output_dir=processed_dir / "sequences",
    )

    logger.info("Preprocessing complete!")
