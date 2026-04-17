#!/usr/bin/env python3
"""Download all available CloudCV and BMS data for SolarSDE.

CloudCV: 8 days of 10-second sky images + irradiance from NREL OEDI.
BMS: Full 90-day period of 1-minute meteorological data from NREL SRRL MIDC.
"""

import os
import sys
import tarfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ============================================================
# CloudCV Dataset (8 available days from NREL OEDI)
# ============================================================

CLOUDCV_FILES = {
    "README.md": "https://data.nlr.gov/system/files/248/1727758900-README.md",
    "2019_09_07.tar.gz": "https://data.nlr.gov/system/files/248/1727737056-2019_09_07.tar.gz",
    "2019_09_08.tar.gz": "https://data.nlr.gov/system/files/248/1727737056-2019_09_08.tar.gz",
    "2019_09_14.tar.gz": "https://data.nlr.gov/system/files/248/1727737056-2019_09_14.tar.gz",
    "2019_09_15.tar.gz": "https://data.nlr.gov/system/files/248/1727737056-2019_09_15.tar.gz",
    "2019_09_21.tar.gz": "https://data.nlr.gov/system/files/248/1727737586-2019_09_21.tar.gz",
    "2019_09_22.tar.gz": "https://data.nlr.gov/system/files/248/1727737586-2019_09_22.tar.gz",
    "2019_09_28.tar.gz": "https://data.nlr.gov/system/files/248/1727737586-2019_09_28.tar.gz",
    "2019_09_29.tar.gz": "https://data.nlr.gov/system/files/248/1727737586-2019_09_29.tar.gz",
}

# ============================================================
# BMS Data (NREL SRRL Baseline Measurement System)
# Full 90-day period: Sep 5 - Dec 3, 2019
# ============================================================

BMS_VARIABLES = [
    "GHI",           # Global Horizontal Irradiance (W/m²)
    "DNI",           # Direct Normal Irradiance (W/m²)
    "DHI",           # Diffuse Horizontal Irradiance (W/m²)
    "Temperature",   # Ambient temperature (°C)
    "Humidity",      # Relative humidity (%)
    "Pressure",      # Barometric pressure (mbar)
    "Wind Speed",    # Wind speed (m/s)
    "Wind Direction", # Wind direction (degrees)
]

# MIDC data download URL for SRRL BMS
BMS_MIDC_URL = "https://midcdmz.nrel.gov/apps/data_api.pl"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar, following redirects."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Already exists: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_tarball(tar_path: Path, dest_dir: Path) -> bool:
    """Extract a tar.gz file."""
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest_dir)
        logger.info(f"Extracted: {tar_path.name} -> {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        return False


def download_cloudcv():
    """Download all available CloudCV daily archives."""
    cloudcv_dir = DATA_DIR / "raw" / "cloudcv"
    cloudcv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading CloudCV Dataset (8 days)")
    logger.info("=" * 60)

    # Download README first
    download_file(
        CLOUDCV_FILES["README.md"],
        cloudcv_dir / "README.md",
        "README.md",
    )

    # Download and extract each day's archive
    successes = 0
    for filename, url in CLOUDCV_FILES.items():
        if filename == "README.md":
            continue

        tar_path = cloudcv_dir / filename
        if download_file(url, tar_path, filename):
            # Extract the archive
            extract_tarball(tar_path, cloudcv_dir)
            successes += 1

    logger.info(f"CloudCV: {successes}/{len(CLOUDCV_FILES) - 1} days downloaded successfully")
    return successes > 0


def download_bms():
    """Download BMS 1-minute meteorological data for the CloudCV period.

    Uses the NREL MIDC Data API to fetch CSV data.
    """
    bms_dir = DATA_DIR / "raw" / "bms"
    bms_dir.mkdir(parents=True, exist_ok=True)
    output_path = bms_dir / "bms_srrl_2019.csv"

    if output_path.exists() and output_path.stat().st_size > 1000:
        logger.info(f"BMS data already exists: {output_path}")
        return True

    logger.info("=" * 60)
    logger.info("Downloading BMS Meteorological Data")
    logger.info("Sep 5 - Dec 3, 2019 (1-minute resolution)")
    logger.info("=" * 60)

    # MIDC API for SRRL BMS 1-minute data
    # The MIDC interface uses specific variable IDs
    # We'll request all available variables for the period
    params = {
        "site": "BMS",
        "begin": "20190905",
        "end": "20191203",
        "inst": "1",     # 1-minute data
        "type": "data",
        "wession": "default",
    }

    logger.info(f"Fetching from MIDC: {BMS_MIDC_URL}")
    try:
        response = requests.get(BMS_MIDC_URL, params=params, timeout=600)
        response.raise_for_status()

        if len(response.text) < 100:
            logger.warning("MIDC response too short, trying alternate approach")
            raise ValueError("Short response")

        with open(output_path, "w") as f:
            f.write(response.text)
        logger.info(f"BMS data saved: {output_path} ({len(response.text)} bytes)")
        return True

    except Exception as e:
        logger.warning(f"MIDC API failed: {e}")
        logger.info("Trying alternate MIDC download URL...")

        # Try the alternate download format
        alt_url = (
            "https://midcdmz.nrel.gov/apps/data_api.pl?"
            "site=BMS&begin=20190905&end=20191203"
        )
        try:
            response = requests.get(alt_url, timeout=600)
            if len(response.text) > 1000:
                with open(output_path, "w") as f:
                    f.write(response.text)
                logger.info(f"BMS data saved via alternate URL: {output_path}")
                return True
        except Exception as e2:
            logger.warning(f"Alternate URL also failed: {e2}")

        # Try raw download endpoint
        raw_url = (
            "https://midcdmz.nrel.gov/apps/rawdata.pl?"
            "site=BMS&st=1&en=2&yr=2019&mo=9&dy=5&ession=default"
        )
        try:
            response = requests.get(raw_url, timeout=600)
            if len(response.text) > 500:
                with open(output_path, "w") as f:
                    f.write(response.text)
                logger.info(f"BMS data saved via raw endpoint: {output_path}")
                return True
        except Exception as e3:
            logger.warning(f"Raw endpoint also failed: {e3}")

        logger.error(
            "Could not download BMS data automatically.\n"
            "Please download manually from:\n"
            "  https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS\n"
            "Select: 1-Minute Data, Sep 5 2019 to Dec 3 2019\n"
            f"Save to: {output_path}"
        )
        return False


def download_bms_daily():
    """Download BMS data day-by-day as a fallback approach."""
    bms_dir = DATA_DIR / "raw" / "bms"
    bms_dir.mkdir(parents=True, exist_ok=True)
    combined_path = bms_dir / "bms_srrl_2019.csv"

    if combined_path.exists() and combined_path.stat().st_size > 1000:
        logger.info("BMS data already exists")
        return True

    logger.info("Downloading BMS data day-by-day...")

    from datetime import date, timedelta
    start = date(2019, 9, 5)
    end = date(2019, 12, 3)
    current = start

    all_data = []
    header = None

    while current <= end:
        day_str = current.strftime("%Y%m%d")
        url = (
            f"https://midcdmz.nrel.gov/apps/data_api.pl?"
            f"site=BMS&begin={day_str}&end={day_str}&inst=1&type=data"
        )

        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 100:
                lines = resp.text.strip().split("\n")
                if header is None and len(lines) > 1:
                    header = lines[0]
                    all_data.append(header)
                # Skip header for subsequent days
                for line in lines[1:]:
                    if line.strip():
                        all_data.append(line)
                logger.info(f"  {current}: {len(lines) - 1} rows")
            else:
                logger.warning(f"  {current}: empty or error ({resp.status_code})")
        except Exception as e:
            logger.warning(f"  {current}: {e}")

        current += timedelta(days=1)

    if all_data:
        with open(combined_path, "w") as f:
            f.write("\n".join(all_data))
        logger.info(f"BMS data combined: {len(all_data)} rows -> {combined_path}")
        return True
    return False


def verify_data():
    """Verify downloaded data and print summary."""
    logger.info("\n" + "=" * 60)
    logger.info("Data Verification Summary")
    logger.info("=" * 60)

    # CloudCV
    cloudcv_dir = DATA_DIR / "raw" / "cloudcv"
    day_dirs = sorted([d for d in cloudcv_dir.iterdir() if d.is_dir()])
    total_images = 0
    total_csvs = 0
    for day_dir in day_dirs:
        images = list(day_dir.rglob("*.jpg"))
        csvs = list(day_dir.rglob("*.csv"))
        total_images += len(images)
        total_csvs += len(csvs)
        logger.info(f"  {day_dir.name}: {len(images)} images, {len(csvs)} CSVs")

    logger.info(f"CloudCV Total: {len(day_dirs)} days, {total_images} images, {total_csvs} CSVs")

    # BMS
    bms_path = DATA_DIR / "raw" / "bms" / "bms_srrl_2019.csv"
    if bms_path.exists():
        with open(bms_path) as f:
            lines = f.readlines()
        logger.info(f"BMS: {len(lines)} rows, header: {lines[0][:100].strip() if lines else 'N/A'}...")
    else:
        logger.warning("BMS data not found")


def main():
    """Run full data download pipeline."""
    logger.info("SolarSDE Data Download Pipeline")
    logger.info(f"Data directory: {DATA_DIR}")

    # Download CloudCV
    cloudcv_ok = download_cloudcv()

    # Download BMS - try bulk first, then daily fallback
    bms_ok = download_bms()
    if not bms_ok:
        logger.info("Trying day-by-day BMS download...")
        bms_ok = download_bms_daily()

    # Verify
    verify_data()

    if cloudcv_ok:
        logger.info("\nData download complete!")
    else:
        logger.warning("\nSome downloads failed. Check logs above.")


if __name__ == "__main__":
    main()
