"""Download CloudCV and BMS datasets from NREL."""

import logging
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CLOUDCV_URL = "https://data.nrel.gov/system/files/248/CloudCV_data.zip"
BMS_BASE_URL = "https://midcdmz.nrel.gov/apps/data_api.pl"

# NREL SRRL location
SRRL_LAT = 39.742
SRRL_LON = -105.18

# CloudCV date range
CLOUDCV_START = "2019-09-05"
CLOUDCV_END = "2019-12-03"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"File already exists: {dest}")
        return dest

    logger.info(f"Downloading {url} -> {dest}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract zip or tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    logger.info(f"Extracted {archive_path} -> {dest_dir}")


def download_cloudcv(raw_dir: str | Path = "data/raw/cloudcv") -> Path:
    """Download the CloudCV 10-second sky image + irradiance dataset."""
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / "CloudCV_data.zip"
    download_file(CLOUDCV_URL, archive_path)
    extract_archive(archive_path, raw_dir)
    return raw_dir


def download_bms(
    raw_dir: str | Path = "data/raw/bms",
    start_date: str = CLOUDCV_START,
    end_date: str = CLOUDCV_END,
) -> Path:
    """Download SRRL BMS 1-minute meteorological data for the CloudCV period.

    Uses the NREL MIDC data API to fetch CSV data for the overlapping period.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "bms_data.csv"

    if output_path.exists():
        logger.info(f"BMS data already exists: {output_path}")
        return output_path

    # MIDC API parameters for SRRL BMS
    params = {
        "site": "BMS",
        "begin": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "inst": "1",  # 1-minute data
        "type": "data",
        "wession": "default",
    }

    logger.info(f"Downloading BMS data from {start_date} to {end_date}")
    try:
        response = requests.get(BMS_BASE_URL, params=params, timeout=600)
        response.raise_for_status()
        with open(output_path, "w") as f:
            f.write(response.text)
        logger.info(f"BMS data saved to {output_path}")
    except requests.RequestException as e:
        logger.warning(
            f"Could not download BMS data via API: {e}. "
            "Please download manually from https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS"
        )
        # Create a placeholder so downstream code can check
        output_path.touch()

    return output_path


def download_all(raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download all required datasets."""
    raw_dir = Path(raw_dir)
    paths = {}
    paths["cloudcv"] = download_cloudcv(raw_dir / "cloudcv")
    paths["bms"] = download_bms(raw_dir / "bms")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
