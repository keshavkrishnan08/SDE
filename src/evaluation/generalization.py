"""Multi-site generalization experiment using NSRDB data."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Five geographically diverse test locations
TEST_LOCATIONS = {
    "phoenix_az": {"lat": 33.45, "lon": -112.07, "desc": "Desert, minimal clouds"},
    "miami_fl": {"lat": 25.76, "lon": -80.19, "desc": "Tropical, afternoon thunderstorms"},
    "seattle_wa": {"lat": 47.61, "lon": -122.33, "desc": "Marine, persistent overcast"},
    "chicago_il": {"lat": 41.88, "lon": -87.63, "desc": "Continental, variable"},
    "honolulu_hi": {"lat": 21.31, "lon": -157.86, "desc": "Trade wind cumulus"},
}


def load_nsrdb_location(
    location_key: str,
    nsrdb_dir: str | Path = "data/raw/nsrdb",
    year: int = 2019,
) -> Optional[dict]:
    """Load NSRDB data for a specific location.

    Args:
        location_key: Key from TEST_LOCATIONS.
        nsrdb_dir: Directory containing NSRDB HDF5 files.
        year: Year to load.

    Returns:
        Dict with 'ghi', 'timestamps', 'location_info', or None if not available.
    """
    nsrdb_dir = Path(nsrdb_dir)
    loc = TEST_LOCATIONS[location_key]

    # Look for pre-downloaded NSRDB file
    possible_files = list(nsrdb_dir.glob(f"*{location_key}*{year}*.h5"))
    if not possible_files:
        possible_files = list(nsrdb_dir.glob(f"*{year}*.h5"))

    if not possible_files:
        logger.warning(f"No NSRDB data found for {location_key}. Download required.")
        return None

    try:
        import h5py
        with h5py.File(possible_files[0], "r") as f:
            ghi = f["ghi"][:]
            timestamps = f.get("time_index", f.get("timestamps"))
            if timestamps is not None:
                timestamps = timestamps[:]
    except Exception as e:
        logger.warning(f"Error reading NSRDB file for {location_key}: {e}")
        return None

    return {
        "ghi": ghi.astype(np.float32),
        "location_info": loc,
        "location_key": location_key,
    }


def evaluate_generalization(
    model_predict_fn,
    test_locations: dict | None = None,
    nsrdb_dir: str | Path = "data/raw/nsrdb",
    num_samples: int = 100,
) -> dict[str, dict]:
    """Evaluate model generalization across multiple sites.

    Tests zero-shot performance (no fine-tuning) on NSRDB locations.

    Args:
        model_predict_fn: Callable(ghi_history, num_samples) -> forecast samples.
        test_locations: Dict of location configs. Default: TEST_LOCATIONS.
        nsrdb_dir: Path to NSRDB data.
        num_samples: Number of MC samples.

    Returns:
        Dict mapping location_key -> metrics dict.
    """
    if test_locations is None:
        test_locations = TEST_LOCATIONS

    from src.evaluation.metrics import compute_all_metrics

    results = {}
    for loc_key in test_locations:
        data = load_nsrdb_location(loc_key, nsrdb_dir)
        if data is None:
            logger.info(f"Skipping {loc_key} — data not available")
            continue

        ghi = data["ghi"]
        # Simple evaluation: use rolling windows
        seq_len = 30
        horizon = 60  # ~10 minutes for 10s data
        N = len(ghi) - seq_len - horizon

        if N <= 0:
            continue

        y_true = ghi[seq_len + horizon : seq_len + horizon + N]
        # Generate predictions
        y_samples = np.zeros((N, num_samples))
        for i in range(N):
            history = ghi[i : i + seq_len]
            y_samples[i] = model_predict_fn(history, num_samples)

        metrics = compute_all_metrics(y_true, y_samples)
        metrics["location"] = test_locations[loc_key]["desc"]
        results[loc_key] = metrics
        logger.info(f"{loc_key}: CRPS={metrics['crps']:.4f}")

    return results
