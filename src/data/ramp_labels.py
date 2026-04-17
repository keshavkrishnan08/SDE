"""Ramp event detection and labeling."""

import numpy as np
import pandas as pd


def detect_ramp_events(
    ghi: pd.Series,
    threshold: float = 50.0,
    window_seconds: int = 60,
    dt_seconds: int = 10,
) -> pd.Series:
    """Detect irradiance ramp events.

    A ramp event occurs when |ΔGHI| > threshold within a time window.

    Args:
        ghi: GHI time series (W/m²).
        threshold: Ramp threshold in W/m² per minute.
        window_seconds: Window over which to compute the change.
        dt_seconds: Time resolution of the data in seconds.

    Returns:
        Boolean Series indicating ramp events.
    """
    steps = window_seconds // dt_seconds
    delta_ghi = ghi.diff(periods=steps).abs()
    # Convert to per-minute rate
    rate = delta_ghi / (window_seconds / 60.0)
    return rate > threshold


def compute_ramp_magnitude(
    ghi: pd.Series,
    window_seconds: int = 60,
    dt_seconds: int = 10,
) -> pd.Series:
    """Compute ramp magnitude (W/m² per minute) for each timestep."""
    steps = window_seconds // dt_seconds
    delta_ghi = ghi.diff(periods=steps).abs()
    return delta_ghi / (window_seconds / 60.0)


def ramp_event_statistics(
    ghi: pd.Series,
    threshold: float = 50.0,
    window_seconds: int = 60,
    dt_seconds: int = 10,
) -> dict:
    """Compute summary statistics of ramp events in the series."""
    is_ramp = detect_ramp_events(ghi, threshold, window_seconds, dt_seconds)
    magnitude = compute_ramp_magnitude(ghi, window_seconds, dt_seconds)

    total_steps = len(ghi)
    ramp_steps = is_ramp.sum()

    return {
        "total_timesteps": total_steps,
        "ramp_timesteps": int(ramp_steps),
        "ramp_fraction": float(ramp_steps / total_steps) if total_steps > 0 else 0.0,
        "mean_ramp_magnitude": float(magnitude[is_ramp].mean()) if ramp_steps > 0 else 0.0,
        "max_ramp_magnitude": float(magnitude[is_ramp].max()) if ramp_steps > 0 else 0.0,
    }
