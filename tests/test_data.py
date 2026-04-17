"""Tests for data pipeline components."""

import numpy as np
import pandas as pd
import pytest

from src.data.clear_sky import compute_solar_position, compute_clear_sky, compute_clear_sky_index
from src.data.ramp_labels import detect_ramp_events, ramp_event_statistics


class TestClearSky:
    def test_solar_position_shape(self):
        times = pd.date_range("2019-09-15 10:00", periods=10, freq="1min", tz="America/Denver")
        solpos = compute_solar_position(times)
        assert "apparent_zenith" in solpos.columns
        assert len(solpos) == 10

    def test_clear_sky_positive(self):
        times = pd.date_range("2019-09-15 10:00", periods=10, freq="1min", tz="America/Denver")
        cs = compute_clear_sky(times)
        assert (cs["ghi"] >= 0).all()
        assert cs["ghi"].max() > 0

    def test_clear_sky_index_bounds(self):
        times = pd.date_range("2019-09-15 10:00", periods=10, freq="1min", tz="America/Denver")
        ghi = pd.Series(np.full(10, 500.0), index=times)
        kt = compute_clear_sky_index(ghi, times)
        assert (kt >= 0).all()
        assert (kt <= 1.5).all()


class TestRampLabels:
    def test_detect_ramp_events(self):
        # Create a series with a clear ramp
        ghi = pd.Series(np.concatenate([
            np.full(10, 500.0),  # stable
            np.linspace(500, 200, 6),  # ramp down
            np.full(10, 200.0),  # stable
        ]))
        ramps = detect_ramp_events(ghi, threshold=50.0, window_seconds=60, dt_seconds=10)
        assert ramps.any(), "Should detect ramp events"

    def test_no_ramp_in_constant(self):
        ghi = pd.Series(np.full(50, 500.0))
        ramps = detect_ramp_events(ghi, threshold=50.0)
        # After the initial NaN window, all should be False
        assert not ramps.iloc[10:].any()

    def test_ramp_statistics(self):
        ghi = pd.Series(np.random.randn(100) * 10 + 500)
        stats = ramp_event_statistics(ghi, threshold=50.0)
        assert "total_timesteps" in stats
        assert stats["total_timesteps"] == 100
