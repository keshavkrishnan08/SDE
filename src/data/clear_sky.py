"""Clear-sky irradiance computation using the Ineichen-Perez model via pvlib."""

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location


# NREL SRRL, Golden, CO
SRRL_LOCATION = Location(
    latitude=39.742,
    longitude=-105.18,
    tz="America/Denver",
    altitude=1829,
    name="NREL SRRL",
)


def compute_solar_position(
    timestamps: pd.DatetimeIndex,
    location: Location = SRRL_LOCATION,
) -> pd.DataFrame:
    """Compute solar zenith and azimuth for each timestamp.

    Returns DataFrame with columns: apparent_zenith, azimuth, apparent_elevation.
    """
    return location.get_solarposition(timestamps)


def compute_clear_sky(
    timestamps: pd.DatetimeIndex,
    location: Location = SRRL_LOCATION,
    model: str = "ineichen",
) -> pd.DataFrame:
    """Compute clear-sky GHI, DNI, DHI using Ineichen-Perez model.

    Returns DataFrame with columns: ghi, dni, dhi.
    """
    return location.get_clearsky(timestamps, model=model)


def compute_clear_sky_index(
    ghi_measured: pd.Series,
    timestamps: pd.DatetimeIndex,
    location: Location = SRRL_LOCATION,
) -> pd.Series:
    """Compute clear-sky index k_t = GHI_measured / GHI_clearsky.

    Clamps output to [0, 1.5] to handle measurement noise near sunrise/sunset.
    """
    clearsky = compute_clear_sky(timestamps, location)
    ghi_clear = clearsky["ghi"].values

    # Avoid division by zero for low sun angles
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi_measured.values / ghi_clear
        kt = np.where(ghi_clear < 10.0, 0.0, kt)  # Set to 0 when clearsky is tiny
        kt = np.clip(kt, 0.0, 1.5)

    return pd.Series(kt, index=timestamps, name="clear_sky_index")


def filter_daytime(
    df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    zenith_max: float = 85.0,
    location: Location = SRRL_LOCATION,
) -> pd.DataFrame:
    """Remove nighttime rows where solar zenith angle exceeds threshold."""
    solpos = compute_solar_position(timestamps, location)
    mask = solpos["apparent_zenith"] <= zenith_max
    return df.loc[mask]


if __name__ == "__main__":
    # Quick test with a single day
    times = pd.date_range("2019-09-15 06:00", "2019-09-15 18:00", freq="1min", tz="America/Denver")
    cs = compute_clear_sky(times)
    print(f"Clear sky GHI peak: {cs['ghi'].max():.1f} W/m²")
    solpos = compute_solar_position(times)
    print(f"Min zenith: {solpos['apparent_zenith'].min():.1f}°")
