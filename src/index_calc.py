"""
index_calc.py
--------------
Defines the Climate Stress Index (CSI): a custom 0-100 composite score
built from five sub-components, each normalized 0-100:

  1. Heat stress       - apparent temperature mapped against a discomfort curve
  2. Air quality stress - US AQI normalized
  3. UV stress          - UV index normalized against WHO scale (0-11+)
  4. Humidity stress    - relative humidity, amplified when paired with heat
  5. Stagnation stress  - low wind speed (pollutant build-up proxy)

Weights are tunable in WEIGHTS below. This is original methodology
designed for this project (not an official/standardized index).
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "heat": 0.32,
    "air": 0.28,
    "uv": 0.16,
    "humidity": 0.12,
    "stagnation": 0.12,
}

CSI_BANDS = [
    (0, 25, "Low", "#2ecc71"),
    (25, 45, "Moderate", "#f1c40f"),
    (45, 65, "High", "#e67e22"),
    (65, 85, "Severe", "#e74c3c"),
    (85, 101, "Extreme", "#8e44ad"),
]


def _clip01(x):
    return np.clip(x, 0, 1)


def heat_score(apparent_temp: pd.Series) -> pd.Series:
    """0 at <=18C (comfortable), 100 at >=45C (dangerous heat)."""
    return _clip01((apparent_temp - 18) / (45 - 18)) * 100


def air_score(us_aqi: pd.Series) -> pd.Series:
    """US AQI already 0-500 scale; we cap stress contribution at 300."""
    return _clip01(us_aqi / 300) * 100


def uv_score(uv_index: pd.Series) -> pd.Series:
    """WHO UV index scale tops out functionally around 11+ (extreme)."""
    return _clip01(uv_index / 11) * 100


def humidity_score(rel_humidity: pd.Series, apparent_temp: pd.Series) -> pd.Series:
    """Humidity only really 'stresses' when paired with heat (muggy heat)."""
    heat_factor = _clip01((apparent_temp - 20) / 20)
    return _clip01(rel_humidity / 100) * heat_factor * 100


def stagnation_score(wind_speed_kmh: pd.Series) -> pd.Series:
    """Low wind => pollutant stagnation risk. 0 km/h -> 100, >=20km/h -> 0."""
    return _clip01((20 - wind_speed_kmh) / 20) * 100


def compute_csi(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """
    Adds component scores + final CSI column(s) to df.
    Expects columns: apparent_temperature, us_aqi, uv_index,
    relative_humidity_2m, wind_speed_10m  (with optional suffix, e.g. for
    hourly data the columns may not have a suffix at all).
    """
    df = df.copy()
    at = df.get(f"apparent_temperature{suffix}")
    aqi = df.get(f"us_aqi{suffix}")
    uv = df.get(f"uv_index{suffix}")
    rh = df.get(f"relative_humidity_2m{suffix}")
    wind = df.get(f"wind_speed_10m{suffix}")

    df["heat_score"] = heat_score(at)
    df["air_score"] = air_score(aqi)
    df["uv_score"] = uv_score(uv)
    df["humidity_score"] = humidity_score(rh, at)
    df["stagnation_score"] = stagnation_score(wind)

    df["csi"] = (
        df["heat_score"] * WEIGHTS["heat"]
        + df["air_score"] * WEIGHTS["air"]
        + df["uv_score"] * WEIGHTS["uv"]
        + df["humidity_score"] * WEIGHTS["humidity"]
        + df["stagnation_score"] * WEIGHTS["stagnation"]
    ).round(1)

    df["csi_band"] = df["csi"].apply(csi_band_label)
    return df


def csi_band_label(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    for lo, hi, label, _ in CSI_BANDS:
        if lo <= score < hi:
            return label
    return "Extreme"


def csi_band_color(score: float) -> str:
    if pd.isna(score):
        return "#7f8c8d"
    for lo, hi, _, color in CSI_BANDS:
        if lo <= score < hi:
            return color
    return CSI_BANDS[-1][3]
