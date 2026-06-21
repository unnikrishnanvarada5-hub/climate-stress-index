"""
fetch_data.py
--------------
Pulls live weather + air-quality data from Open-Meteo (free, no API key)
for a list of cities, and returns a tidy DataFrame ready for index
computation.

Open-Meteo docs: https://open-meteo.com/en/docs
"""

import time
import requests
import pandas as pd
import streamlit as st

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "uv_index",
    "wind_speed_10m",
    "precipitation_probability",
]

AIR_QUALITY_HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "us_aqi",
    "ozone",
    "carbon_monoxide",
]


def _get(url: str, params: dict, retries: int = 2, timeout: int = 12):
    """Small retry wrapper around requests.get."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_city_bundle(lat: float, lon: float) -> dict:
    """Fetch current + 7-day hourly weather and air quality for one point."""
    weather = _get(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_HOURLY_VARS),
            "current": ",".join(
                ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                 "uv_index", "wind_speed_10m"]
            ),
            "forecast_days": 7,
            "timezone": "auto",
        },
    )
    air = _get(
        AIR_QUALITY_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
            "current": "pm2_5,pm10,us_aqi,ozone",
            "forecast_days": 7,
            "timezone": "auto",
        },
    )
    return {"weather": weather, "air": air}


@st.cache_data(ttl=900, show_spinner=False)
def load_cities(path: str = "data/cities.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_snapshot(cities: pd.DataFrame) -> pd.DataFrame:
    """One row per city: latest current-conditions reading."""
    rows = []
    for _, c in cities.iterrows():
        try:
            bundle = fetch_city_bundle(c["lat"], c["lon"])
            w = bundle["weather"]["current"]
            a = bundle["air"]["current"]
            rows.append(
                {
                    "city": c["city"],
                    "country": c["country"],
                    "region": c["region"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "temperature_2m": w.get("temperature_2m"),
                    "apparent_temperature": w.get("apparent_temperature"),
                    "relative_humidity_2m": w.get("relative_humidity_2m"),
                    "uv_index": w.get("uv_index"),
                    "wind_speed_10m": w.get("wind_speed_10m"),
                    "pm2_5": a.get("pm2_5"),
                    "pm10": a.get("pm10"),
                    "us_aqi": a.get("us_aqi"),
                    "ozone": a.get("ozone"),
                    "fetched_at": w.get("time", pd.Timestamp.utcnow().isoformat()),
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "city": c["city"], "country": c["country"], "region": c["region"],
                    "lat": c["lat"], "lon": c["lon"], "error": str(e),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hourly_forecast(city_row: pd.Series) -> pd.DataFrame:
    """Hourly 7-day weather + air-quality series for ONE city, merged on time."""
    bundle = fetch_city_bundle(city_row["lat"], city_row["lon"])
    w = bundle["weather"]["hourly"]
    a = bundle["air"]["hourly"]

    df_w = pd.DataFrame(w)
    df_a = pd.DataFrame(a)
    df = pd.merge(df_w, df_a, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_row["city"]
    return df
