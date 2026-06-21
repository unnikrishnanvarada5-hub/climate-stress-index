"""
ml_insights.py
---------------
Two lightweight, genuinely-useful data-science components:

1. trend_slope(): fits a simple linear regression of CSI vs time over the
   7-day hourly forecast for a city, returning the slope (CSI points/day).
   Used to flag "fastest worsening" / "fastest improving" cities.

2. cluster_archetypes(): KMeans clustering of cities into climate-stress
   "archetypes" (e.g. Heat-dominated, Pollution-dominated, UV-dominated,
   Balanced/Mild) based on their current component-score profile.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def trend_slope(hourly_df: pd.DataFrame) -> float:
    """Slope of CSI over time, in CSI points per day. Positive = worsening."""
    if hourly_df.empty or "csi" not in hourly_df.columns:
        return np.nan
    t = (hourly_df["time"] - hourly_df["time"].min()).dt.total_seconds() / 86400.0
    X = t.values.reshape(-1, 1)
    y = hourly_df["csi"].values
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return np.nan
    model = LinearRegression().fit(X[mask], y[mask])
    return round(float(model.coef_[0]), 2)


ARCHETYPE_NAMES = {
    0: "Balanced / Mild",
    1: "Heat-Dominated",
    2: "Pollution-Dominated",
    3: "UV-Dominated",
}


def cluster_archetypes(snapshot_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    """
    Clusters cities by their normalized component-score profile
    (heat_score, air_score, uv_score, humidity_score, stagnation_score).

    Returns the input df with an added 'archetype_raw' (cluster id) and
    'archetype' (human-readable label, assigned by which component
    dominates that cluster's centroid) column, plus the fitted model info.
    """
    features = ["heat_score", "air_score", "uv_score", "humidity_score", "stagnation_score"]
    df = snapshot_df.dropna(subset=features).copy()
    if len(df) < n_clusters:
        df["archetype"] = "Insufficient data"
        return df, None

    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    k = min(n_clusters, len(df))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    df["archetype_raw"] = labels

    # Label each cluster by its dominant raw-score component at the centroid
    centroids_raw = df.groupby("archetype_raw")[features].mean()
    dominant = centroids_raw.idxmax(axis=1)

    label_map = {
        "heat_score": "Heat-Dominated",
        "air_score": "Pollution-Dominated",
        "uv_score": "UV-Dominated",
        "humidity_score": "Humidity-Dominated",
        "stagnation_score": "Stagnant-Air-Dominated",
    }
    # If a cluster's spread across components is low (all similar), call it Balanced
    spread = centroids_raw.max(axis=1) - centroids_raw.min(axis=1)
    archetype_labels = {}
    for cid in centroids_raw.index:
        if spread[cid] < 12:
            archetype_labels[cid] = "Balanced / Mild"
        else:
            archetype_labels[cid] = label_map[dominant[cid]]

    df["archetype"] = df["archetype_raw"].map(archetype_labels)
    return df, km
