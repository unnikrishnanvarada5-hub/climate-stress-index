"""
Global Urban Climate Stress Index (CSI) Dashboard
===================================================
A live, multi-city climate-stress monitor built on free public weather
and air-quality data (Open-Meteo). Combines real-time data engineering,
a custom composite index, regression-based trend detection, and
unsupervised clustering into one Streamlit app.

Run locally:  streamlit run app.py
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fetch_data import load_cities, fetch_current_snapshot, fetch_hourly_forecast  # noqa: E402
from index_calc import compute_csi, csi_band_color, CSI_BANDS  # noqa: E402
from ml_insights import trend_slope, cluster_archetypes  # noqa: E402
from ui_components import render_gauge, render_kpi_card, render_leaderboard_table, BAND_COLORS, _flatten  # noqa: E402

st.set_page_config(
    page_title="Climate Stress Index — Atmospheric Monitor",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling --
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-deep: #0A1628;
        --bg-panel: #101F36;
        --bg-panel-light: #16273F;
        --border: #22344C;
        --text-primary: #EAF0F7;
        --text-muted: #7E92AC;
        --amber: #FF6B47;
        --teal: #2DD4BF;
        --violet: #B14EFF;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--bg-deep); }
    h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif !important; }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }

    .hero-wrap { border-bottom: 1px solid var(--border); padding-bottom: 1.4rem; margin-bottom: 1.6rem; }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace; letter-spacing: 3px; font-size: 0.72rem;
        color: var(--teal); text-transform: uppercase; margin-bottom: 0.4rem;
    }
    .hero-title { font-size: 2.6rem; font-weight: 700; color: var(--text-primary); line-height: 1.1; margin: 0; }
    .hero-sub { color: var(--text-muted); font-size: 0.98rem; max-width: 760px; margin-top: 0.6rem; line-height: 1.5; }

    .kpi-row { display: flex; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
    .kpi-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
                padding: 0.9rem 1.1rem; flex: 1; min-width: 150px; }
    .kpi-label { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 1.5px;
                 color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.35rem; }
    .kpi-value { font-size: 1.55rem; font-weight: 700; color: var(--text-primary); font-family: 'Space Grotesk'; }
    .kpi-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.15rem; }

    .panel-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 2px;
        color: var(--text-muted); text-transform: uppercase; margin: 1.2rem 0 0.5rem 0;
        border-left: 2px solid var(--teal); padding-left: 8px;
    }

    .band-pill { display:inline-block; padding: 3px 12px; border-radius: 999px;
                 font-weight:600; font-size:0.78rem; font-family: 'JetBrains Mono', monospace; }

    .lb-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .lb-table th { text-align: left; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; letter-spacing: 1.5px; text-transform: uppercase;
        padding: 8px 10px; border-bottom: 1px solid var(--border); }
    .lb-table td { padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
    .lb-table .rank { font-family: 'JetBrains Mono', monospace; color: var(--text-muted); }
    .city-cell { display: flex; flex-direction: column; }
    .city-name { color: var(--text-primary); font-weight: 600; }
    .city-country { color: var(--text-muted); font-size: 0.74rem; }
    .bar-cell { width: 28%; }
    .bar-track { background: var(--bg-panel-light); border-radius: 4px; height: 7px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; }
    .score-cell { font-family: 'JetBrains Mono', monospace; font-weight: 700; }

    .gauge-wrap { display: flex; justify-content: center; align-items: center;
                  background: var(--bg-panel); border: 1px solid var(--border);
                  border-radius: 8px; padding: 0.6rem 0 0 0; }

    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    .stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk'; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    _flatten("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">// LIVE ATMOSPHERIC MONITORING</div>
        <div class="hero-title">Global Climate Stress Index</div>
        <div class="hero-sub">
            A real-time instrument panel tracking how heat, air quality, UV exposure,
            humidity, and air stagnation combine into a single stress reading —
            across 39 cities worldwide. Built on live public weather &amp; air-quality data.
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 15 minutes; click above to force-refresh.")

    st.divider()
    st.subheader("📖 About the CSI")
    st.markdown(
        """
        The **Climate Stress Index (0–100)** blends five sub-scores:
        - 🔥 **Heat** (32%) — apparent temperature
        - 🌫️ **Air Quality** (28%) — US AQI
        - ☀️ **UV** (16%) — UV index
        - 💧 **Humidity** (12%) — humidity × heat interaction
        - 🍃 **Stagnation** (12%) — low wind speed

        See the *Methodology* tab for full detail.
        """
    )
    st.divider()
    st.caption("Data source: [Open-Meteo](https://open-meteo.com) (free, no key). "
               "Built with Streamlit, scikit-learn, Plotly.")

# ------------------------------------------------------------------- data --
cities = load_cities("data/cities.csv")

with st.spinner("Fetching live global climate data..."):
    snapshot = fetch_current_snapshot(cities)

snapshot = compute_csi(snapshot)
valid = snapshot.dropna(subset=["csi"])

# ------------------------------------------------------------------- tabs --
tab_map, tab_city, tab_compare, tab_cluster, tab_method = st.tabs(
    ["🗺️ Global Snapshot", "🔎 City Deep Dive", "⚖️ Compare Cities",
     "🧬 Climate Archetypes", "📐 Methodology"]
)

# =========================================================== TAB 1: MAP ===
with tab_map:
    worst = valid.loc[valid["csi"].idxmax()]
    best = valid.loc[valid["csi"].idxmin()]
    kpi_html = (
        '<div class="kpi-row">'
        + render_kpi_card("Cities Monitored", str(len(valid)), "live feed", "#2DD4BF")
        + render_kpi_card("Highest Stress", worst["city"], f"CSI {worst['csi']:.1f}", "#FF6B47")
        + render_kpi_card("Lowest Stress", best["city"], f"CSI {best['csi']:.1f}", "#2DD4BF")
        + render_kpi_card("Global Avg CSI", f"{valid['csi'].mean():.1f}", "across all cities", "#F5C24D")
        + "</div>"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Live Climate Stress Map</div>', unsafe_allow_html=True)
    fig = px.scatter_geo(
        valid,
        lat="lat", lon="lon",
        color="csi",
        size=np.clip(valid["csi"], 5, 100),
        hover_name="city",
        hover_data={"country": True, "csi": True, "lat": False, "lon": False},
        color_continuous_scale=["#2DD4BF", "#F5C24D", "#FF9447", "#FF6B47", "#B14EFF"],
        range_color=(0, 100),
        projection="natural earth",
    )
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0A1628", geo_bgcolor="#0A1628",
        font=dict(color="#EAF0F7", family="Inter"),
    )
    fig.update_geos(showland=True, landcolor="#16273F", showocean=True,
                     oceancolor="#0A1628", showcountries=True, countrycolor="#22344C")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel-label">Stress Leaderboard</div>', unsafe_allow_html=True)
    leaderboard_df = valid.sort_values("csi", ascending=False).reset_index(drop=True)
    st.markdown(render_leaderboard_table(leaderboard_df, max_rows=20), unsafe_allow_html=True)

# ====================================================== TAB 2: CITY VIEW ==
with tab_city:
    sel_city = st.selectbox("Choose a city", sorted(valid["city"].unique()))
    row = valid[valid["city"] == sel_city].iloc[0]

    band_color = csi_band_color(row["csi"])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="gauge-wrap">' + render_gauge(row["csi"], row["csi_band"]) + '</div>',
                    unsafe_allow_html=True)
        st.markdown(
            _flatten(f"""
            <div style="display:flex; justify-content:space-between; margin-top:0.8rem;">
              <div class="kpi-card" style="flex:1; margin-right:6px;">
                <div class="kpi-label">Apparent Temp</div>
                <div class="kpi-value" style="font-size:1.2rem;">{row['apparent_temperature']:.1f}°C</div>
              </div>
              <div class="kpi-card" style="flex:1; margin-left:6px;">
                <div class="kpi-label">Humidity</div>
                <div class="kpi-value" style="font-size:1.2rem;">{row['relative_humidity_2m']:.0f}%</div>
              </div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:0.6rem;">
              <div class="kpi-card" style="flex:1; margin-right:6px;">
                <div class="kpi-label">UV Index</div>
                <div class="kpi-value" style="font-size:1.2rem;">{row['uv_index']:.1f}</div>
              </div>
              <div class="kpi-card" style="flex:1; margin-left:6px;">
                <div class="kpi-label">US AQI</div>
                <div class="kpi-value" style="font-size:1.2rem;">{row['us_aqi']:.0f}</div>
              </div>
            </div>
            <div class="kpi-card" style="margin-top:0.6rem;">
                <div class="kpi-label">Wind Speed</div>
                <div class="kpi-value" style="font-size:1.2rem;">{row['wind_speed_10m']:.1f} km/h</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with c2:
        components = pd.DataFrame({
            "Component": ["Heat", "Air Quality", "UV", "Humidity", "Stagnation"],
            "Score": [row["heat_score"], row["air_score"], row["uv_score"],
                      row["humidity_score"], row["stagnation_score"]],
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=components["Score"], theta=components["Component"], fill="toself",
            name=sel_city, line_color="#2DD4BF", fillcolor="rgba(45,212,191,0.25)",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#101F36",
                radialaxis=dict(visible=True, range=[0, 100], color="#7E92AC", gridcolor="#22344C"),
                angularaxis=dict(color="#EAF0F7", gridcolor="#22344C"),
            ),
            showlegend=False, height=350, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#EAF0F7", family="Inter"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        with st.spinner("Loading 7-day forecast..."):
            hourly = fetch_hourly_forecast(cities[cities["city"] == sel_city].iloc[0])
            hourly = compute_csi(hourly)
            slope = trend_slope(hourly)

        trend_word = "worsening" if slope > 0.3 else ("improving" if slope < -0.3 else "stable")
        st.markdown(
            f'<div class="kpi-card" style="border-top:2px solid #F5C24D; margin-bottom:0.8rem;">'
            f'<div class="kpi-label">7-Day Trend (Linear Regression)</div>'
            f'<div class="kpi-value" style="font-size:1.1rem;">{trend_word.upper()} '
            f'<span style="color:#7E92AC; font-size:0.9rem;">({slope:+.2f} CSI pts/day)</span></div>'
            f'</div>', unsafe_allow_html=True,
        )

        fig_line = px.line(hourly, x="time", y="csi", title=f"{sel_city}: 7-Day CSI Forecast")
        fig_line.update_traces(line_color="#FF6B47")
        fig_line.add_hrect(y0=65, y1=100, fillcolor="#B14EFF", opacity=0.08, line_width=0)
        fig_line.update_layout(
            height=320, margin=dict(t=40, b=10),
            plot_bgcolor="#101F36", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAF0F7", family="Inter"),
            xaxis=dict(gridcolor="#22344C"), yaxis=dict(gridcolor="#22344C", range=[0, 100]),
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ===================================================== TAB 3: COMPARE =====
with tab_compare:
    sel_cities = st.multiselect(
        "Select cities to compare (7-day CSI forecast)",
        sorted(valid["city"].unique()),
        default=list(valid.sort_values("csi", ascending=False)["city"].head(4)),
    )
    if sel_cities:
        frames = []
        with st.spinner("Loading forecasts..."):
            for c in sel_cities:
                h = fetch_hourly_forecast(cities[cities["city"] == c].iloc[0])
                h = compute_csi(h)
                frames.append(h)
        combined = pd.concat(frames, ignore_index=True)
        fig_cmp = px.line(combined, x="time", y="csi", color="city",
                           title="7-Day CSI Forecast Comparison",
                           color_discrete_sequence=["#FF6B47", "#2DD4BF", "#F5C24D", "#B14EFF", "#5B9BD5", "#7E92AC"])
        fig_cmp.update_layout(
            height=480, plot_bgcolor="#101F36", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAF0F7", family="Inter"),
            xaxis=dict(gridcolor="#22344C"), yaxis=dict(gridcolor="#22344C", range=[0, 100]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown('<div class="panel-label">Trend Slopes (CSI points/day)</div>', unsafe_allow_html=True)
        slopes = pd.DataFrame([
            {"city": c, "trend_slope": trend_slope(combined[combined["city"] == c])}
            for c in sel_cities
        ]).sort_values("trend_slope", ascending=False)
        st.dataframe(slopes, use_container_width=True)
    else:
        st.info("Pick at least one city above to compare.")

# ================================================== TAB 4: CLUSTERING =====
with tab_cluster:
    st.markdown(
        "Cities are grouped into **climate-stress archetypes** using K-Means "
        "clustering on their five normalized stress components — revealing "
        "which underlying *driver* (heat, pollution, UV, humidity, or "
        "stagnant air) shapes each city's risk profile right now."
    )
    n_clusters = st.slider("Number of archetypes (k)", 2, 6, 4)
    clustered, model = cluster_archetypes(valid, n_clusters=n_clusters)

    if model is not None:
        fig_cl = px.scatter(
            clustered, x="heat_score", y="air_score",
            color="archetype", size="csi", hover_name="city",
            labels={"heat_score": "Heat Score", "air_score": "Air Quality Score"},
            title="City Archetypes (Heat vs. Air-Quality Stress)",
        )
        fig_cl.update_layout(
            height=480, plot_bgcolor="#101F36", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAF0F7", family="Inter"),
            xaxis=dict(gridcolor="#22344C"), yaxis=dict(gridcolor="#22344C"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown('<div class="panel-label">Archetype Membership</div>', unsafe_allow_html=True)
        for arch in sorted(clustered["archetype"].unique()):
            members = clustered[clustered["archetype"] == arch].sort_values("csi", ascending=False)
            with st.expander(f"**{arch}** ({len(members)} cities)"):
                st.dataframe(
                    members[["city", "country", "csi", "csi_band"]].reset_index(drop=True),
                    use_container_width=True,
                )
    else:
        st.warning("Not enough valid data points to cluster right now.")

# =================================================== TAB 5: METHODOLOGY ===
with tab_method:
    st.markdown(
        """
        ### How the Climate Stress Index (CSI) is built

        This project treats "climate stress" as a **multi-dimensional** problem —
        heat alone doesn't capture how unpleasant or dangerous a city's
        atmosphere feels right now. The CSI combines five normalized (0–100)
        sub-scores into one number:

        | Component | Weight | Formula intuition |
        |---|---|---|
        | 🔥 Heat | 32% | Apparent temperature scaled between 18°C (comfortable) and 45°C (dangerous) |
        | 🌫️ Air Quality | 28% | US AQI scaled, capped at 300 |
        | ☀️ UV | 16% | UV Index scaled against WHO 0–11+ scale |
        | 💧 Humidity | 12% | Relative humidity, *amplified* only when paired with heat (muggy-heat effect) |
        | 🍃 Stagnation | 12% | Low wind speed, a proxy for pollutant build-up risk |

        **Bands:** Low (0–25) · Moderate (25–45) · High (45–65) · Severe (65–85) · Extreme (85–100)

        ### Data Science techniques used
        - **Real-time API data engineering** — live ingestion & caching from Open-Meteo's
          weather and air-quality APIs (no key required).
        - **Feature engineering** — a novel composite index built from weighted,
          normalized, and interaction-based components (humidity × heat).
        - **Linear regression** — fits CSI vs. time over the 7-day hourly forecast
          per city to estimate a trend slope ("fastest worsening" cities).
        - **Unsupervised learning (K-Means)** — clusters cities into climate-stress
          *archetypes* based on which component dominates their profile, after
          standardizing features with `StandardScaler`.

        ### Caveats
        This is an original, illustrative index for analytical and educational
        purposes — it is **not** an official health/safety standard. Always
        consult local meteorological and public-health authorities for safety
        guidance.
        """
    )

st.divider()
st.caption("Built with Streamlit · scikit-learn · Plotly · Open-Meteo API — "
           "refreshes every 15 minutes. © Climate Stress Index Project")
