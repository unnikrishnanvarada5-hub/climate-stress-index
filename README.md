# 🌍 Global Urban Climate Stress Index

A live, interactive dashboard that monitors **39 world cities** in real time and
scores each one on an original **Climate Stress Index (CSI)** — a 0–100
composite combining heat, air quality, UV exposure, humidity, and air
stagnation, built entirely on free, key-free public APIs.

**[🚀 Live demo →](#)** *(add your Streamlit Cloud URL here after deploying)*

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.36+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this project

Most "climate dashboards" just plot temperature. This project asks a more
interesting question: **what does it actually feel like, and how risky is it,
to be outside in a city right now?** That requires combining several signals
— not just heat, but pollution, sun exposure, mugginess, and even how
stagnant the air is. The CSI is an original methodology designed to answer
that, with the data science to back it up:

- **Real-time data engineering** — live ingestion from the [Open-Meteo](https://open-meteo.com)
  weather and air-quality APIs (no API key required, fully free).
- **Feature engineering** — five normalized sub-scores combined into one
  weighted, interaction-aware composite index.
- **Regression-based trend detection** — linear regression over each city's
  7-day hourly forecast to flag which cities are trending toward more (or
  less) climate stress.
- **Unsupervised learning** — K-Means clustering groups cities into
  data-driven "climate-stress archetypes" (Heat-Dominated, Pollution-Dominated,
  UV-Dominated, Balanced/Mild, etc.).

## Features

| Tab | What it shows |
|---|---|
| 🗺️ Global Snapshot | World map + leaderboard of live CSI scores across all cities |
| 🔎 City Deep Dive | Per-city radar chart of stress components + 7-day CSI forecast trend |
| ⚖️ Compare Cities | Overlay multiple cities' forecasted CSI trajectories |
| 🧬 Climate Archetypes | K-Means clustering of cities into climate-stress "personality types" |
| 📐 Methodology | Full transparency on the index formula, weights, and ML methods |

## Project structure

```
climate-stress-index/
├── app.py                  # Streamlit app (UI + orchestration)
├── src/
│   ├── fetch_data.py        # Open-Meteo API ingestion + caching
│   ├── index_calc.py        # CSI composite index formula
│   └── ml_insights.py       # Regression trend slope + KMeans archetypes
├── data/
│   └── cities.csv            # 39 global cities with coordinates
├── .streamlit/config.toml    # Theme config
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/<your-username>/climate-stress-index.git
cd climate-stress-index
pip install -r requirements.txt
streamlit run app.py
```

No API keys, `.env` files, or signup required — Open-Meteo's API is free and
open for non-commercial use.

## Deploy on Streamlit Community Cloud

1. Push this repo to your GitHub account (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file
   path to `app.py`.
4. Click **Deploy**. No secrets needed.

## Methodology (short version)

```
CSI = 0.32 × Heat + 0.28 × AirQuality + 0.16 × UV + 0.12 × Humidity + 0.12 × Stagnation
```

Each component is normalized 0–100 from raw meteorological/air-quality
values (apparent temperature, US AQI, UV index, relative humidity, wind
speed). Full detail and formulas are in the in-app **Methodology** tab and in
[`src/index_calc.py`](src/index_calc.py).

> This is an original, illustrative index built for analytical/educational
> purposes — it is **not** an official health or safety standard.

## Tech stack

`Python` · `Streamlit` · `pandas` · `NumPy` · `scikit-learn` (LinearRegression, KMeans, StandardScaler) · `Plotly` · `Open-Meteo API`

## License

MIT — see [LICENSE](LICENSE).
