"""
ui_components.py
------------------
Custom-built visual components for the instrument-panel design language:
an SVG barometer-style gauge (the project's signature element) plus
helpers for styled KPI cards and a styled leaderboard table.
"""

import math
import pandas as pd

BAND_COLORS = {
    "Low": "#2DD4BF",
    "Moderate": "#F5C24D",
    "High": "#FF9447",
    "Severe": "#FF6B47",
    "Extreme": "#B14EFF",
    "Unknown": "#5B6B85",
}


def _polar_to_xy(cx, cy, r, angle_deg):
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def render_gauge(score: float, band: str, size: int = 220) -> str:
    """Semicircular instrument-panel gauge, 180deg sweep (180 -> 0 deg), 0-100."""
    score = 0 if pd.isna(score) else max(0, min(100, score))
    color = BAND_COLORS.get(band, "#5B6B85")
    cx, cy, r = size / 2, size / 2 + 6, size * 0.40

    start_angle, end_angle = 180, 0
    needle_angle = start_angle + (score / 100) * (end_angle - start_angle)

    # tick marks every 10 units
    ticks = []
    for i in range(0, 11):
        ang = start_angle + (i / 10) * (end_angle - start_angle)
        major = i % 5 == 0
        r_out = r + 2
        r_in = r - (10 if major else 5)
        x1, y1 = _polar_to_xy(cx, cy, r_out, ang)
        x2, y2 = _polar_to_xy(cx, cy, r_in, ang)
        ticks.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#3A4D6A" stroke-width="{2 if major else 1}"/>'
        )

    # colored arc (progress)
    arc_r = r
    x_start, y_start = _polar_to_xy(cx, cy, arc_r, start_angle)
    x_end, y_end = _polar_to_xy(cx, cy, arc_r, needle_angle)
    large_arc = 0
    arc_path = (
        f'<path d="M {x_start:.1f} {y_start:.1f} A {arc_r:.1f} {arc_r:.1f} 0 '
        f'{large_arc} 1 {x_end:.1f} {y_end:.1f}" fill="none" stroke="{color}" '
        f'stroke-width="10" stroke-linecap="round"/>'
    )
    # background track
    x_bg2, y_bg2 = _polar_to_xy(cx, cy, arc_r, end_angle)
    bg_path = (
        f'<path d="M {x_start:.1f} {y_start:.1f} A {arc_r:.1f} {arc_r:.1f} 0 1 1 '
        f'{x_bg2:.1f} {y_bg2:.1f}" fill="none" stroke="#1E2F47" stroke-width="10" '
        f'stroke-linecap="round"/>'
    )

    # needle
    nx, ny = _polar_to_xy(cx, cy, r - 14, needle_angle)
    needle = (
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}"/>'
    )

    svg = f"""
    <svg width="{size}" height="{size*0.62:.0f}" viewBox="0 0 {size} {size*0.62:.0f}"
         xmlns="http://www.w3.org/2000/svg">
      {bg_path}
      {''.join(ticks)}
      {arc_path}
      {needle}
      <text x="{cx}" y="{cy - r*0.32:.0f}" text-anchor="middle"
            font-family="JetBrains Mono, monospace" font-size="34" font-weight="700"
            fill="#EAF0F7">{score:.0f}</text>
      <text x="{cx}" y="{cy - r*0.32 + 20:.0f}" text-anchor="middle"
            font-family="Inter, sans-serif" font-size="11" letter-spacing="2"
            fill="{color}">{band.upper()}</text>
    </svg>
    """
    return svg


def render_kpi_card(label: str, value: str, sub: str = "", accent: str = "#2DD4BF") -> str:
    return f"""
    <div class="kpi-card" style="border-top: 2px solid {accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_leaderboard_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    """Custom HTML leaderboard with instrument-style bars + band pills."""
    rows_html = []
    for i, (_, r) in enumerate(df.head(max_rows).iterrows()):
        color = BAND_COLORS.get(r["csi_band"], "#5B6B85")
        bar_pct = max(2, min(100, r["csi"]))
        rows_html.append(f"""
        <tr>
          <td class="rank">{i+1:02d}</td>
          <td class="city-cell">
              <span class="city-name">{r['city']}</span>
              <span class="city-country">{r['country']}</span>
          </td>
          <td class="bar-cell">
              <div class="bar-track">
                  <div class="bar-fill" style="width:{bar_pct}%; background:{color};"></div>
              </div>
          </td>
          <td class="score-cell" style="color:{color};">{r['csi']:.1f}</td>
          <td><span class="band-pill" style="background:{color}22; color:{color}; border:1px solid {color};">{r['csi_band']}</span></td>
        </tr>
        """)
    return f"""
    <table class="lb-table">
      <thead>
        <tr><th>#</th><th>City</th><th>Stress Level</th><th>CSI</th><th>Band</th></tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
