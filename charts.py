"""
Chart Utilities — Plotly figures for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PALETTE = {
    "bg":       "#0a0f0a",
    "panel":    "#111811",
    "border":   "#1e3a1e",
    "green":    "#00ff88",
    "yellow":   "#ffd700",
    "orange":   "#ff8c00",
    "red":      "#ff3333",
    "muted":    "#4a6741",
    "text":     "#c8e6c9",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"], family="monospace"),
    margin=dict(l=20, r=20, t=40, b=20),
)

STATUS_COLORS = {
    "READY":    PALETTE["green"],
    "MODERATE": PALETTE["yellow"],
    "CAUTION":  PALETTE["orange"],
    "RECOVERY": PALETTE["red"],
}

RISK_COLORS = {
    "LOW":    PALETTE["green"],
    "MEDIUM": PALETTE["yellow"],
    "HIGH":   PALETTE["red"],
}

FATIGUE_COLORS = {
    "LOW":      PALETTE["green"],
    "MODERATE": PALETTE["yellow"],
    "HIGH":     PALETTE["red"],
}


def readiness_bar_chart(df: pd.DataFrame) -> go.Figure:
    colors = [STATUS_COLORS.get(s, PALETTE["muted"]) for s in df["rec_status"]]
    fig = go.Figure(go.Bar(
        x=df["name"],
        y=df["readiness_score"],
        marker_color=colors,
        text=df["readiness_score"].astype(str),
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="READINESS SCORE — ALL OPERATORS", x=0.5,
                   font=dict(size=13, color=PALETTE["green"])),
        yaxis=dict(range=[0, 110], gridcolor=PALETTE["border"],
                   title="Score", tickfont=dict(color=PALETTE["muted"])),
        xaxis=dict(tickangle=-30, tickfont=dict(color=PALETTE["text"])),
        height=320,
    )
    fig.add_hline(y=80, line_dash="dot", line_color=PALETTE["green"],   annotation_text="READY",   annotation_font_color=PALETTE["green"])
    fig.add_hline(y=60, line_dash="dot", line_color=PALETTE["yellow"],  annotation_text="MODERATE",annotation_font_color=PALETTE["yellow"])
    fig.add_hline(y=40, line_dash="dot", line_color=PALETTE["orange"],  annotation_text="CAUTION", annotation_font_color=PALETTE["orange"])
    return fig


def injury_risk_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["injury_risk"].value_counts()
    colors = [RISK_COLORS.get(r, PALETTE["muted"]) for r in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors,
                    line=dict(color=PALETTE["bg"], width=2)),
        hole=0.55,
        textfont=dict(color="#ffffff", size=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="INJURY RISK DISTRIBUTION", x=0.5,
                   font=dict(size=13, color=PALETTE["green"])),
        height=300,
        showlegend=True,
        legend=dict(font=dict(color=PALETTE["text"])),
    )
    return fig


def fatigue_scatter(df: pd.DataFrame) -> go.Figure:
    colors = [FATIGUE_COLORS.get(f, PALETTE["muted"]) for f in df["fatigue_level"]]
    fig = go.Figure(go.Scatter(
        x=df["training_load"],
        y=df["sleep_hours"],
        mode="markers+text",
        text=df["name"],
        textposition="top center",
        textfont=dict(size=9, color=PALETTE["muted"]),
        marker=dict(color=colors, size=14,
                    line=dict(color=PALETTE["bg"], width=1)),
        customdata=df[["readiness_score", "fatigue_level"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Load: %{x}<br>"
            "Sleep: %{y}h<br>"
            "Readiness: %{customdata[0]}<br>"
            "Fatigue: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="TRAINING LOAD vs SLEEP — FATIGUE MAP", x=0.5,
                   font=dict(size=13, color=PALETTE["green"])),
        xaxis=dict(title="Training Load", gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["muted"])),
        yaxis=dict(title="Sleep Hours", gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["muted"])),
        height=350,
    )
    return fig


def gauge_chart(score: float, label: str = "READINESS") -> go.Figure:
    color = (PALETTE["green"]  if score >= 80 else
             PALETTE["yellow"] if score >= 60 else
             PALETTE["orange"] if score >= 40 else
             PALETTE["red"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(color=color, size=42)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=PALETTE["muted"],
                      tickfont=dict(color=PALETTE["muted"])),
            bar=dict(color=color),
            bgcolor=PALETTE["panel"],
            bordercolor=PALETTE["border"],
            steps=[
                dict(range=[0, 40],  color="#1a0505"),
                dict(range=[40, 60], color="#1a1000"),
                dict(range=[60, 80], color="#161400"),
                dict(range=[80, 100],color="#051a0a"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=score,
            ),
        ),
        title=dict(text=label, font=dict(color=PALETTE["text"], size=14)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
    )
    return fig


def hr_sleep_bar(athlete_row: pd.Series) -> go.Figure:
    categories = ["Heart Rate", "Sleep Hours", "Training Load", "Recovery"]
    normalized = [
        min(1.0, max(0.0, 1 - abs(athlete_row["heart_rate"] - 67) / 40)),
        min(1.0, athlete_row["sleep_hours"] / 8),
        min(1.0, 1 - athlete_row["training_load"] / 100),
        athlete_row["recovery_indicator"],
    ]
    colors = [PALETTE["green"] if v > 0.7 else
              PALETTE["yellow"] if v > 0.4 else
              PALETTE["red"] for v in normalized]

    fig = go.Figure(go.Bar(
        x=categories,
        y=[v * 100 for v in normalized],
        marker_color=colors,
        text=[f"{v*100:.0f}" for v in normalized],
        textposition="outside",
        textfont=dict(color=PALETTE["text"]),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="BIOMETRIC INDICATORS", x=0.5,
                   font=dict(size=13, color=PALETTE["green"])),
        yaxis=dict(range=[0, 115], gridcolor=PALETTE["border"],
                   title="Score %", tickfont=dict(color=PALETTE["muted"])),
        xaxis=dict(tickfont=dict(color=PALETTE["text"])),
        height=280,
    )
    return fig
