"""Charts v2 — Plotly figures"""
import plotly.graph_objects as go
import pandas as pd

P = {
    "bg": "#050908", "panel": "#0a120e", "border": "#0f2a1a",
    "green": "#00e676", "teal": "#00bfa5", "yellow": "#ffd600",
    "orange": "#ff6d00", "red": "#ff1744", "muted": "#37574a",
    "text": "#b2dfdb", "accent": "#69f0ae",
}

LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(color=P["text"], family="'Share Tech Mono', monospace"),
              margin=dict(l=20, r=20, t=44, b=20))

SC = {"READY": P["green"], "MODERATE": P["yellow"], "CAUTION": P["orange"], "RECOVERY": P["red"]}
RC = {"LOW": P["green"], "MEDIUM": P["yellow"], "HIGH": P["red"]}
FC = {"LOW": P["green"], "MODERATE": P["yellow"], "HIGH": P["red"]}

def readiness_bar_chart(df):
    colors = [SC.get(s, P["muted"]) for s in df["rec_status"]]
    fig = go.Figure(go.Bar(x=df["name"], y=df["readiness_score"],
                           marker_color=colors, marker_line_color=P["bg"], marker_line_width=1,
                           text=df["readiness_score"].astype(str), textposition="outside",
                           textfont=dict(color=P["text"], size=10)))
    fig.update_layout(**LAYOUT,
                      title=dict(text="OPERATOR READINESS SCORES", x=0.5, font=dict(size=12, color=P["accent"])),
                      yaxis=dict(range=[0,115], gridcolor=P["border"], title="", tickfont=dict(color=P["muted"])),
                      xaxis=dict(tickangle=-25, tickfont=dict(color=P["text"], size=10)), height=300)
    for val, color, label in [(80, P["green"], "READY"), (60, P["yellow"], "MOD"), (40, P["orange"], "CAU")]:
        fig.add_hline(y=val, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=label, annotation_font_color=color, annotation_font_size=9)
    return fig

def injury_risk_pie(df):
    counts = df["injury_risk"].value_counts()
    colors = [RC.get(r, P["muted"]) for r in counts.index]
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                           marker=dict(colors=colors, line=dict(color=P["bg"], width=3)),
                           hole=0.6, textfont=dict(color="#fff", size=11)))
    fig.update_layout(**LAYOUT,
                      title=dict(text="INJURY RISK", x=0.5, font=dict(size=12, color=P["accent"])),
                      height=280, showlegend=True, legend=dict(font=dict(color=P["text"], size=10)))
    return fig

def fatigue_scatter(df):
    colors = [FC.get(f, P["muted"]) for f in df["fatigue_level"]]
    fig = go.Figure(go.Scatter(
        x=df["training_load"], y=df["sleep_hours"], mode="markers+text",
        text=df["name"], textposition="top center", textfont=dict(size=8, color=P["muted"]),
        marker=dict(color=colors, size=13, line=dict(color=P["bg"], width=1)),
        customdata=df[["readiness_score","fatigue_level"]].values,
        hovertemplate="<b>%{text}</b><br>Load: %{x}<br>Sleep: %{y}h<br>Readiness: %{customdata[0]}<br>Fatigue: %{customdata[1]}<extra></extra>"))
    fig.update_layout(**LAYOUT,
                      title=dict(text="LOAD vs SLEEP — FATIGUE MAP", x=0.5, font=dict(size=12, color=P["accent"])),
                      xaxis=dict(title="Training Load", gridcolor=P["border"], tickfont=dict(color=P["muted"])),
                      yaxis=dict(title="Sleep (hrs)", gridcolor=P["border"], tickfont=dict(color=P["muted"])),
                      height=320)
    return fig

def gauge_chart(score, label="READINESS"):
    color = P["green"] if score >= 80 else P["yellow"] if score >= 60 else P["orange"] if score >= 40 else P["red"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(color=color, size=40, family="Share Tech Mono")),
        gauge=dict(axis=dict(range=[0,100], tickcolor=P["muted"], tickfont=dict(color=P["muted"])),
                   bar=dict(color=color, thickness=0.25),
                   bgcolor=P["panel"], bordercolor=P["border"],
                   steps=[dict(range=[0,40], color="#1a0508"), dict(range=[40,60], color="#1a1200"),
                          dict(range=[60,80], color="#121800"), dict(range=[80,100], color="#021508")],
                   threshold=dict(line=dict(color=color, width=4), thickness=0.85, value=score)),
        title=dict(text=label, font=dict(color=P["text"], size=12, family="Share Tech Mono")),
        domain=dict(x=[0,1], y=[0,1])))
    fig.update_layout(**LAYOUT, height=240)
    return fig

def performance_trend(history_df):
    if history_df.empty: return go.Figure()
    completed = history_df[history_df["completed"] == True].copy()
    if completed.empty: return go.Figure()
    intensity_map = {"REST": 0, "LOW": 25, "MODERATE": 60, "HIGH": 90}
    completed["intensity_val"] = completed["intensity"].map(intensity_map).fillna(50)
    color_map = {"REST": P["muted"], "LOW": P["teal"], "MODERATE": P["yellow"], "HIGH": P["red"]}
    colors = [color_map.get(i, P["muted"]) for i in completed["intensity"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=completed["date"], y=completed["duration_min"],
                         marker_color=colors, name="Duration",
                         text=completed["training_type"], textposition="auto",
                         textfont=dict(size=9, color="#fff"),
                         hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Duration: %{y} min<extra></extra>"))
    if "performance_rating" in completed.columns:
        ratings = completed[completed["performance_rating"] > 0]
        if not ratings.empty:
            fig.add_trace(go.Scatter(x=ratings["date"], y=ratings["performance_rating"] * 10,
                                     mode="lines+markers", name="Rating×10",
                                     line=dict(color=P["accent"], width=2),
                                     marker=dict(color=P["accent"], size=7)))
    fig.update_layout(**LAYOUT,
                      title=dict(text="TRAINING HISTORY", x=0.5, font=dict(size=12, color=P["accent"])),
                      xaxis=dict(tickfont=dict(color=P["text"], size=9), gridcolor=P["border"]),
                      yaxis=dict(title="Duration (min)", gridcolor=P["border"], tickfont=dict(color=P["muted"])),
                      height=280, barmode="overlay",
                      legend=dict(font=dict(color=P["text"], size=10)))
    return fig

def biometric_radar(row):
    hr_s = min(1.0, max(0.0, 1 - abs(row["heart_rate"] - 67) / 40))
    sleep_s = min(1.0, row["sleep_hours"] / 8)
    load_s = min(1.0, 1 - row["training_load"] / 100)
    rec_s = row["recovery_indicator"]
    score_s = row["readiness_score"] / 100
    categories = ["Heart Rate", "Sleep Quality", "Load Balance", "Recovery", "Readiness"]
    values = [hr_s, sleep_s, load_s, rec_s, score_s]
    values_pct = [v * 100 for v in values]
    fig = go.Figure(go.Scatterpolar(
        r=values_pct + [values_pct[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor=f"rgba(0,230,118,0.1)",
        line=dict(color=P["green"], width=2),
        marker=dict(color=P["green"], size=6)))
    fig.update_layout(**LAYOUT,
                      polar=dict(bgcolor=P["panel"],
                                 radialaxis=dict(visible=True, range=[0,100],
                                                 tickfont=dict(color=P["muted"], size=8),
                                                 gridcolor=P["border"]),
                                 angularaxis=dict(tickfont=dict(color=P["text"], size=10),
                                                  gridcolor=P["border"])),
                      height=300,
                      title=dict(text="BIOMETRIC RADAR", x=0.5, font=dict(size=12, color=P["accent"])))
    return fig
