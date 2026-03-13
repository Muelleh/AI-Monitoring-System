"""
╔══════════════════════════════════════════════════════════╗
║  PERF-AI COMMAND · v2.0                                 ║
║  AI Athlete & Soldier Performance Monitoring System     ║
╚══════════════════════════════════════════════════════════╝
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime

from ai_engine        import analyse_team, compute_readiness_score, classify_fatigue, classify_injury_risk, training_recommendation
from sensor_simulator import simulate_team
from perf_charts      import readiness_bar_chart, injury_risk_pie, fatigue_scatter, gauge_chart, performance_trend, biometric_radar

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="PERF-AI COMMAND", page_icon="⬡", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# CSS — Premium Tactical Dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@300;400;600&display=swap');

:root {
  --bg:        #050908;
  --bg2:       #070d0b;
  --panel:     #0a120e;
  --panel2:    #0d1a14;
  --border:    #0f2a1a;
  --border2:   #1a3d2a;
  --green:     #00e676;
  --green2:    #69f0ae;
  --teal:      #00bfa5;
  --yellow:    #ffd600;
  --orange:    #ff6d00;
  --red:       #ff1744;
  --muted:     #37574a;
  --text:      #b2dfdb;
  --text2:     #80cbc4;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Exo 2', sans-serif !important;
}

/* ── Scanline overlay ── */
.stApp::before {
  content: '';
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,230,118,0.015) 2px, rgba(0,230,118,0.015) 4px);
  pointer-events: none; z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stRadio label { font-family: 'Share Tech Mono', monospace !important; font-size: 12px !important; letter-spacing: 1px !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--panel2) !important;
  border: 1px solid var(--border2) !important;
  border-top: 2px solid var(--green) !important;
  border-radius: 0 !important;
  padding: 14px 16px 10px !important;
  position: relative;
}
[data-testid="metric-container"]::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, var(--green), transparent);
}
[data-testid="metric-container"] label {
  color: var(--muted) !important; font-size: 9px !important;
  letter-spacing: 3px !important; text-transform: uppercase !important;
  font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--green) !important; font-size: 28px !important;
  font-family: 'Orbitron', monospace !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 10px !important; font-family: 'Share Tech Mono', monospace !important; }

/* ── Typography ── */
h1 { font-family: 'Orbitron', monospace !important; color: var(--green) !important; letter-spacing: 4px !important; font-size: 20px !important; text-transform: uppercase !important; }
h2 { font-family: 'Orbitron', monospace !important; color: var(--text) !important; letter-spacing: 3px !important; font-size: 15px !important; text-transform: uppercase !important; }
h3 { font-family: 'Share Tech Mono', monospace !important; color: var(--muted) !important; letter-spacing: 4px !important; font-size: 11px !important; text-transform: uppercase !important; }

/* ── Buttons ── */
.stButton > button {
  background: transparent !important; border: 1px solid var(--green) !important;
  color: var(--green) !important; font-family: 'Share Tech Mono', monospace !important;
  letter-spacing: 2px !important; text-transform: uppercase !important; font-size: 11px !important;
  border-radius: 0 !important; transition: all 0.15s !important; padding: 8px 20px !important;
}
.stButton > button:hover { background: var(--green) !important; color: var(--bg) !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea textarea {
  background: var(--panel2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; border-radius: 0 !important;
  font-family: 'Share Tech Mono', monospace !important; font-size: 12px !important;
}
.stSlider * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--panel) !important; border-bottom: 1px solid var(--border2) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; border-radius: 0 !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: var(--green) !important; border-bottom: 2px solid var(--green) !important; background: var(--panel2) !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid var(--border2) !important; }
.stDataFrame th { background: var(--panel2) !important; color: var(--muted) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 10px !important; letter-spacing: 2px !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: var(--text) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; }

/* ── Divider ── */
hr { border-color: var(--border2) !important; margin: 12px 0 !important; }

/* ── Custom components ── */
.sys-header {
  background: linear-gradient(135deg, var(--panel) 0%, var(--panel2) 100%);
  border: 1px solid var(--border2); border-left: 3px solid var(--green);
  padding: 16px 24px; margin-bottom: 20px;
  display: flex; align-items: center; gap: 20px;
}
.sys-title { font-family: 'Orbitron', monospace; color: var(--green); font-size: 18px; font-weight: 700; letter-spacing: 5px; }
.sys-sub { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 10px; letter-spacing: 3px; margin-top: 3px; }
.sys-status { font-family: 'Share Tech Mono', monospace; font-size: 11px; text-align: right; }

.card {
  background: var(--panel2); border: 1px solid var(--border2);
  border-radius: 0; padding: 16px; margin: 6px 0;
  position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, var(--green), transparent 60%);
}
.card-label { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 9px; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 6px; }
.card-value { font-family: 'Orbitron', monospace; font-size: 22px; font-weight: 700; }
.card-sub { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 10px; margin-top: 3px; }

.badge { display: inline-block; padding: 3px 10px; font-family: 'Share Tech Mono', monospace; font-size: 11px; font-weight: bold; letter-spacing: 2px; border: 1px solid; border-radius: 0; }

.alert-box { border-left: 3px solid; padding: 12px 16px; font-family: 'Share Tech Mono', monospace; font-size: 12px; letter-spacing: 1px; margin: 6px 0; }
.alert-ready    { border-color: #00e676; background: rgba(0,230,118,0.06); color: #00e676; }
.alert-moderate { border-color: #ffd600; background: rgba(255,214,0,0.06);  color: #ffd600; }
.alert-caution  { border-color: #ff6d00; background: rgba(255,109,0,0.06);  color: #ff6d00; }
.alert-recovery { border-color: #ff1744; background: rgba(255,23,68,0.06);  color: #ff1744; }

.profile-header {
  background: linear-gradient(135deg, var(--panel2) 0%, #0a1f18 100%);
  border: 1px solid var(--border2); border-left: 4px solid var(--green);
  padding: 20px 24px; margin-bottom: 16px; display: flex; align-items: center; gap: 20px;
}
.avatar {
  width: 56px; height: 56px; border-radius: 50%;
  background: linear-gradient(135deg, var(--green), var(--teal));
  display: flex; align-items: center; justify-content: center;
  font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700; color: var(--bg);
  border: 2px solid var(--green); flex-shrink: 0;
}
.operator-name { font-family: 'Orbitron', monospace; color: var(--green); font-size: 20px; font-weight: 700; letter-spacing: 3px; }
.operator-meta { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 11px; letter-spacing: 2px; margin-top: 4px; }

.training-card {
  background: var(--panel); border: 1px solid var(--border2);
  padding: 14px 16px; margin: 6px 0; display: flex; align-items: center; gap: 14px;
  transition: border-color 0.2s;
}
.training-card:hover { border-color: var(--green); }
.t-date { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 10px; letter-spacing: 1px; min-width: 80px; }
.t-type { font-family: 'Exo 2', sans-serif; color: var(--text); font-size: 13px; font-weight: 600; }
.t-meta { font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 10px; margin-top: 2px; }
.t-rating { font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700; margin-left: auto; }

.section-title {
  font-family: 'Share Tech Mono', monospace; color: var(--muted); font-size: 10px;
  letter-spacing: 4px; text-transform: uppercase; padding: 8px 0 6px;
  border-bottom: 1px solid var(--border2); margin: 16px 0 12px;
  display: flex; align-items: center; gap: 8px;
}
.section-title::before { content: '◈'; color: var(--green); }

.intensity-HIGH     { color: #ff1744; border-color: #ff1744; }
.intensity-MODERATE { color: #ffd600; border-color: #ffd600; }
.intensity-LOW      { color: #00bfa5; border-color: #00bfa5; }
.intensity-REST     { color: #37574a; border-color: #37574a; }

.form-section {
  background: var(--panel2); border: 1px solid var(--border2);
  padding: 20px; margin: 8px 0;
}
.form-section::before { content: ''; display: block; width: 40px; height: 2px; background: var(--green); margin-bottom: 12px; }

.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.stat-chip { background: var(--panel); border: 1px solid var(--border2); padding: 4px 10px; font-family: 'Share Tech Mono', monospace; font-size: 10px; color: var(--text2); }
.stat-chip span { color: var(--green); font-weight: bold; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); }
::-webkit-scrollbar-thumb:hover { background: var(--green); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "training_log" not in st.session_state:
    # Load from CSV
    hist_path = Path(__file__).parent / "data" / "training_history.csv"
    if hist_path.exists():
        st.session_state.training_log = pd.read_csv(hist_path)
    else:
        st.session_state.training_log = pd.DataFrame(columns=[
            "athlete_id","date","training_type","duration_min","intensity",
            "notes","assigned_by","completed","performance_rating"])

if "coach_notes" not in st.session_state:
    st.session_state.coach_notes = {}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_team():
    p = Path(__file__).parent / "data" / "simulated_data.csv"
    if p.exists():
        df = pd.read_csv(p)
    else:
        df = simulate_team(n=10)
    return analyse_team(df)

df = load_team()
log = st.session_state.training_log


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def status_css(status):
    return {"READY":"alert-ready","MODERATE":"alert-moderate","CAUTION":"alert-caution","RECOVERY":"alert-recovery"}.get(status,"alert-moderate")

def badge_color(status):
    return {"READY":"#00e676","MODERATE":"#ffd600","CAUTION":"#ff6d00","RECOVERY":"#ff1744"}.get(status,"#aaa")

def risk_color(risk):
    return {"LOW":"#00e676","MEDIUM":"#ffd600","HIGH":"#ff1744"}.get(risk,"#aaa")

def intensity_stars(rating):
    if rating == 0: return "—"
    return "★" * rating + "☆" * (10 - rating)

def get_athlete_log(athlete_id):
    return log[log["athlete_id"] == athlete_id].sort_values("date", ascending=False)

def initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()

def completion_rate(athlete_id):
    alog = get_athlete_log(athlete_id)
    if alog.empty: return 0
    done = alog[alog["completed"] == True]
    return round(len(done) / len(alog) * 100)

def avg_rating(athlete_id):
    alog = get_athlete_log(athlete_id)
    rated = alog[(alog["completed"] == True) & (alog["performance_rating"] > 0)]
    return round(rated["performance_rating"].mean(), 1) if not rated.empty else 0.0

def total_training_hours(athlete_id):
    alog = get_athlete_log(athlete_id)
    done = alog[alog["completed"] == True]
    return round(done["duration_min"].sum() / 60, 1)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:12px 0 8px;">'
        '<div style="font-family:Orbitron,monospace;color:#00e676;font-size:14px;letter-spacing:4px;font-weight:700;">⬡ PERF-AI</div>'
        '<div style="font-family:Share Tech Mono,monospace;color:#80cbc4;font-size:10px;letter-spacing:3px;margin-top:2px;">COMMAND SYSTEM v2.0</div>'
        '</div><hr style="border-color:#1a3d2a;margin:8px 0 16px;">',
        unsafe_allow_html=True)

    st.markdown('<div style="color:#80cbc4;font-size:11px;letter-spacing:3px;margin-bottom:8px;font-family:monospace;">▸ USER ROLE</div>', unsafe_allow_html=True)
    role = st.selectbox("", ["⬡  Coach / Commander", "◈  Athlete / Soldier"], label_visibility="collapsed")
    is_coach = "Coach" in role

    st.markdown('<hr style="border-color:#1a3d2a;margin:12px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="color:#80cbc4;font-size:11px;letter-spacing:3px;margin-bottom:8px;font-family:monospace;">▸ NAVIGATION</div>', unsafe_allow_html=True)

    if is_coach:
        pages = {"🗺  Team Overview": "team", "👤  Athlete Profile": "profile",
                 "📋  Assign Training": "assign", "📊  Analytics": "analytics"}
    else:
        pages = {"◈  My Dashboard": "dash", "📋  My Training Log": "mylog",
                 "💪  Log Workout": "logworkout"}

    page_label = st.radio("", list(pages.keys()), label_visibility="collapsed")
    page = pages[page_label]

    if not is_coach:
        st.markdown('<hr style="border-color:#1a3d2a;margin:12px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#80cbc4;font-size:11px;letter-spacing:3px;margin-bottom:8px;font-family:monospace;">▸ LOGGED IN AS</div>', unsafe_allow_html=True)
        athlete_self = st.selectbox("", df["name"].tolist(), label_visibility="collapsed")

    if is_coach:
        st.markdown('<hr style="border-color:#1a3d2a;margin:12px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#80cbc4;font-size:11px;letter-spacing:3px;margin-bottom:8px;font-family:monospace;">▸ OPERATOR ROSTER</div>', unsafe_allow_html=True)

        # Search box
        search_query = st.text_input("", placeholder="🔍  Search name, ID, position...",
                                     label_visibility="collapsed", key="roster_search")

        # Filter buttons row
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_status = st.selectbox("", ["ALL STATUS","READY","MODERATE","CAUTION","RECOVERY"],
                                         label_visibility="collapsed", key="filter_status")
        with filter_col2:
            filter_risk = st.selectbox("", ["ALL RISK","LOW","MEDIUM","HIGH"],
                                       label_visibility="collapsed", key="filter_risk")

        # Apply filters
        filtered_df = df.copy()
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["name"].str.lower().str.contains(q) |
                filtered_df["id"].str.lower().str.contains(q) |
                filtered_df.get("position", pd.Series([""] * len(filtered_df))).fillna("").str.lower().str.contains(q) |
                filtered_df.get("squad", pd.Series([""] * len(filtered_df))).fillna("").str.lower().str.contains(q)
            ]
        if filter_status != "ALL STATUS":
            filtered_df = filtered_df[filtered_df["rec_status"] == filter_status]
        if filter_risk != "ALL RISK":
            filtered_df = filtered_df[filtered_df["injury_risk"] == filter_risk]

        # Result count
        total_filtered = len(filtered_df)
        total_all = len(df)
        st.markdown(
            f'<div style="font-family:monospace;color:#4a7a6a;font-size:10px;letter-spacing:2px;margin-bottom:6px;">'
            f'SHOWING {total_filtered} / {total_all} OPERATORS</div>',
            unsafe_allow_html=True)

        # Roster list
        status_icon = {"READY": "🟢", "MODERATE": "🟡", "CAUTION": "🟠", "RECOVERY": "🔴"}
        if filtered_df.empty:
            st.markdown('<div style="text-align:center;padding:20px;font-family:monospace;color:#1a3d2a;border:1px dashed #0f2a1a;font-size:11px;">NO OPERATORS FOUND</div>', unsafe_allow_html=True)
        else:
            for _, r in filtered_df.iterrows():
                bc        = badge_color(r["rec_status"])
                icon      = status_icon.get(r["rec_status"], "⚪")
                ri        = " ⚠" if r["injury_risk"] == "HIGH" else ""
                name_val  = r["name"]
                id_val    = r["id"]
                pos_val   = r.get("position", "—")
                score_val = r["readiness_score"]
                # Highlight search match
                if search_query:
                    hl = search_query
                    name_display = name_val.replace(hl, f'<span style="background:#00e67633;color:#00e676;">{hl}</span>') if hl.lower() in name_val.lower() else name_val
                else:
                    name_display = name_val
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:8px 10px;'
                    f'margin:3px 0;border:1px solid #1a3d2a;background:#0a120e;border-left:3px solid {bc};">'
                    f'<span style="font-size:11px;">{icon}</span>'
                    f'<div style="flex:1;min-width:0;">'
                    f'<div style="font-family:Exo 2,sans-serif;color:#e0f2f1;font-size:13px;font-weight:700;">{name_display}{ri}</div>'
                    f'<div style="font-family:monospace;color:#80cbc4;font-size:10px;">{id_val} · {pos_val}</div>'
                    f'</div>'
                    f'<div style="font-family:Orbitron,monospace;color:{bc};font-size:14px;font-weight:700;">{score_val}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1a3d2a;margin:12px 0;">', unsafe_allow_html=True)
    if st.button("↻  REFRESH", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    total_ops = len(df)
    today_str = str(date.today())
    st.markdown(
        f'<div style="margin-top:12px;font-family:monospace;color:#4a7a6a;font-size:10px;letter-spacing:2px;line-height:2;">'
        f'STATUS: <span style="color:#00e676;">● ONLINE</span><br>'
        f'OPERATORS: <span style="color:#b2dfdb;">{total_ops}</span><br>'
        f'DATE: <span style="color:#b2dfdb;">{today_str}</span>'
        f'</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GLOBAL HEADER
# ─────────────────────────────────────────────
high_risk = int((df["injury_risk"] == "HIGH").sum())
st.markdown(f"""
<div class="sys-header">
  <div>
    <div class="sys-title">⬡ PERF-AI COMMAND</div>
    <div class="sys-sub">AI ATHLETE & SOLDIER PERFORMANCE MONITORING · {date.today()}</div>
  </div>
  <div style="margin-left:auto;" class="sys-status">
    <div style="color:#37574a;">SYSTEM</div>
    <div style="color:#00e676;font-size:14px;font-weight:bold;">● OPERATIONAL</div>
    {"<div style='color:#ff1744;font-size:11px;margin-top:4px;'>⚠ " + str(high_risk) + " HIGH RISK</div>" if high_risk > 0 else ""}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: TEAM OVERVIEW
# ══════════════════════════════════════════════
if page == "team":
    st.markdown("## TEAM COMMAND CENTER")

    # KPIs
    avg_score = round(df["readiness_score"].mean(), 1)
    ready_count = (df["rec_status"] == "READY").sum()
    recovery_count = (df["rec_status"] == "RECOVERY").sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("OPERATORS", len(df))
    k2.metric("AVG READINESS", avg_score)
    k3.metric("READY FOR OPS", ready_count)
    k4.metric("HIGH RISK", int(high_risk), delta=f"{'⚠ ATTENTION' if high_risk > 0 else 'CLEAR'}", delta_color="inverse")
    k5.metric("RECOVERY REQ.", recovery_count)

    # Alerts
    if high_risk > 0:
        st.markdown('<div class="section-title">CRITICAL ALERTS</div>', unsafe_allow_html=True)
        alert_cols = st.columns(min(high_risk, 4))
        for i, (_, r) in enumerate(df[df["injury_risk"] == "HIGH"].iterrows()):
            if i < 4:
                with alert_cols[i]:
                    st.markdown(f"""
                    <div class="alert-box alert-recovery">
                      🚨 <strong>{r['name']}</strong><br>
                      ID: {r['id']} · {r.get('squad','—')}<br>
                      Score: {r['readiness_score']} · Risk: HIGH<br>
                      <span style="font-size:10px;">{r['rec_text']}</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">PERFORMANCE OVERVIEW</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1: st.plotly_chart(readiness_bar_chart(df), use_container_width=True)
    with c2: st.plotly_chart(injury_risk_pie(df), use_container_width=True)
    st.plotly_chart(fatigue_scatter(df), use_container_width=True)

    st.markdown('<div class="section-title">OPERATOR STATUS MATRIX</div>', unsafe_allow_html=True)

    # Build rich table
    rows_html = ""
    for _, r in df.iterrows():
        bc = badge_color(r["rec_status"])
        rc = risk_color(r["injury_risk"])
        fc = {"LOW":"#00e676","MODERATE":"#ffd600","HIGH":"#ff1744"}.get(r["fatigue_level"],"#aaa")
        comp = completion_rate(r["id"])
        avg_r = avg_rating(r["id"])
        hrs = total_training_hours(r["id"])
        rows_html += f"""
        <tr style="border-bottom:1px solid #0f2a1a;transition:background 0.15s;" onmouseover="this.style.background='#0d1a14'" onmouseout="this.style.background='transparent'">
          <td style="padding:10px 12px;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:10px;">{r['id']}</td>
          <td style="padding:10px 12px;">
            <div style="font-family:'Exo 2',sans-serif;font-weight:600;color:#b2dfdb;">{r['name']}</div>
            <div style="font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:1px;">{r.get('position','—')} · {r.get('squad','—')}</div>
          </td>
          <td style="padding:10px 12px;text-align:center;">
            <span style="font-family:'Orbitron',monospace;color:{bc};font-size:18px;font-weight:700;">{r['readiness_score']}</span>
          </td>
          <td style="padding:10px 12px;text-align:center;"><span class="badge" style="color:{fc};border-color:{fc};">{r['fatigue_level']}</span></td>
          <td style="padding:10px 12px;text-align:center;"><span class="badge" style="color:{rc};border-color:{rc};">{r['injury_risk']}</span></td>
          <td style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:10px;">{comp}%</td>
          <td style="padding:10px 12px;text-align:center;font-family:'Orbitron',monospace;color:#69f0ae;font-size:13px;">{avg_r}</td>
          <td style="padding:10px 12px;font-family:'Share Tech Mono',monospace;font-size:10px;color:{bc};max-width:200px;">{r['rec_label']}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid #0f2a1a;">
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
          <tr style="background:#0d1a14;border-bottom:1px solid #1a3d2a;">
            <th style="padding:10px 12px;text-align:left;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">ID</th>
            <th style="padding:10px 12px;text-align:left;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">OPERATOR</th>
            <th style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">SCORE</th>
            <th style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">FATIGUE</th>
            <th style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">RISK</th>
            <th style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">COMPLETION</th>
            <th style="padding:10px 12px;text-align:center;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">AVG RATING</th>
            <th style="padding:10px 12px;font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">AI DECISION</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ATHLETE PROFILE (Coach View)
# ══════════════════════════════════════════════
elif page == "profile":
    st.markdown("## ATHLETE PROFILE")

    selected = st.selectbox("SELECT OPERATOR", df["name"].tolist())
    row = df[df["name"] == selected].iloc[0]
    alog = get_athlete_log(row["id"])
    bc = badge_color(row["rec_status"])

    # Profile header
    st.markdown(f"""
    <div class="profile-header">
      <div class="avatar">{initials(row['name'])}</div>
      <div style="flex:1;">
        <div class="operator-name">{row['name']}</div>
        <div class="operator-meta">{row['id']} · {row['role'].upper()} · {row.get('position','—')} · SQUAD {row.get('squad','—')}</div>
        <div class="stat-row" style="margin-top:8px;">
          <div class="stat-chip">AGE <span>{row.get('age','—')}</span></div>
          <div class="stat-chip">HEIGHT <span>{row.get('height','—')} cm</span></div>
          <div class="stat-chip">WEIGHT <span>{row.get('weight','—')} kg</span></div>
          <div class="stat-chip">SESSIONS <span>{len(alog)}</span></div>
          <div class="stat-chip">COMPLETION <span>{completion_rate(row['id'])}%</span></div>
          <div class="stat-chip">AVG RATING <span>{avg_rating(row['id'])}/10</span></div>
          <div class="stat-chip">TOTAL HRS <span>{total_training_hours(row['id'])}h</span></div>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">TODAY'S STATUS</div>
        <div style="font-family:'Orbitron',monospace;color:{bc};font-size:32px;font-weight:900;">{row['readiness_score']}</div>
        <div style="font-family:'Share Tech Mono',monospace;color:{bc};font-size:10px;letter-spacing:2px;">{row['rec_status']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["◈  AI ANALYSIS", "📋  TRAINING LOG", "📊  PERFORMANCE"])

    with tab1:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: st.plotly_chart(gauge_chart(row["readiness_score"], "READINESS"), use_container_width=True)
        with c2: st.plotly_chart(biometric_radar(row), use_container_width=True)
        with c3:
            st.markdown('<div class="section-title">AI DECISION</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="alert-box {status_css(row['rec_status'])}">
              {row['rec_icon']} <strong>{row['rec_label']}</strong><br>
              <span style="opacity:0.8;">{row['rec_condition']}</span><br>
              <span style="font-size:11px;">{row['rec_text']}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
            fc = {"LOW":"#00e676","MODERATE":"#ffd600","HIGH":"#ff1744"}.get(row["fatigue_level"],"#aaa")
            rc = risk_color(row["injury_risk"])
            st.markdown(f"""
            <div class="card">
              <div class="card-label">FATIGUE LEVEL</div>
              <div class="card-value" style="color:{fc};">{row['fatigue_level']}</div>
            </div>
            <div class="card" style="margin-top:8px;">
              <div class="card-label">INJURY RISK</div>
              <div class="card-value" style="color:{rc};">{row['injury_risk']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">BIOMETRIC READINGS</div>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("HEART RATE", f"{int(row['heart_rate'])} bpm")
        b2.metric("SLEEP", f"{row['sleep_hours']} hrs")
        b3.metric("TRAINING LOAD", f"{int(row['training_load'])}")
        b4.metric("BODY TEMP", f"{row.get('body_temp','—')}°C")
        b5.metric("RECOVERY IDX", f"{row['recovery_indicator']:.2f}")

    with tab2:
        if alog.empty:
            st.markdown('<div style="text-align:center;padding:40px;font-family:\'Share Tech Mono\',monospace;color:#1a3d2a;">NO TRAINING RECORDS FOUND</div>', unsafe_allow_html=True)
        else:
            # Summary stats
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("TOTAL SESSIONS", len(alog))
            s2.metric("COMPLETED", len(alog[alog["completed"]==True]))
            s3.metric("TOTAL HOURS", f"{total_training_hours(row['id'])}h")
            s4.metric("AVG PERFORMANCE", f"{avg_rating(row['id'])}/10")

            st.markdown('<div class="section-title">SESSION HISTORY</div>', unsafe_allow_html=True)

            intensity_colors = {"HIGH":"#ff1744","MODERATE":"#ffd600","LOW":"#00bfa5","REST":"#37574a"}
            for _, entry in alog.iterrows():
                ic = intensity_colors.get(entry["intensity"], "#37574a")
                completed_icon = "✅" if entry["completed"] else "⬜"
                rating_display = f"{entry['performance_rating']}/10" if entry.get("performance_rating",0) > 0 else "—"
                rating_color = "#69f0ae" if entry.get("performance_rating",0) >= 8 else "#ffd600" if entry.get("performance_rating",0) >= 5 else "#ff1744" if entry.get("performance_rating",0) > 0 else "#37574a"

                st.markdown(f"""
                <div class="training-card">
                  <div class="t-date">{entry['date']}</div>
                  <div style="flex:1;">
                    <div class="t-type">{completed_icon} {entry['training_type']}</div>
                    <div class="t-meta">
                      <span style="color:{ic};">{entry['intensity']}</span>
                      · {entry['duration_min']} min
                      · by {entry.get('assigned_by','—')}
                    </div>
                    {f'<div class="t-meta" style="margin-top:4px;color:#80cbc4;">{entry["notes"]}</div>' if pd.notna(entry.get("notes","")) and entry.get("notes","") != "" else ""}
                  </div>
                  <div class="t-rating" style="color:{rating_color};">{rating_display}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(performance_trend(alog), use_container_width=True)

        if not alog.empty:
            completed_log = alog[alog["completed"]==True]
            if not completed_log.empty:
                intensity_counts = completed_log["intensity"].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="section-title">TRAINING DISTRIBUTION</div>', unsafe_allow_html=True)
                    for intensity, count in intensity_counts.items():
                        ic = intensity_colors.get(intensity, "#37574a") if 'intensity_colors' in dir() else "#37574a"
                        pct = round(count / len(completed_log) * 100)
                        st.markdown(f"""
                        <div style="margin:6px 0;">
                          <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:10px;margin-bottom:3px;">
                            <span style="color:{ic};">{intensity}</span><span style="color:#37574a;">{count} sessions ({pct}%)</span>
                          </div>
                          <div style="background:#0a120e;height:4px;border-radius:0;">
                            <div style="background:{ic};height:4px;width:{pct}%;transition:width 0.5s;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="section-title">PERFORMANCE TREND</div>', unsafe_allow_html=True)
                    rated = completed_log[completed_log["performance_rating"] > 0]
                    if not rated.empty:
                        recent_avg = rated.tail(3)["performance_rating"].mean()
                        overall_avg = rated["performance_rating"].mean()
                        trend = "↑ IMPROVING" if recent_avg > overall_avg else "↓ DECLINING" if recent_avg < overall_avg else "→ STABLE"
                        trend_color = "#00e676" if "IMPROVING" in trend else "#ff1744" if "DECLINING" in trend else "#ffd600"
                        st.markdown(f"""
                        <div class="card">
                          <div class="card-label">OVERALL AVG</div>
                          <div class="card-value" style="color:#69f0ae;">{overall_avg:.1f}<span style="font-size:14px;color:#37574a;">/10</span></div>
                        </div>
                        <div class="card" style="margin-top:8px;">
                          <div class="card-label">RECENT TREND (LAST 3)</div>
                          <div class="card-value" style="color:{trend_color};font-size:16px;">{trend}</div>
                          <div class="card-sub">{recent_avg:.1f} avg last 3 sessions</div>
                        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ASSIGN TRAINING (Coach)
# ══════════════════════════════════════════════
elif page == "assign":
    st.markdown("## ASSIGN TRAINING SESSION")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-title">NEW TRAINING ASSIGNMENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)

        target_mode = st.radio("Assign to", ["Individual", "Entire Team"], horizontal=True)
        if target_mode == "Individual":
            target_athlete = st.selectbox("Select Operator", df["name"].tolist())
            target_ids = [df[df["name"]==target_athlete]["id"].values[0]]
        else:
            target_ids = df["id"].tolist()
            st.info(f"Will assign to all {len(target_ids)} operators")

        training_date = st.date_input("Training Date", value=date.today())
        training_type = st.selectbox("Training Type", [
            "Strength Training", "Endurance Run", "Sprint Intervals",
            "Tactical Drills", "Recovery Swim", "Recovery Yoga",
            "Goalkeeper Training", "Plyometrics", "Circuit Training",
            "Battlefield Simulation", "Marksmanship", "Combat Drills",
            "Rest Day", "Custom..."])

        if training_type == "Custom...":
            training_type = st.text_input("Custom Training Type")

        col_a, col_b = st.columns(2)
        with col_a:
            duration = st.slider("Duration (min)", 0, 180, 60, step=15)
        with col_b:
            intensity = st.selectbox("Intensity", ["HIGH", "MODERATE", "LOW", "REST"])

        notes = st.text_area("Session Notes / Instructions", placeholder="E.g., Focus on explosive power. 4 sets of 8 reps @ 75% 1RM. Rest 90s between sets.", height=100)
        coach_name = st.text_input("Assigned By", value="Coach Rivera")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("▶  ASSIGN TRAINING SESSION", use_container_width=True):
            if training_type:
                new_rows = []
                for tid in target_ids:
                    new_rows.append({
                        "athlete_id": tid, "date": str(training_date),
                        "training_type": training_type, "duration_min": duration,
                        "intensity": intensity, "notes": notes,
                        "assigned_by": coach_name, "completed": False,
                        "performance_rating": 0
                    })
                st.session_state.training_log = pd.concat(
                    [st.session_state.training_log, pd.DataFrame(new_rows)],
                    ignore_index=True)
                log = st.session_state.training_log
                names = [df[df["id"]==tid]["name"].values[0] for tid in target_ids]
                st.success(f"✅ Training assigned to: {', '.join(names)}")
                st.rerun()
            else:
                st.error("Please enter a training type.")

    with c2:
        st.markdown('<div class="section-title">TODAY\'S ASSIGNMENTS</div>', unsafe_allow_html=True)
        today_log = log[log["date"] == str(date.today())]

        if today_log.empty:
            st.markdown('<div style="padding:30px;text-align:center;font-family:\'Share Tech Mono\',monospace;color:#1a3d2a;border:1px dashed #0f2a1a;">NO ASSIGNMENTS FOR TODAY</div>', unsafe_allow_html=True)
        else:
            for _, entry in today_log.iterrows():
                athlete_row = df[df["id"]==entry["athlete_id"]]
                name = athlete_row["name"].values[0] if not athlete_row.empty else entry["athlete_id"]
                score = athlete_row["readiness_score"].values[0] if not athlete_row.empty else "—"
                rec_status = athlete_row["rec_status"].values[0] if not athlete_row.empty else "MODERATE"
                bc = badge_color(rec_status)
                ic = {"HIGH":"#ff1744","MODERATE":"#ffd600","LOW":"#00bfa5","REST":"#37574a"}.get(entry["intensity"],"#37574a")
                completed_icon = "✅" if entry["completed"] else "⏳"

                st.markdown(f"""
                <div class="training-card">
                  <div>
                    <div class="t-type">{completed_icon} {name}</div>
                    <div class="t-meta" style="margin-top:2px;">
                      {entry['training_type']} · <span style="color:{ic};">{entry['intensity']}</span> · {entry['duration_min']}min
                    </div>
                  </div>
                  <div style="margin-left:auto;text-align:right;">
                    <div style="font-family:'Orbitron',monospace;color:{bc};font-size:16px;">{score}</div>
                    <div style="font-family:'Share Tech Mono',monospace;color:{bc};font-size:9px;">{rec_status}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:16px;">AI RECOMMENDATIONS FOR TODAY</div>', unsafe_allow_html=True)
        for _, r in df.iterrows():
            bc = badge_color(r["rec_status"])
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:8px 12px;border-bottom:1px solid #0f2a1a;gap:12px;">
              <div style="font-family:'Share Tech Mono',monospace;color:#37574a;font-size:10px;min-width:70px;">{r['id']}</div>
              <div style="font-family:'Exo 2',sans-serif;color:#b2dfdb;font-size:12px;flex:1;">{r['name']}</div>
              <div style="font-family:'Share Tech Mono',monospace;color:{bc};font-size:10px;">{r['rec_icon']} {r['rec_status']}</div>
              <div style="font-family:'Orbitron',monospace;color:{bc};font-size:14px;">{r['readiness_score']}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════
elif page == "analytics":
    st.markdown("## TEAM ANALYTICS")

    st.plotly_chart(readiness_bar_chart(df), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(injury_risk_pie(df), use_container_width=True)
    with c2: st.plotly_chart(fatigue_scatter(df), use_container_width=True)

    st.markdown('<div class="section-title">SQUAD PERFORMANCE BREAKDOWN</div>', unsafe_allow_html=True)
    if "squad" in df.columns:
        for squad in df["squad"].unique():
            squad_df = df[df["squad"] == squad]
            avg = round(squad_df["readiness_score"].mean(), 1)
            best = squad_df.loc[squad_df["readiness_score"].idxmax(), "name"]
            risks = (squad_df["injury_risk"] == "HIGH").sum()
            color = "#00e676" if avg >= 70 else "#ffd600" if avg >= 50 else "#ff1744"
            st.markdown(f"""
            <div class="card" style="margin:6px 0;">
              <div style="display:flex;align-items:center;gap:16px;">
                <div>
                  <div class="card-label">SQUAD {squad}</div>
                  <div class="card-value" style="color:{color};">{avg}</div>
                  <div class="card-sub">avg readiness · {len(squad_df)} operators</div>
                </div>
                <div style="margin-left:auto;text-align:right;font-family:'Share Tech Mono',monospace;font-size:11px;">
                  <div style="color:#37574a;">TOP PERFORMER</div>
                  <div style="color:#69f0ae;">{best}</div>
                  <div style="color:#ff1744;margin-top:4px;">{"⚠ " + str(risks) + " HIGH RISK" if risks > 0 else "✅ NO HIGH RISK"}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ATHLETE PERSONAL DASHBOARD
# ══════════════════════════════════════════════
elif page == "dash":
    row = df[df["name"] == athlete_self].iloc[0]
    bc = badge_color(row["rec_status"])

    st.markdown(f"""
    <div class="profile-header">
      <div class="avatar">{initials(row['name'])}</div>
      <div>
        <div class="operator-name">{row['name']}</div>
        <div class="operator-meta">{row['id']} · {row['role'].upper()} · {row.get('position','—')}</div>
      </div>
      <div style="margin-left:auto;text-align:right;">
        <div style="font-family:'Share Tech Mono',monospace;color:#37574a;font-size:9px;letter-spacing:3px;">TODAY'S READINESS</div>
        <div style="font-family:'Orbitron',monospace;color:{bc};font-size:40px;font-weight:900;line-height:1;">{row['readiness_score']}</div>
        <div style="font-family:'Share Tech Mono',monospace;color:{bc};font-size:11px;letter-spacing:3px;">{row['rec_status']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(gauge_chart(row["readiness_score"]), use_container_width=True)
    with c2:
        st.plotly_chart(biometric_radar(row), use_container_width=True)

    st.markdown(f"""
    <div class="alert-box {status_css(row['rec_status'])}">
      {row['rec_icon']} <strong>{row['rec_label']}</strong><br>
      {row['rec_condition']}<br>
      <span style="font-size:11px;">{row['rec_text']}</span>
    </div>""", unsafe_allow_html=True)

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("HEART RATE", f"{int(row['heart_rate'])} bpm")
    b2.metric("SLEEP", f"{row['sleep_hours']} hrs")
    b3.metric("TRAINING LOAD", f"{int(row['training_load'])}")
    b4.metric("BODY TEMP", f"{row.get('body_temp','—')}°C")
    b5.metric("RECOVERY", f"{row['recovery_indicator']:.2f}")

    # Today's assigned training
    today_assigned = log[(log["athlete_id"]==row["id"]) & (log["date"]==str(date.today()))]
    if not today_assigned.empty:
        st.markdown('<div class="section-title">TODAY\'S ASSIGNED TRAINING</div>', unsafe_allow_html=True)
        for _, t in today_assigned.iterrows():
            ic = {"HIGH":"#ff1744","MODERATE":"#ffd600","LOW":"#00bfa5","REST":"#37574a"}.get(t["intensity"],"#37574a")
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;align-items:center;gap:16px;">
                <div>
                  <div style="font-family:'Orbitron',monospace;color:#b2dfdb;font-size:16px;">{t['training_type']}</div>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:10px;margin-top:4px;">
                    <span style="color:{ic};">{t['intensity']}</span> · {t['duration_min']} MIN · by {t.get('assigned_by','—')}
                  </div>
                  {f'<div style="font-family:\'Exo 2\',sans-serif;color:#80cbc4;font-size:12px;margin-top:8px;">{t["notes"]}</div>' if pd.notna(t.get("notes","")) and t.get("notes","") != "" else ""}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: MY TRAINING LOG (Athlete)
# ══════════════════════════════════════════════
elif page == "mylog":
    row = df[df["name"] == athlete_self].iloc[0]
    alog = get_athlete_log(row["id"])

    st.markdown(f"## TRAINING LOG — {row['name'].upper()}")

    if not alog.empty:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("TOTAL SESSIONS", len(alog))
        s2.metric("COMPLETED", len(alog[alog["completed"]==True]))
        s3.metric("TOTAL HOURS", f"{total_training_hours(row['id'])}h")
        s4.metric("AVG PERFORMANCE", f"{avg_rating(row['id'])}/10")

        st.plotly_chart(performance_trend(alog), use_container_width=True)

        st.markdown('<div class="section-title">SESSION HISTORY</div>', unsafe_allow_html=True)
        intensity_colors = {"HIGH":"#ff1744","MODERATE":"#ffd600","LOW":"#00bfa5","REST":"#37574a"}
        for _, entry in alog.iterrows():
            ic = intensity_colors.get(entry["intensity"], "#37574a")
            completed_icon = "✅" if entry["completed"] else "⬜"
            rating_display = f"{entry['performance_rating']}/10" if entry.get("performance_rating",0) > 0 else "—"
            rating_color = "#69f0ae" if entry.get("performance_rating",0) >= 8 else "#ffd600" if entry.get("performance_rating",0) >= 5 else "#ff1744" if entry.get("performance_rating",0) > 0 else "#37574a"
            st.markdown(f"""
            <div class="training-card">
              <div class="t-date">{entry['date']}</div>
              <div style="flex:1;">
                <div class="t-type">{completed_icon} {entry['training_type']}</div>
                <div class="t-meta"><span style="color:{ic};">{entry['intensity']}</span> · {entry['duration_min']} min</div>
                {f'<div class="t-meta" style="color:#80cbc4;margin-top:3px;">{entry["notes"]}</div>' if pd.notna(entry.get("notes","")) and entry.get("notes","") != "" else ""}
              </div>
              <div class="t-rating" style="color:{rating_color};">{rating_display}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:60px;font-family:\'Share Tech Mono\',monospace;color:#1a3d2a;border:1px dashed #0f2a1a;">NO TRAINING RECORDS FOUND</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: LOG WORKOUT (Athlete)
# ══════════════════════════════════════════════
elif page == "logworkout":
    row = df[df["name"] == athlete_self].iloc[0]
    st.markdown(f"## LOG WORKOUT — {row['name'].upper()}")

    # Pending sessions
    pending = log[(log["athlete_id"]==row["id"]) & (log["completed"]==False)]

    if not pending.empty:
        st.markdown('<div class="section-title">MARK SESSION AS COMPLETE</div>', unsafe_allow_html=True)
        for idx, entry in pending.iterrows():
            ic = {"HIGH":"#ff1744","MODERATE":"#ffd600","LOW":"#00bfa5","REST":"#37574a"}.get(entry["intensity"],"#37574a")
            with st.expander(f"⏳  {entry['date']} — {entry['training_type']} ({entry['intensity']})"):
                st.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;color:{ic};font-size:11px;">Intensity: {entry["intensity"]} · Duration: {entry["duration_min"]} min</div>', unsafe_allow_html=True)
                if pd.notna(entry.get("notes","")) and entry.get("notes","") != "":
                    st.markdown(f'<div style="font-family:\'Exo 2\',sans-serif;color:#80cbc4;font-size:12px;margin:8px 0;">{entry["notes"]}</div>', unsafe_allow_html=True)
                rating = st.slider(f"Performance Rating", 1, 10, 7, key=f"rating_{idx}")
                comments = st.text_input("Personal Notes", key=f"notes_{idx}", placeholder="How did it feel?")
                if st.button(f"✅ Mark Complete", key=f"complete_{idx}"):
                    st.session_state.training_log.loc[idx, "completed"] = True
                    st.session_state.training_log.loc[idx, "performance_rating"] = rating
                    if comments:
                        st.session_state.training_log.loc[idx, "notes"] = str(entry.get("notes","")) + f" | Athlete: {comments}"
                    st.success("Session marked as complete!")
                    st.rerun()

    st.markdown('<div class="section-title">LOG ADDITIONAL WORKOUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    w_type = st.selectbox("Workout Type", ["Strength Training","Endurance Run","Sprint Intervals","Flexibility","Swimming","Cycling","Sports Practice","Custom..."])
    if w_type == "Custom...":
        w_type = st.text_input("Custom type")
    wc1, wc2 = st.columns(2)
    with wc1:
        w_dur = st.slider("Duration (min)", 10, 180, 45, step=5)
    with wc2:
        w_intensity = st.selectbox("How intense?", ["HIGH","MODERATE","LOW"])
    w_rating = st.slider("Self Rating", 1, 10, 7)
    w_notes = st.text_area("Notes", placeholder="What did you do? How did you feel?", height=80)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾  SAVE WORKOUT", use_container_width=True):
        new_row = {"athlete_id": row["id"], "date": str(date.today()),
                   "training_type": w_type, "duration_min": w_dur,
                   "intensity": w_intensity, "notes": w_notes,
                   "assigned_by": "Self", "completed": True,
                   "performance_rating": w_rating}
        st.session_state.training_log = pd.concat(
            [st.session_state.training_log, pd.DataFrame([new_row])], ignore_index=True)
        st.success("✅ Workout logged successfully!")
        st.rerun()
