"""
╔══════════════════════════════════════════════════════════════════╗
║   AI ATHLETE & SOLDIER PERFORMANCE MONITORING SYSTEM            ║
║   Streamlit Dashboard — v1.0                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os
# Support both flat (Streamlit Cloud) and nested (local) structures
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
for p in [_dir, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from ai_engine        import analyse_team
from sensor_simulator import simulate_team, simulate_single
from charts           import (readiness_bar_chart, injury_risk_pie,
                               fatigue_scatter, gauge_chart, hr_sleep_bar)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PERF-AI MONITORING",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  — tactical dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
  --bg:        #060c06;
  --panel:     #0d150d;
  --border:    #1a3a1a;
  --green:     #00ff88;
  --yellow:    #ffd700;
  --orange:    #ff8c00;
  --red:       #ff3333;
  --muted:     #4a7a4a;
  --text:      #c8e6c9;
  --heading:   #00ff88;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 12px 16px !important;
}
[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--green) !important;
  font-size: 26px !important;
  font-weight: 700 !important;
  font-family: 'Share Tech Mono', monospace !important;
}

/* Headers */
h1, h2, h3 {
  font-family: 'Rajdhani', sans-serif !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
}
h1 { color: var(--green) !important; }
h2 { color: var(--text)  !important; }
h3 { color: var(--muted) !important; font-size: 13px !important; letter-spacing: 4px !important; }

/* Buttons */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  font-family: 'Share Tech Mono', monospace !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: var(--green) !important;
  color: var(--bg) !important;
}

/* Selectbox / inputs */
.stSelectbox > div > div,
.stSlider * {
  background: var(--panel) !important;
  color: var(--text) !important;
}

/* Dataframe */
.stDataFrame { border: 1px solid var(--border) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Alert boxes (custom) */
.alert-box {
  border-left: 3px solid;
  padding: 12px 16px;
  border-radius: 2px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 13px;
  letter-spacing: 1px;
  margin: 8px 0;
}
.alert-ready    { border-color: #00ff88; background: #00ff8811; color: #00ff88; }
.alert-moderate { border-color: #ffd700; background: #ffd70011; color: #ffd700; }
.alert-caution  { border-color: #ff8c00; background: #ff8c0011; color: #ff8c00; }
.alert-recovery { border-color: #ff3333; background: #ff333311; color: #ff3333; }

/* Score badge */
.score-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 2px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 13px;
  font-weight: bold;
  letter-spacing: 2px;
  border: 1px solid;
}

/* Top banner */
.sys-banner {
  border: 1px solid var(--border);
  background: var(--panel);
  padding: 16px 24px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  font-family: 'Share Tech Mono', monospace;
}
.sys-banner .title { color: var(--green); font-size: 22px; letter-spacing: 4px; }
.sys-banner .sub   { color: var(--muted); font-size: 11px; letter-spacing: 3px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--panel) !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"]      { color: var(--muted) !important; font-family: 'Share Tech Mono', monospace !important; }
.stTabs [aria-selected="true"]    { color: var(--green) !important; border-bottom: 2px solid var(--green) !important; }

/* Expander */
.streamlit-expanderHeader { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
_here = Path(__file__).parent
DATA_PATH = _here / "simulated_data.csv"
if not DATA_PATH.exists():
    DATA_PATH = _here.parent / "data" / "simulated_data.csv"


@st.cache_data(ttl=60)
def load_and_analyse(use_csv: bool = True) -> pd.DataFrame:
    if use_csv and DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = simulate_team(n=10)
    return analyse_team(df)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
STATUS_CSS = {
    "READY":    "alert-ready",
    "MODERATE": "alert-moderate",
    "CAUTION":  "alert-caution",
    "RECOVERY": "alert-recovery",
}

def alert_html(icon, label, condition, recommendation, status):
    css = STATUS_CSS.get(status, "alert-moderate")
    return f"""
    <div class="alert-box {css}">
      {icon} <strong>{label}</strong><br>
      <span style="opacity:0.8;">{condition}</span><br>
      <span style="font-size:12px;">{recommendation}</span>
    </div>"""

def score_badge(score, status):
    colors = {"READY":"#00ff88","MODERATE":"#ffd700","CAUTION":"#ff8c00","RECOVERY":"#ff3333"}
    c = colors.get(status, "#aaa")
    return f'<span class="score-badge" style="color:{c};border-color:{c};">{score}</span>'

def risk_badge(risk):
    c = {"LOW":"#00ff88","MEDIUM":"#ffd700","HIGH":"#ff3333"}.get(risk, "#aaa")
    return f'<span class="score-badge" style="color:{c};border-color:{c};font-size:11px;">{risk}</span>'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#00ff88;
                font-size:14px;letter-spacing:4px;padding:8px 0 4px;
                border-bottom:1px solid #1a3a1a;margin-bottom:16px;">
      ▋ PERF-AI SYS
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### USER ROLE")
    role = st.selectbox("Select Role", ["Coach / Commander", "Athlete / Soldier"],
                        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### NAVIGATION")
    if role == "Coach / Commander":
        page_options = ["Team Dashboard", "Sensor Simulator"]
    else:
        page_options = ["Athlete Dashboard", "Sensor Simulator"]

    page = st.radio("", page_options, label_visibility="collapsed")

    st.markdown("---")
    use_csv = st.checkbox("Use CSV data", value=True)
    if st.button("↻ REFRESH DATA"):
        st.cache_data.clear()

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                font-size:10px;letter-spacing:2px;margin-top:32px;">
      v1.0 · PROTOTYPE<br>
      AI MONITORING SYSTEM
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_and_analyse(use_csv=use_csv)


# ─────────────────────────────────────────────
# TOP BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="sys-banner">
  <div>
    <div class="title">⬡ PERF-AI MONITORING SYSTEM</div>
    <div class="sub">AI ATHLETE & SOLDIER PERFORMANCE ANALYTICS · {df['date'].iloc[0] if 'date' in df.columns else '—'}</div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div style="color:#00ff88;font-size:11px;letter-spacing:2px;">SYSTEM STATUS</div>
    <div style="color:#00ff88;font-size:18px;font-weight:bold;">● OPERATIONAL</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE: TEAM DASHBOARD
# ══════════════════════════════════════════════
if page == "Team Dashboard":

    st.markdown("## TEAM COMMAND DASHBOARD")

    # — KPI Row ———————————————————————————————
    high_risk    = (df["injury_risk"] == "HIGH").sum()
    recovery_req = (df["rec_status"] == "RECOVERY").sum()
    avg_score    = round(df["readiness_score"].mean(), 1)
    ready_pct    = round((df["rec_status"] == "READY").sum() / len(df) * 100)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TOTAL OPERATORS",  len(df))
    k2.metric("AVG READINESS",    f"{avg_score}")
    k3.metric("HIGH RISK ALERTS", high_risk,
              delta=f"{high_risk} require attention",
              delta_color="inverse")
    k4.metric("READY FOR OPS",    f"{ready_pct}%")

    # — Alerts ————————————————————————————————
    if high_risk > 0 or recovery_req > 0:
        with st.container():
            st.markdown("""
            <div style="border:1px solid #ff3333;background:#ff333308;
                        padding:12px;border-radius:2px;margin:12px 0;
                        font-family:'Share Tech Mono',monospace;color:#ff3333;">
              🚨 OPERATIONAL ALERTS DETECTED
            </div>""", unsafe_allow_html=True)
            cols = st.columns(min(high_risk + recovery_req, 4))
            alert_rows = df[df["injury_risk"] == "HIGH"]
            for i, (_, r) in enumerate(alert_rows.iterrows()):
                if i < len(cols):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="alert-box alert-recovery">
                          🚨 <strong>{r['name']}</strong><br>
                          Injury Risk: HIGH<br>
                          Score: {r['readiness_score']}
                        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # — Charts ————————————————————————————————
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(readiness_bar_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(injury_risk_pie(df), use_container_width=True)

    st.plotly_chart(fatigue_scatter(df), use_container_width=True)

    # — Team Table ————————————————————————————
    st.markdown("### ◈ OPERATOR STATUS MATRIX")

    display_df = df[["id","name","role","heart_rate","sleep_hours",
                     "training_load","readiness_score",
                     "fatigue_level","injury_risk","rec_label"]].copy()
    display_df.columns = ["ID","NAME","ROLE","HR","SLEEP","LOAD",
                          "READINESS","FATIGUE","RISK","RECOMMENDATION"]

    # colour-coded display
    def color_row(row):
        base = "background-color: #0d150d; color: #c8e6c9;"
        if row["RISK"] == "HIGH":
            return [base.replace("#0d150d","#1a0505")] * len(row)
        if row["FATIGUE"] == "HIGH":
            return [base.replace("#0d150d","#1a1000")] * len(row)
        return [base] * len(row)

    st.dataframe(display_df.style.apply(color_row, axis=1),
                 use_container_width=True, height=320)


# ══════════════════════════════════════════════
#  PAGE: ATHLETE DASHBOARD
# ══════════════════════════════════════════════
elif page == "Athlete Dashboard":

    st.markdown("## PERSONAL HEALTH DASHBOARD")

    # Operator selector
    operator_names = df["name"].tolist()
    selected_name  = st.selectbox("SELECT OPERATOR", operator_names)
    row = df[df["name"] == selected_name].iloc[0]

    st.markdown("---")

    # — Status header ————————————————————————
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:20px;padding:12px 0;">
      <div>
        <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                    font-size:11px;letter-spacing:3px;">OPERATOR ID</div>
        <div style="font-family:'Rajdhani',sans-serif;color:#00ff88;
                    font-size:24px;font-weight:700;">{row['id']} · {row['name']}</div>
        <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                    font-size:11px;letter-spacing:2px;">{row['role'].upper()}</div>
      </div>
      <div style="margin-left:auto;">
        {alert_html(row['rec_icon'], row['rec_label'], row['rec_condition'],
                    row['rec_text'], row['rec_status'])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # — Gauges ———————————————————————————————
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_chart(row["readiness_score"], "READINESS SCORE"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(hr_sleep_bar(row), use_container_width=True)

    # — Biometric KPIs ————————————————————————
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("HEART RATE",    f"{int(row['heart_rate'])} bpm")
    b2.metric("SLEEP",         f"{row['sleep_hours']} hrs")
    b3.metric("TRAINING LOAD", f"{int(row['training_load'])}")
    b4.metric("BODY TEMP",     f"{row['body_temp']}°C" if 'body_temp' in row else "—")
    b5.metric("RECOVERY IDX",  f"{row['recovery_indicator']:.2f}")

    st.markdown("---")

    # — AI Decision Cards ————————————————————
    st.markdown("### ◈ AI ANALYSIS REPORT")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""
        <div style="border:1px solid #1a3a1a;background:#0d150d;padding:16px;border-radius:2px;">
          <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                      font-size:10px;letter-spacing:3px;">READINESS SCORE</div>
          {score_badge(row['readiness_score'], row['rec_status'])}
        </div>""", unsafe_allow_html=True)
    with d2:
        fc = {"LOW":"#00ff88","MODERATE":"#ffd700","HIGH":"#ff3333"}.get(row["fatigue_level"],"#aaa")
        st.markdown(f"""
        <div style="border:1px solid #1a3a1a;background:#0d150d;padding:16px;border-radius:2px;">
          <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                      font-size:10px;letter-spacing:3px;">FATIGUE LEVEL</div>
          <span class="score-badge" style="color:{fc};border-color:{fc};">{row['fatigue_level']}</span>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div style="border:1px solid #1a3a1a;background:#0d150d;padding:16px;border-radius:2px;">
          <div style="font-family:'Share Tech Mono',monospace;color:#4a7a4a;
                      font-size:10px;letter-spacing:3px;">INJURY RISK</div>
          {risk_badge(row['injury_risk'])}
        </div>""", unsafe_allow_html=True)

    # — Full recommendation ———————————————————
    st.markdown(
        alert_html(row["rec_icon"], row["rec_label"],
                   row["rec_condition"], row["rec_text"], row["rec_status"]),
        unsafe_allow_html=True,
    )

    # — Debug expander ——————————————————————
    with st.expander("◉ RAW SENSOR DATA"):
        sensor_data = {
            "heart_rate":         row["heart_rate"],
            "sleep_hours":        row["sleep_hours"],
            "training_load":      row["training_load"],
            "body_temp":          row.get("body_temp", "N/A"),
            "recovery_indicator": row["recovery_indicator"],
            "date":               row.get("date", "N/A"),
        }
        st.json(sensor_data)


# ══════════════════════════════════════════════
#  PAGE: SENSOR SIMULATOR
# ══════════════════════════════════════════════
elif page == "Sensor Simulator":

    st.markdown("## SENSOR DATA SIMULATOR")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;color:#4a7a4a;'
        'font-size:12px;letter-spacing:2px;margin-bottom:20px;">'
        'GENERATE SYNTHETIC PHYSIOLOGICAL DATA FOR TESTING & VALIDATION'
        '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ◈ CONFIGURE SIMULATION")
        profile = st.selectbox("Operator Profile",
                               ["standard", "elite", "fatigued"],
                               help="Elite: well-rested top performer | Fatigued: overloaded")
        n_ops   = st.slider("Number of Operators", 2, 20, 10)
        sim_seed= st.number_input("Random Seed", value=42, step=1)

        if st.button("▶ RUN SIMULATION"):
            sim_df = simulate_team(n=n_ops, seed=int(sim_seed))
            sim_df = analyse_team(sim_df)
            st.session_state["sim_df"] = sim_df

        st.markdown("---")
        st.markdown("### ◈ MANUAL INPUT")
        with st.form("manual_input"):
            m_hr  = st.slider("Heart Rate (bpm)",  45, 120, 72)
            m_sl  = st.slider("Sleep Hours",       0.0, 10.0, 7.5, step=0.5)
            m_ld  = st.slider("Training Load",     0, 100, 60)
            m_rec = st.slider("Recovery Index",    0.0, 1.0, 0.75, step=0.05)
            submitted = st.form_submit_button("◆ ANALYSE")

        if submitted:
            from ai_engine import (compute_readiness_score,
                                         classify_fatigue,
                                         classify_injury_risk,
                                         training_recommendation)
            score = compute_readiness_score(m_hr, m_sl, m_ld, m_rec)
            fat   = classify_fatigue(score)
            risk  = classify_injury_risk(m_ld, m_rec, score)
            rec   = training_recommendation(score)

            st.markdown("### INSTANT ANALYSIS")
            st.plotly_chart(gauge_chart(score), use_container_width=True)
            st.markdown(f"**Fatigue:** {fat} &nbsp;&nbsp; **Risk:** {risk}")
            st.markdown(
                alert_html(rec["icon"], rec["label"],
                           rec["condition"], rec["recommendation"], rec["status"]),
                unsafe_allow_html=True)

    with col2:
        if "sim_df" in st.session_state:
            sim = st.session_state["sim_df"]
            st.markdown("### ◈ SIMULATION RESULTS")
            st.plotly_chart(readiness_bar_chart(sim), use_container_width=True)
            st.plotly_chart(injury_risk_pie(sim),     use_container_width=True)
            disp = sim[["id","name","role","readiness_score",
                        "fatigue_level","injury_risk","rec_label"]].copy()
            disp.columns = ["ID","NAME","ROLE","READINESS","FATIGUE","RISK","RECOMMENDATION"]
            st.dataframe(disp, use_container_width=True, height=280)
        else:
            st.markdown("""
            <div style="border:1px dashed #1a3a1a;padding:60px 20px;
                        text-align:center;border-radius:2px;
                        font-family:'Share Tech Mono',monospace;color:#2a5a2a;
                        font-size:13px;letter-spacing:2px;">
              ◈ RUN SIMULATION TO SEE RESULTS
            </div>""", unsafe_allow_html=True)
