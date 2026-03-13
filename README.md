# ⬡ AI ATHLETE & SOLDIER PERFORMANCE MONITORING SYSTEM

> AI-powered physiological monitoring platform for coaches, commanders, athletes, and soldiers.

---

## 🎯 Project Overview

This system provides real-time (simulated) physiological monitoring and AI-driven training recommendations. It helps prevent overtraining, injury, and extreme fatigue through intelligent data analysis.

---

## 📁 Project Structure

```
ai-monitoring-system/
├── app/
│   └── app.py                  # Main Streamlit application
├── data/
│   └── simulated_data.csv      # Sample sensor data (10 operators)
├── model/
│   └── ai_engine.py            # Core AI analysis engine
├── utils/
│   ├── sensor_simulator.py     # Synthetic data generator
│   └── charts.py               # Plotly visualization module
└── requirements.txt
```

---

## 🚀 Installation & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
cd app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🧠 AI Engine Logic

### Readiness Score (0–100)
Computed from 4 weighted biometric inputs:

| Input               | Weight |
|---------------------|--------|
| Heart Rate          | 20%    |
| Sleep Hours         | 30%    |
| Training Load       | 25%    |
| Recovery Indicator  | 25%    |

Each metric is normalised to 0–1 before weighting.

### Score Interpretation
| Range   | Status   | Label                      |
|---------|----------|----------------------------|
| 80–100  | READY    | Ready for Intensive Training |
| 60–79   | MODERATE | Moderate Training          |
| 40–59   | CAUTION  | Light Training Only        |
| 0–39    | RECOVERY | Recovery Required          |

### Fatigue Level
- **LOW**: Score ≥ 70
- **MODERATE**: Score 50–69
- **HIGH**: Score < 50

### Injury Risk
Combines training load (40%), recovery deficit (35%), and readiness deficit (25%).

---

## 👥 User Roles

### Coach / Commander View
- Full team monitoring dashboard
- KPI metrics (total operators, avg readiness, high-risk alerts)
- Readiness bar chart, injury risk pie, fatigue scatter map
- Operator status matrix table with colour coding

### Athlete / Soldier View
- Personal health dashboard
- Readiness gauge + biometric bar chart
- Fatigue, injury risk, and training recommendation cards
- Raw sensor data expander

### Sensor Simulator
- Generate randomised teams (elite / standard / fatigued profiles)
- Manual input form for instant single-operator analysis

---

## 🎨 UI Design

- **Theme**: Tactical military dark UI
- **Colors**: Dark green, dark gray, status-coded alerts
- **Fonts**: Rajdhani (headings) + Share Tech Mono (data)
- **Charts**: Plotly (bar, pie, scatter, gauge)

---

## 🔮 Future Development Roadmap

### Phase 2 — Real Data Integration
- [ ] Connect to wearable device APIs (Garmin, Polar, WHOOP)
- [ ] Real-time WebSocket data streaming
- [ ] PostgreSQL / InfluxDB time-series storage

### Phase 3 — Advanced ML
- [ ] LSTM model for fatigue prediction over time
- [ ] Anomaly detection for sudden physiological changes
- [ ] Personalized baseline learning per operator

### Phase 4 — Enterprise Features
- [ ] User authentication (JWT)
- [ ] Mobile-responsive PWA
- [ ] Push notification alerts
- [ ] Export reports (PDF)
- [ ] Multi-unit / multi-team hierarchy

### Phase 5 — Clinical Grade
- [ ] FDA/CE marking pathway
- [ ] HIPAA compliance
- [ ] Integration with EHR systems
- [ ] Clinical validation studies

---

## ⚠️ Disclaimer

This is a **prototype / research tool**. It is not a medical device and should not replace professional medical judgment for training decisions.

---

*Built with Python · Streamlit · Plotly · Pandas · NumPy*
