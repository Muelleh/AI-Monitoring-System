"""AI Analysis Engine v2"""
import numpy as np
import pandas as pd

WEIGHTS = {"heart_rate": 0.20, "sleep_hours": 0.30, "training_load": 0.25, "recovery_indicator": 0.25}

def _hr_score(hr):
    if hr <= 75: return max(0.0, 1.0 - abs(hr - 65) / 25)
    return max(0.0, 1.0 - (hr - 75) / 30)

def _sleep_score(h):
    if h >= 8.0: return 1.0
    return max(0.0, (h - 4.0) / 4.0)

def _load_score(l):
    if l <= 50: return 1.0
    return max(0.0, 1.0 - (l - 50) / 50)

def compute_readiness_score(heart_rate, sleep_hours, training_load, recovery_indicator):
    c = (_hr_score(heart_rate) * WEIGHTS["heart_rate"]
         + _sleep_score(sleep_hours) * WEIGHTS["sleep_hours"]
         + _load_score(training_load) * WEIGHTS["training_load"]
         + recovery_indicator * WEIGHTS["recovery_indicator"])
    return round(min(100.0, max(0.0, c * 100)), 1)

def classify_fatigue(score):
    if score >= 70: return "LOW"
    elif score >= 50: return "MODERATE"
    return "HIGH"

def classify_injury_risk(training_load, recovery_indicator, readiness_score):
    risk = (training_load / 100) * 0.40 + (1 - recovery_indicator) * 0.35 + (1 - readiness_score / 100) * 0.25
    if risk < 0.35: return "LOW"
    elif risk < 0.60: return "MEDIUM"
    return "HIGH"

def training_recommendation(score):
    if score >= 80:
        return {"status": "READY", "label": "READY FOR INTENSIVE TRAINING",
                "condition": "Excellent physical condition",
                "recommendation": "You are cleared for intensive training today.",
                "color": "#00ff88", "icon": "✅"}
    elif score >= 60:
        return {"status": "MODERATE", "label": "MODERATE TRAINING",
                "condition": "Slight fatigue detected",
                "recommendation": "Moderate intensity training recommended.",
                "color": "#ffd700", "icon": "⚡"}
    elif score >= 40:
        return {"status": "CAUTION", "label": "LIGHT TRAINING ONLY",
                "condition": "Fatigue detected",
                "recommendation": "Perform light training only. Monitor vitals.",
                "color": "#ff8c00", "icon": "⚠️"}
    return {"status": "RECOVERY", "label": "RECOVERY REQUIRED",
            "condition": "High fatigue detected",
            "recommendation": "Rest and recovery required. Avoid training today.",
            "color": "#ff3333", "icon": "🚨"}

def analyse_team(df):
    results = []
    for _, row in df.iterrows():
        score = compute_readiness_score(row["heart_rate"], row["sleep_hours"],
                                        row["training_load"], row["recovery_indicator"])
        fatigue = classify_fatigue(score)
        risk = classify_injury_risk(row["training_load"], row["recovery_indicator"], score)
        rec = training_recommendation(score)
        results.append({"readiness_score": score, "fatigue_level": fatigue,
                        "injury_risk": risk, "rec_status": rec["status"],
                        "rec_label": rec["label"], "rec_condition": rec["condition"],
                        "rec_text": rec["recommendation"], "rec_color": rec["color"],
                        "rec_icon": rec["icon"]})
    enriched = df.copy().reset_index(drop=True)
    return pd.concat([enriched, pd.DataFrame(results)], axis=1)
