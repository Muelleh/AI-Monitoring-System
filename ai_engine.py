"""
AI Analysis Engine
Computes readiness score, fatigue level, injury risk, and training recommendation.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# SCORING WEIGHTS
# ──────────────────────────────────────────────
WEIGHTS = {
    "heart_rate":           0.20,
    "sleep_hours":          0.30,
    "training_load":        0.25,
    "recovery_indicator":   0.25,
}

HR_OPTIMAL_MIN  = 60
HR_OPTIMAL_MAX  = 75
HR_CRITICAL     = 100

SLEEP_OPTIMAL   = 8.0
SLEEP_CRITICAL  = 4.0

LOAD_OPTIMAL    = 50
LOAD_CRITICAL   = 95


# ──────────────────────────────────────────────
# HELPER: normalise individual metrics → 0-1
# ──────────────────────────────────────────────
def _hr_score(hr: float) -> float:
    if hr <= HR_OPTIMAL_MAX:
        return max(0.0, 1.0 - abs(hr - HR_OPTIMAL_MIN) / (HR_OPTIMAL_MIN + 10))
    return max(0.0, 1.0 - (hr - HR_OPTIMAL_MAX) / (HR_CRITICAL - HR_OPTIMAL_MAX))


def _sleep_score(hours: float) -> float:
    if hours >= SLEEP_OPTIMAL:
        return 1.0
    return max(0.0, (hours - SLEEP_CRITICAL) / (SLEEP_OPTIMAL - SLEEP_CRITICAL))


def _load_score(load: float) -> float:
    if load <= LOAD_OPTIMAL:
        return 1.0
    return max(0.0, 1.0 - (load - LOAD_OPTIMAL) / (LOAD_CRITICAL - LOAD_OPTIMAL))


# ──────────────────────────────────────────────
# CORE: compute readiness score 0-100
# ──────────────────────────────────────────────
def compute_readiness_score(heart_rate: float,
                            sleep_hours: float,
                            training_load: float,
                            recovery_indicator: float) -> float:
    composite = (
        _hr_score(heart_rate)         * WEIGHTS["heart_rate"]
        + _sleep_score(sleep_hours)   * WEIGHTS["sleep_hours"]
        + _load_score(training_load)  * WEIGHTS["training_load"]
        + recovery_indicator          * WEIGHTS["recovery_indicator"]
    )
    return round(min(100.0, max(0.0, composite * 100)), 1)


# ──────────────────────────────────────────────
# FATIGUE LEVEL
# ──────────────────────────────────────────────
def classify_fatigue(readiness_score: float) -> str:
    if readiness_score >= 70:
        return "LOW"
    elif readiness_score >= 50:
        return "MODERATE"
    return "HIGH"


# ──────────────────────────────────────────────
# INJURY RISK
# ──────────────────────────────────────────────
def classify_injury_risk(training_load: float,
                         recovery_indicator: float,
                         readiness_score: float) -> str:
    risk_score = (
        (training_load / 100) * 0.40
        + (1 - recovery_indicator) * 0.35
        + (1 - readiness_score / 100) * 0.25
    )
    if risk_score < 0.35:
        return "LOW"
    elif risk_score < 0.60:
        return "MEDIUM"
    return "HIGH"


# ──────────────────────────────────────────────
# TRAINING RECOMMENDATION
# ──────────────────────────────────────────────
def training_recommendation(readiness_score: float) -> dict:
    if readiness_score >= 80:
        return {
            "status":         "READY",
            "label":          "READY FOR INTENSIVE TRAINING",
            "condition":      "Excellent physical condition",
            "recommendation": "You are cleared for intensive training today.",
            "color":          "#00ff88",
            "icon":           "✅",
        }
    elif readiness_score >= 60:
        return {
            "status":         "MODERATE",
            "label":          "MODERATE TRAINING",
            "condition":      "Slight fatigue detected",
            "recommendation": "Moderate intensity training recommended.",
            "color":          "#ffd700",
            "icon":           "⚡",
        }
    elif readiness_score >= 40:
        return {
            "status":         "CAUTION",
            "label":          "LIGHT TRAINING ONLY",
            "condition":      "Fatigue detected",
            "recommendation": "Perform light training only. Monitor vitals.",
            "color":          "#ff8c00",
            "icon":           "⚠️",
        }
    return {
        "status":         "RECOVERY",
        "label":          "RECOVERY REQUIRED",
        "condition":      "High fatigue detected",
        "recommendation": "Rest and recovery required. Avoid training today.",
        "color":          "#ff3333",
        "icon":           "🚨",
    }


# ──────────────────────────────────────────────
# ANALYSE FULL DATAFRAME
# ──────────────────────────────────────────────
def analyse_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns:
        heart_rate, sleep_hours, training_load,
        recovery_indicator  (and optionally body_temp)
    Returns the same DataFrame enriched with AI columns.
    """
    results = []
    for _, row in df.iterrows():
        score   = compute_readiness_score(
                      row["heart_rate"],
                      row["sleep_hours"],
                      row["training_load"],
                      row["recovery_indicator"])
        fatigue = classify_fatigue(score)
        risk    = classify_injury_risk(
                      row["training_load"],
                      row["recovery_indicator"],
                      score)
        rec     = training_recommendation(score)
        results.append({
            "readiness_score":  score,
            "fatigue_level":    fatigue,
            "injury_risk":      risk,
            "rec_status":       rec["status"],
            "rec_label":        rec["label"],
            "rec_condition":    rec["condition"],
            "rec_text":         rec["recommendation"],
            "rec_color":        rec["color"],
            "rec_icon":         rec["icon"],
        })

    enriched = df.copy().reset_index(drop=True)
    enriched = pd.concat([enriched, pd.DataFrame(results)], axis=1)
    return enriched
