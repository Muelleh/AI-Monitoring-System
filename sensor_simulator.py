"""
Sensor Simulator
Generates realistic randomised physiological data for testing.
"""

import numpy as np
import pandas as pd
from datetime import date


PROFILES = {
    "elite":    dict(hr=(58, 72),  sleep=(7.5, 9.0), load=(40, 75),  rec=(0.75, 1.00)),
    "standard": dict(hr=(65, 88),  sleep=(6.0, 8.0), load=(50, 85),  rec=(0.50, 0.85)),
    "fatigued": dict(hr=(85, 105), sleep=(3.5, 5.5), load=(75, 100), rec=(0.20, 0.55)),
}


def simulate_single(profile: str = "standard", seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    p   = PROFILES.get(profile, PROFILES["standard"])
    return {
        "heart_rate":         float(rng.integers(*[int(v) for v in p["hr"]])),
        "sleep_hours":        round(float(rng.uniform(*p["sleep"])), 1),
        "training_load":      float(rng.integers(*[int(v) for v in p["load"]])),
        "body_temp":          round(float(rng.uniform(36.4, 37.5)), 1),
        "recovery_indicator": round(float(rng.uniform(*p["rec"])), 2),
        "date":               str(date.today()),
    }


def simulate_team(n: int = 10,
                  names: list[str] | None = None,
                  seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    rows  = []
    profkeys = list(PROFILES.keys())

    default_names = [
        "Alpha-1", "Alpha-2", "Alpha-3", "Bravo-1", "Bravo-2",
        "Bravo-3", "Charlie-1", "Charlie-2", "Delta-1", "Delta-2",
    ]
    names = names or default_names

    for i in range(n):
        profile = rng.choice(profkeys, p=[0.3, 0.5, 0.2])
        row     = simulate_single(profile=profile, seed=int(rng.integers(0, 9999)))
        row["id"]   = f"S{i+1:02d}"
        row["name"] = names[i] if i < len(names) else f"Operator-{i+1:02d}"
        row["role"] = "Soldier" if i >= n // 2 else "Athlete"
        rows.append(row)

    cols = ["id", "name", "role", "heart_rate", "sleep_hours",
            "training_load", "body_temp", "recovery_indicator", "date"]
    return pd.DataFrame(rows)[cols]
