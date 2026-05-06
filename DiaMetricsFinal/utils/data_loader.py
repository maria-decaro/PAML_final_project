"""
Data loading helpers for DiaMetrics.

Looks for the team's BRFSS CSVs next to the Streamlit app. Falls back to
a deterministic synthetic sample that matches the real column schema and
roughly the BRFSS 2024 distributions described in the project appendix,
so Page 2 renders even before someone drops the real CSV in place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

FEATURE_COLS = ["EXERANY2", "_BMI5CAT", "_SMOKER3", "MENTHLTH", "SSBFRUT3", "ALCDAY4"]
TARGET_COL = "DIABETE4"

# Search order matches the team repo layout.
_CANDIDATE_FILES = [
    "diabetes_dataset.csv",
    "data_oversampling.csv",
    "data_smote.csv",
    "data_undersampling.csv",
]


def _repo_search_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here.parent,                             # DiaMetrics/
        here.parent.parent,                      # outputs/
        here.parent.parent / "repo" / "PAML_final_project-main",
        Path.cwd(),
        Path.cwd() / "data",
    ]


def find_dataset() -> Path | None:
    for base in _repo_search_paths():
        for name in _CANDIDATE_FILES:
            p = base / name
            if p.exists():
                return p
    return None


def _synthetic_sample(n: int = 8000, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic BRFSS-like sample."""
    rng = np.random.default_rng(seed)

    # ~23% no exercise
    exer = rng.choice([1, 2], size=n, p=[0.77, 0.23])

    # BMI: rough BRFSS distribution
    bmi = rng.choice([1, 2, 3, 4], size=n, p=[0.015, 0.29, 0.36, 0.335])

    # Smoking: 4-level
    smoker = rng.choice([1, 2, 3, 4], size=n, p=[0.10, 0.04, 0.26, 0.60])

    # Mental health days: 59% zero, rest exponential 1..30
    zero_days = rng.random(n) < 0.59
    heavy = rng.integers(1, 31, size=n)
    menthlth = np.where(zero_days, 88, heavy)

    # Sugary drinks per day proxy — most 888 (none), rest 1-3 per day
    ssb = np.full(n, 888)
    has_ssb = rng.random(n) < 0.55
    per_day = rng.integers(1, 4, size=n)
    ssb = np.where(has_ssb, 100 + per_day, ssb)

    # Alcohol: most 888, rest 1-10 per week
    alc = np.full(n, 888)
    has_alc = rng.random(n) < 0.45
    per_week = rng.integers(1, 11, size=n)
    alc = np.where(has_alc, 100 + per_week, alc)

    # Label: stronger risk for Obese + No exercise + Smoker + bad mental health
    risk_logit = (
        -2.3
        + 0.9 * (bmi == 4).astype(float)
        + 0.4 * (bmi == 3).astype(float)
        + 0.7 * (exer == 2).astype(float)
        + 0.6 * (smoker == 1).astype(float)
        + 0.3 * (smoker == 2).astype(float)
        + 0.02 * np.where(menthlth == 88, 0, menthlth)
    )
    prob = 1.0 / (1.0 + np.exp(-risk_logit))
    has_diabetes = (rng.random(n) < prob).astype(int)
    diabete = np.where(has_diabetes == 1, 1, 3)

    return pd.DataFrame({
        "EXERANY2": exer.astype(float),
        "_BMI5CAT": bmi.astype(float),
        "_SMOKER3": smoker.astype(float),
        "MENTHLTH": menthlth.astype(float),
        "SSBFRUT3": ssb.astype(float),
        "ALCDAY4": alc.astype(float),
        "DIABETE4": diabete.astype(float),
    })


def load_dataset() -> Tuple[pd.DataFrame, str]:
    """
    Load a BRFSS-style dataframe.

    Returns (df, source_label). source_label is the filename used, or
    'synthetic sample' when no real CSV was found.
    """
    p = find_dataset()
    if p is not None:
        df = pd.read_csv(p)
        needed = FEATURE_COLS + [TARGET_COL]
        df = df[[c for c in needed if c in df.columns]].dropna().copy()
        return df, p.name
    return _synthetic_sample(), "synthetic sample"


# --- Decoding helpers used by Page 2 visualizations ---------------------

BMI_LABEL = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}
SMOKER_LABEL = {1: "Everyday", 2: "Someday", 3: "Former", 4: "Never"}
EXER_LABEL = {1: "Active", 2: "Inactive"}
DIABETE_LABEL = {1: "Diabetes", 3: "No diabetes"}


def decode_menthlth(v: float) -> int:
    """BRFSS 88 means 0 days; everything else is literal 1..30."""
    v = int(v)
    if v == 88:
        return 0
    if 1 <= v <= 30:
        return v
    return 0  # 77/99 refused/don't know, drop-effect


def decode_ssb_per_week(v: float) -> float:
    """SSBFRUT3 -> servings per week (best-effort)."""
    v = int(v)
    if v == 888 or v == 0:
        return 0.0
    if 100 < v < 200:
        return (v - 100) * 7.0
    if 200 < v < 300:
        return float(v - 200)
    if 300 < v < 400:
        return (v - 300) / 4.33
    return 0.0


def decode_alc_per_week(v: float) -> float:
    v = int(v)
    if v == 888 or v == 0:
        return 0.0
    if 100 < v < 200:
        return float(v - 100)
    if 200 < v < 300:
        return (v - 200) / 4.33
    return 0.0
