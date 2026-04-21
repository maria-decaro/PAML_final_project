"""
DiaMetrics — risk prediction module.

Uses the final Logistic Regression model trained by Max on the
OVERSAMPLING-balanced BRFSS 2024 training set (LR.ipynb, cells 11-15).
The oversampling variant was selected because raw accuracy is not the
right metric for a rare-positive class: the imbalanced-class model hit
85.7% accuracy by predicting "no diabetes" almost always (recall ~1.2%).
The oversampling model reports 64.67% test accuracy but with much higher
recall, which is the meaningful quantity for a screening-style risk tool.

Feature schema (matches diabetes_dataset.csv and LR.ipynb):

    EXERANY2  : 1 = Yes exercise in last 30 days, 2 = No
    _BMI5CAT  : 1 Underweight, 2 Normal, 3 Overweight, 4 Obese
    _SMOKER3  : 1 Everyday, 2 Someday, 3 Former, 4 Never
    MENTHLTH  : 1-30 bad mental health days, 88 = 0 days
    SSBFRUT3  : 1XX = per day, 2XX = per week, 3XX = per month, 888 = none
    ALCDAY4   : 1XX = per week, 2XX = per month, 888 = none

Label: DIABETE4 -> 1 = diabetes present, 3 = no diabetes.

Reproduction check:
    The weights below were produced by re-running Max's exact training
    pipeline (random_state=1, lr=0.05, n_iter=5000, zero init) on
    data_oversampling.csv. Resulting test accuracy on the held-out
    original test split: 0.6467, matching the number reported for
    Max's final LR model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math

# ---------------------------------------------------------------------------
# Saved LR model — oversampling-trained (Max's final selection).
# Feature order: [EXERANY2, _BMI5CAT, _SMOKER3, MENTHLTH, SSBFRUT3, ALCDAY4]
# ---------------------------------------------------------------------------

_FEATURE_ORDER = ["EXERANY2", "_BMI5CAT", "_SMOKER3", "MENTHLTH", "SSBFRUT3", "ALCDAY4"]

_LR_W = [
    0.2933383587,   # EXERANY2
    0.5243185962,   # _BMI5CAT
   -0.1119669720,   # _SMOKER3
    0.0552399039,   # MENTHLTH
    0.3269493215,   # SSBFRUT3
    0.3513270717,   # ALCDAY4
]

_LR_B = -0.0012427149

_LR_MU = [
    1.2732692559,   # EXERANY2
    3.1552325835,   # _BMI5CAT
    3.3825522386,   # _SMOKER3
    57.0529510986,  # MENTHLTH
    632.3806182839, # SSBFRUT3
    571.5176891422, # ALCDAY4
]

_LR_SIGMA = [
    0.4456379357,
    0.8210959982,
    0.8824327695,
    38.0355852247,
    329.6071088787,
    355.0667279305,
]

# Human-readable name shown in the UI per feature index.
_DISPLAY_NAME = {
    "EXERANY2": "Exercise",
    "_BMI5CAT": "BMI",
    "_SMOKER3": "Smoking",
    "MENTHLTH": "Mental health",
    "SSBFRUT3": "Diet",
    "ALCDAY4":  "Alcohol",
}


@dataclass
class Prediction:
    score: float                      # 0 - 100 overall risk (sigmoid * 100)
    label: str                        # Low / Moderate / High / Very High
    contributions: Dict[str, float]   # per-factor log-odds contribution (signed)
    percent_contributions: Dict[str, float]  # per-factor % of total risk-increasing signal


# ---------------------------------------------------------------------------
# UI input -> BRFSS-code encoders.
# The LR was trained on raw BRFSS codes, so we convert the user-friendly
# UI values (days, drinks/day, drinks/week) into the same code space
# before scoring.
# ---------------------------------------------------------------------------

def _encode_menthlth(days: int) -> int:
    # UI slider 0..30. BRFSS uses 88 to mean "zero bad days".
    days = max(0, min(30, int(days)))
    return 88 if days == 0 else days


def _encode_ssbfrut3(drinks_per_day: float) -> int:
    # 0 -> 888 (never); otherwise 100 + int(drinks) capped 1..99 (per-day band).
    d = float(drinks_per_day)
    if d <= 0:
        return 888
    return 100 + int(min(99, max(1, round(d))))


def _encode_alcday4(drinks_per_week: float) -> int:
    # BRFSS ALCDAY4 weekly band is 101..107 (days/week drinking any alcohol).
    # We approximate from the UI "drinks per week" slider: 1-7 maps linearly
    # to 1-7 days/week; anything >=7 saturates at 107 (daily).
    d = float(drinks_per_week)
    if d <= 0:
        return 888
    days_band = min(7, max(1, int(round(d))))
    return 100 + days_band


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    # clip to avoid overflow
    if z > 500:
        z = 500
    elif z < -500:
        z = -500
    return 1.0 / (1.0 + math.exp(-z))


def predict_risk(
    exercise: int,
    bmi_category: int,
    smoking_status: int,
    mental_health_days: int,
    sugary_drinks_per_day: float,
    alcohol_drinks_per_week: float,
) -> Prediction:
    """
    Score one respondent with Max's oversampling-trained LR.

    The UI supplies user-friendly units; we re-encode them to the BRFSS
    codes the model was trained on, standardize with the saved training
    statistics, and apply the sigmoid to get a 0..1 probability of
    diabetes risk presence. Score is that probability times 100.

    Returns a Prediction with per-factor contribution breakdown for the
    ranked-bar UI. Contributions are per-feature log-odds shifts
    (w_i * (x_std_i)); percent_contributions normalizes the positive
    (risk-increasing) components so they sum to 100%, matching what the
    existing Home.py ranked-bar widget expects.
    """

    # Encode UI values -> BRFSS code space used at training time.
    x_raw = [
        float(int(exercise)),                     # EXERANY2 (1 or 2)
        float(int(bmi_category)),                 # _BMI5CAT (1..4)
        float(int(smoking_status)),               # _SMOKER3 (1..4)
        float(_encode_menthlth(mental_health_days)),      # MENTHLTH
        float(_encode_ssbfrut3(sugary_drinks_per_day)),   # SSBFRUT3
        float(_encode_alcday4(alcohol_drinks_per_week)),  # ALCDAY4
    ]

    # Standardize with saved training mu/sigma.
    x_std = [
        (x_raw[i] - _LR_MU[i]) / _LR_SIGMA[i]
        for i in range(6)
    ]

    # Linear log-odds and per-feature contributions.
    per_feat_logodds = [_LR_W[i] * x_std[i] for i in range(6)]
    z = sum(per_feat_logodds) + _LR_B
    prob = _sigmoid(z)
    score = round(prob * 100.0, 1)

    if score < 25:
        label = "Low"
    elif score < 50:
        label = "Moderate"
    elif score < 75:
        label = "High"
    else:
        label = "Very High"

    # Per-factor contribution dict (display names), signed log-odds shift.
    contributions: Dict[str, float] = {}
    for i, feat in enumerate(_FEATURE_ORDER):
        contributions[_DISPLAY_NAME[feat]] = per_feat_logodds[i]

    # Percent-of-risk breakdown: only count positive (risk-increasing)
    # contributions so the ranked bars sum to 100% when any factor pushes
    # risk upward. Factors pushing risk DOWN (e.g. regular exercise)
    # display as 0% here — their value is that they reduce the overall
    # score, not that they explain a slice of it.
    positive = {k: max(0.0, v) for k, v in contributions.items()}
    total_pos = sum(positive.values())
    if total_pos > 0:
        percent = {k: (v / total_pos) * 100.0 for k, v in positive.items()}
    else:
        percent = {k: 0.0 for k in contributions}

    return Prediction(
        score=score,
        label=label,
        contributions=contributions,
        percent_contributions=percent,
    )


# ---------------------------------------------------------------------------
# Human-readable option maps used by the UI layer.
# Kept identical to the previous version so Home.py needs no changes.
# ---------------------------------------------------------------------------

EXERCISE_OPTIONS = {
    "Daily": 1,
    "Several/week": 1,
    "Occasionally": 1,
    "Rarely": 2,
    "Never": 2,
}

BMI_OPTIONS = {
    "Underweight (<18.5)": 1,
    "Normal (18.5-24.9)": 2,
    "Overweight (25-29.9)": 3,
    "Obese (30+)": 4,
}

SMOKING_OPTIONS = {
    "Current — daily": 1,
    "Current — some days": 2,
    "Former smoker": 3,
    "Never smoked": 4,
}
