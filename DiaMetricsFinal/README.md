# DiaMetrics — Streamlit App

Personal diabetes risk calculator and BRFSS data explorer, built for the
PAML Final Project (Spring 2026).

## What's inside

```
DiaMetrics/
├── Home.py                               # Page 1 — Personal Risk Calculator
├── pages/
│   └── 2_Peek_Through_Our_Data.py        # Page 2 — Data exploration dashboard
├── utils/
│   ├── predictor.py                      # Risk scoring (swap in trained model here)
│   └── data_loader.py                    # Loads BRFSS CSV + fallback synthetic sample
├── data/
│   └── diabetes_dataset.csv              # Team BRFSS dataset (cleaned)
├── requirements.txt
└── README.md
```

## How to run (local)

1. Clone / download the folder.

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # macOS / Linux
   # .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:

   ```bash
   streamlit run Home.py
   ```

   Streamlit will print a local URL (default: `http://localhost:8501`).
   Open it in a browser. Use the sidebar to switch between **Home**
   (Risk Calculator) and **Peek Through Our Data**.

## How it wires into the team's models

The prediction logic lives in a single function:

```python
# utils/predictor.py
def predict_risk(exercise, bmi_category, smoking_status,
                 mental_health_days, sugary_drinks_per_day,
                 alcohol_drinks_per_week) -> Prediction
```

The feature signature matches the team schema exactly
(`EXERANY2, _BMI5CAT, _SMOKER3, MENTHLTH, SSBFRUT3, ALCDAY4`).

To swap in **Max's Logistic Regression** (the best-F1 model per team
testing), replace the body of `predict_risk` with:

```python
import numpy as np

# weights saved from LR.ipynb
_SAVED_LR = {
    "w": np.array([...]),     # copied from Max's notebook
    "b": ...,                 # bias
    "mu": np.array([...]),    # training mean (for standardization)
    "sigma": np.array([...]), # training std
}

def predict_risk(exercise, bmi_category, smoking_status,
                 mental_health_days, sugary_drinks_per_day,
                 alcohol_drinks_per_week):
    x = np.array([exercise, bmi_category, smoking_status,
                  mental_health_days, sugary_drinks_per_day,
                  alcohol_drinks_per_week], dtype=float)
    x_std = (x - _SAVED_LR["mu"]) / _SAVED_LR["sigma"]
    prob = 1 / (1 + np.exp(-(x_std @ _SAVED_LR["w"] + _SAVED_LR["b"])))
    score = float(prob) * 100
    # … build the Prediction return (same as current stub)
```

The notebooks already expose `predict_with_saved_lr` (LR), the `SVM`
class, and the `NaiveBayes` class with the exact same preprocessing
contract, so the hand-off is a copy-paste.

## Dataset

Reads `data/diabetes_dataset.csv` (the cleaned, NaN-dropped BRFSS 2024
extract produced by Nina's preprocessing notebook). If the file is
missing, the app falls back to a deterministic synthetic sample that
matches the column schema so the UI still renders for demo purposes.

## Troubleshooting

- `ModuleNotFoundError: utils.predictor` — run the app from the
  `DiaMetrics/` root, e.g. `streamlit run Home.py` (not
  `streamlit run pages/…`).
- Blank page / port in use — start on another port:
  `streamlit run Home.py --server.port 8502`.
- Pages don't show in the sidebar — the `pages/` folder must sit
  next to `Home.py`; filename prefix `2_` sets sidebar order.
