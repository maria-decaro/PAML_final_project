# Midpoint Checkpoint — Beyza's Section

**Deliverable:** Photos/sketches of website user interface + instructions to
run the Streamlit code. (3 points)

---

## 1. Website UI — current state

The Streamlit app has both pages from the Figma specification implemented:

**Page 1 — Personal Risk Calculator** (`Home.py`)
- Left column: six lifestyle inputs (exercise frequency, BMI category,
  smoking status, poor mental health days, sugary drinks per day, alcohol
  drinks per week). Input-completion chip (`X / 6 completed`).
- Right column: live risk gauge (0–100) with color bands and a risk-level
  pill (`Low / Moderate / High / Very High`), plus a ranked
  "Contributing Factors" breakdown with per-factor percentage bars.
- Matches the Figma mock ("Your Habits — Mapped For The Future").

**Page 2 — Peek Through Our Data** (`pages/2_Peek_Through_Our_Data.py`)
- Three toggleable views (Risk factor distribution · Lifestyle group
  comparison · Age & risk trajectories).
- Top row of summary statistics (current smokers, obese BMI, inactive
  respondents, avg bad mental-health days).
- View 1: BMI distribution, smoking-status breakdown, mental-health-days
  frequency, weekly alcohol consumption.
- View 2: Diabetes prevalence grouped by BMI × exercise, smoking × diabetes,
  and mental-health × diabetes.
- View 3: Risk trajectories across BMI classes and sample composition
  charts.
- Dataset source and row count are displayed in the sidebar and header.

Figma reference (from Lina):
https://www.figma.com/design/OMBlLF2gZgE9HZ9UUhKyQe/paml-wireframe-draft

---

## 2. Instructions to run the Streamlit code

1. Clone or download the repo and `cd` into the project root:

   ```bash
   git clone https://github.com/maria-decaro/PAML_final_project.git
   cd PAML_final_project/DiaMetrics
   ```

2. (Recommended) create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate       # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:

   ```bash
   streamlit run Home.py
   ```

   Streamlit prints a local URL (default `http://localhost:8501`). Open it
   in the browser. Use the left sidebar to switch between the Risk
   Calculator (Home) and the Peek-Through-Our-Data explorer.

5. To stop the app: `Ctrl+C` in the terminal. To change port:
   `streamlit run Home.py --server.port 8502`.

---

## 3. Status vs. what is still pending

- Both pages render end-to-end and are connected to the BRFSS dataset
  (`diabetes_dataset.csv`) used by Nina's preprocessing.
- The risk predictor currently uses a placeholder clinically-weighted
  scorer. The function signature matches the team feature schema
  exactly, so swapping in Max's LR (or Maria's SVM / Rhia's NB) is a
  copy-paste of the saved weights into `utils/predictor.py`. Guide is in
  the README.
- Screenshots: captured locally by running `streamlit run Home.py` and
  taking screenshots of both pages (Page 1 form empty, Page 1 form
  filled with score, Page 2 each of the three views). Embed those PNGs
  below this section when pasting into the checkpoint document.

---

## 4. Screenshots (captured from the running app)

Screenshots of the live Streamlit app are in `DiaMetrics/screenshots/`.
Drop each image into the checkpoint document with the caption shown:

- `screenshots/page1-empty.png` — Fig 1: Risk Calculator landing state
  (before the user picks any radio options; sliders at default 0,
  chip reads 3 / 6 completed, gauge prompts "Please input your data").
- `screenshots/page1-filled.png` — Fig 2: Risk Calculator populated
  result — high-risk profile (Never exercises, Obese, Daily smoker,
  10 bad mental-health days, 2 sugary drinks/day, 3 alcohol
  drinks/week) yields a 56.5 / 100 High Risk score, 6 / 6 completed,
  ranked contributing factors BMI 41 %, Exercise 36 %, Smoking 23 %.
  Score is produced by Max's final Logistic Regression model
  (oversampling-trained, 64.67 % test accuracy) wired into
  `utils/predictor.py`.
- `screenshots/page2-view1.png` — Fig 3: Data Explorer · View 1 —
  Risk factor distribution (BMI classes, smoking status, mental
  health days, weekly alcohol).
- `screenshots/page2-view2.png` — Fig 4: Data Explorer · View 2 —
  Diabetes prevalence grouped by BMI × exercise, with smoking and
  mental-health breakdowns.
- `screenshots/page2-view3.png` — Fig 5: Data Explorer · View 3 —
  Risk trajectories across BMI classes by age band, plus sample
  composition by diabetes status.

All screenshots rendered against the real BRFSS dataset
(`diabetes_dataset.csv`, n = 100,520).
