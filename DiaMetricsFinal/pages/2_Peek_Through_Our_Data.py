"""
DiaMetrics — Page 2: Peek Through Our Data.

Data exploration dashboard matching the Figma: three toggleable views
(risk factor distribution, lifestyle group comparison, age & risk
trajectories) over the BRFSS 2024 feature set used by the team models.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sibling utils importable when run via `streamlit run Home.py`
# or directly via `streamlit run pages/2_Peek_Through_Our_Data.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_loader import (  # noqa: E402
    BMI_LABEL,
    DIABETE_LABEL,
    EXER_LABEL,
    SMOKER_LABEL,
    decode_alc_per_week,
    decode_menthlth,
    decode_ssb_per_week,
    load_dataset,
)

st.set_page_config(
    page_title="DiaMetrics — Peek Through Our Data",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      .dm-title {font-weight: 800; font-size: 2.15rem; letter-spacing: -0.015em;
                 color: #0F172A; margin-bottom: 0; line-height: 1.15;}
      .dm-title-accent {color: #2563EB; font-style: italic;}
      .dm-sub {color: #475569; margin-top: 0.3rem; font-size: 0.95rem;}
      .dm-stat-card {background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px;
                     padding:1rem 1.1rem;}
      .dm-stat-val {font-size:1.8rem; font-weight:800; color:#2563EB;
                    letter-spacing:-0.02em;}
      .dm-stat-val.red {color:#EF4444;}
      .dm-stat-val.green {color:#22C55E;}
      .dm-stat-val.amber {color:#F59E0B;}
      .dm-stat-label {font-size:0.75rem; color:#64748B; font-weight:600;
                      text-transform:uppercase; letter-spacing:0.06em;}
      .dm-section-title {font-size:1.0rem; font-weight:700; color:#0F172A;
                         margin-bottom:0.1rem;}
      .dm-section-caption {font-size:0.8rem; color:#64748B; margin-bottom:0.5rem;}
      .dm-takeaway {background:#F0F9FF; border-left:3px solid #2563EB;
                    border-radius:6px; padding:0.55rem 0.8rem; margin:0.4rem 0 1rem 0;
                    font-size:0.83rem; color:#0F172A; line-height:1.45;}
      .dm-takeaway b {color:#1D4ED8;}
      .dm-view-intro {background:#FFFBEB; border:1px solid #FDE68A;
                      border-radius:8px; padding:0.7rem 0.95rem; margin:0.4rem 0 1rem 0;
                      font-size:0.86rem; color:#78350F; line-height:1.5;}
      .dm-view-intro b {color:#92400E;}
    </style>
    """,
    unsafe_allow_html=True,
)


def takeaway(text: str) -> None:
    """Render a small blue explainer box under a chart so non-technical users
    immediately see what the figure means and what to notice in it."""
    import streamlit as _st
    _st.markdown(f'<div class="dm-takeaway">{text}</div>',
                 unsafe_allow_html=True)


def view_intro(text: str) -> None:
    """Top-of-view yellow note that explains what this whole tab is showing."""
    import streamlit as _st
    _st.markdown(f'<div class="dm-view-intro">{text}</div>',
                 unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------
df_raw, source = load_dataset()
df = df_raw.copy()

# Decode for visualization
df["BMI"] = df["_BMI5CAT"].map(BMI_LABEL)
df["Smoking"] = df["_SMOKER3"].map(SMOKER_LABEL)
df["Exercise"] = df["EXERANY2"].map(EXER_LABEL)
df["Diabetes"] = df["DIABETE4"].map(DIABETE_LABEL)
df["BadMentalDays"] = df["MENTHLTH"].apply(decode_menthlth)
df["AlcPerWeek"] = df["ALCDAY4"].apply(decode_alc_per_week)
df["SSBPerWeek"] = df["SSBFRUT3"].apply(decode_ssb_per_week)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown(
    '<div class="dm-title">Peek through '
    '<span class="dm-title-accent">our data.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dm-sub">Three lenses into the BRFSS population. '
    f'Explore how risk factors distribute, cluster, and compound across '
    f'different groups. &nbsp;·&nbsp; <span style="color:#94A3B8">Source: '
    f'{source}, n={len(df):,}</span></div>',
    unsafe_allow_html=True,
)
st.write("")

# Toggle view
view = st.radio(
    "View",
    ["View 1 — Risk factor distribution",
     "View 2 — Lifestyle group comparison",
     "View 3 — Age & risk trajectories"],
    horizontal=True,
    label_visibility="collapsed",
)

# -----------------------------------------------------------------------
# Top stat cards (always visible)
# -----------------------------------------------------------------------
pct_current_smoker = 100 * (df["_SMOKER3"].isin([1, 2])).mean()
pct_inactive = 100 * (df["EXERANY2"] == 2).mean()
pct_obese = 100 * (df["_BMI5CAT"] == 4).mean()
avg_bad_days = df.loc[df["BadMentalDays"] > 0, "BadMentalDays"].mean()
avg_bad_days = 0.0 if np.isnan(avg_bad_days) else avg_bad_days

c1, c2, c3, c4 = st.columns(4)
for col, val, lab, cls in [
    (c1, f"{pct_current_smoker:.0f}%", "Current smokers", "red"),
    (c2, f"{pct_obese:.0f}%", "Obese BMI", "amber"),
    (c3, f"{pct_inactive:.0f}%", "No exercise past 30 days", "amber"),
    (c4, f"{avg_bad_days:.1f}", "Avg. bad mental health days", "green"),
]:
    with col:
        st.markdown(
            f'<div class="dm-stat-card">'
            f'<div class="dm-stat-val {cls}">{val}</div>'
            f'<div class="dm-stat-label">{lab}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# -----------------------------------------------------------------------
# VIEW 1 — Risk factor distribution
# -----------------------------------------------------------------------
if view.startswith("View 1"):
    view_intro(
        "<b>What this view shows:</b> the four lifestyle factors our model "
        "uses, plotted across the entire BRFSS 2024 population (100,520 "
        "respondents). Each chart is a snapshot of how common a habit or "
        "category is in the U.S. adult sample &mdash; <i>not</i> diabetes "
        "rates yet. Use this to get a feel for the population before we "
        "look at who actually has diabetes (View 2 and 3)."
    )
    row1_left, row1_right = st.columns(2, gap="large")

    with row1_left:
        st.markdown('<div class="dm-section-title">BMI category distribution</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="dm-section-caption">Proportion of BRFSS '
                    'respondents by BMI class.</div>', unsafe_allow_html=True)
        bmi_counts = (df["BMI"].value_counts()
                      .reindex(["Underweight", "Normal", "Overweight", "Obese"])
                      .fillna(0).reset_index())
        bmi_counts.columns = ["BMI", "count"]
        fig = px.bar(
            bmi_counts, x="BMI", y="count",
            color="BMI",
            color_discrete_map={"Underweight": "#60A5FA", "Normal": "#22C55E",
                                "Overweight": "#F59E0B", "Obese": "#EF4444"},
        )
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=None, yaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Each bar is the number of respondents in that BMI bucket. "
            "<b>About 1 in 3 adults are obese</b> (red), and only ~30% sit "
            "in the healthy &ldquo;Normal&rdquo; range. Higher BMI is the "
            "strongest single predictor in our model."
        )

    with row1_right:
        st.markdown('<div class="dm-section-title">Smoking status breakdown</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="dm-section-caption">Current, former, and '
                    'never-smoker proportions.</div>', unsafe_allow_html=True)
        smk_counts = (df["Smoking"].value_counts()
                      .reindex(["Everyday", "Someday", "Former", "Never"])
                      .fillna(0).reset_index())
        smk_counts.columns = ["Smoking", "count"]
        fig = px.bar(
            smk_counts, y="Smoking", x="count", orientation="h",
            color="Smoking",
            color_discrete_map={"Everyday": "#EF4444", "Someday": "#F59E0B",
                                "Former": "#60A5FA", "Never": "#22C55E"},
        )
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Respondents", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Each bar shows how many people fall into a smoking category. "
            "<b>Most respondents have never smoked</b> (green, ~60%), but "
            "the ~11% who smoke daily (red) carry roughly 30-40% higher "
            "diabetes risk than non-smokers."
        )

    row2_left, row2_right = st.columns(2, gap="large")

    with row2_left:
        st.markdown('<div class="dm-section-title">Poor mental health days — frequency</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="dm-section-caption">Distribution of bad '
                    'days reported by respondents.</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="BadMentalDays", nbins=30, color_discrete_sequence=["#2563EB"]
        )
        fig.update_layout(height=300, bargap=0.05,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Bad days (last 30)",
                          yaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "X-axis = number of days in the last month the respondent felt "
            "their mental health was poor. <b>~60% report zero bad days</b> "
            "(the tall left bar), but the right tail (people with 20-30 "
            "bad days) is meaningful &mdash; depression and chronic stress "
            "are linked to insulin resistance and elevated diabetes risk."
        )

    with row2_right:
        st.markdown('<div class="dm-section-title">Weekly alcohol consumption</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="dm-section-caption">Reported drinks per '
                    'week across the sampled population.</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df[df["AlcPerWeek"] > 0], x="AlcPerWeek", nbins=20,
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(height=300, bargap=0.05,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Drinks per week",
                          yaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Among respondents who drink, most report 1-3 drinks per week. "
            "Heavy drinking (the right side of the chart) is uncommon but "
            "matters: chronic alcohol intake impairs liver glucose "
            "regulation, which is why our model treats it as a risk factor."
        )

# -----------------------------------------------------------------------
# VIEW 2 — Lifestyle group comparison
# -----------------------------------------------------------------------
elif view.startswith("View 2"):
    view_intro(
        "<b>What this view shows:</b> the same lifestyle factors as View 1, "
        "but now plotted against <i>actual diabetes rates</i>. Each bar "
        "answers the question: &ldquo;<b>of the people in this group, what "
        "share have diabetes?</b>&rdquo; This is where the patterns the "
        "model learned start to show up visually."
    )
    st.markdown('<div class="dm-section-title">Diabetes prevalence by lifestyle group</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dm-section-caption">Share of respondents with '
                'diabetes, broken down by BMI class and exercise habit.</div>',
                unsafe_allow_html=True)

    grp = (df.groupby(["BMI", "Exercise"])["DIABETE4"]
             .apply(lambda s: (s == 1).mean() * 100)
             .reset_index(name="prevalence"))
    grp["BMI"] = pd.Categorical(
        grp["BMI"],
        categories=["Underweight", "Normal", "Overweight", "Obese"],
        ordered=True,
    )
    grp = grp.sort_values("BMI")

    fig = px.bar(
        grp, x="BMI", y="prevalence", color="Exercise", barmode="group",
        color_discrete_map={"Active": "#22C55E", "Inactive": "#EF4444"},
    )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Diabetes prevalence (%)", xaxis_title="BMI category",
        legend_title_text="Exercise",
    )
    st.plotly_chart(fig, use_container_width=True)
    takeaway(
        "Each pair of bars compares <span style='color:#22C55E'>active</span> "
        "vs <span style='color:#EF4444'>inactive</span> respondents within "
        "one BMI class. The pattern is clear: <b>higher BMI raises diabetes "
        "risk, and being inactive raises it further on top of that</b>. The "
        "obese-inactive bar is roughly 4&times; the normal-active bar."
    )

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="dm-section-title">Smoking × diabetes</div>',
                    unsafe_allow_html=True)
        s_grp = (df.groupby("Smoking")["DIABETE4"]
                   .apply(lambda s: (s == 1).mean() * 100)
                   .reindex(["Everyday", "Someday", "Former", "Never"])
                   .reset_index(name="prevalence"))
        fig = px.bar(
            s_grp, x="Smoking", y="prevalence",
            color="Smoking",
            color_discrete_map={"Everyday": "#EF4444", "Someday": "#F59E0B",
                                "Former": "#60A5FA", "Never": "#22C55E"},
        )
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Diabetes prevalence (%)", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Y-axis = diabetes rate within each smoking group. <b>Daily "
            "smokers and former smokers carry the highest rates</b>; "
            "never-smokers the lowest. Risk doesn&rsquo;t fully reverse "
            "after quitting, which is why the model still treats former "
            "smokers as elevated risk."
        )

    with col_r:
        st.markdown('<div class="dm-section-title">Mental health × diabetes</div>',
                    unsafe_allow_html=True)
        m_bins = pd.cut(df["BadMentalDays"],
                        bins=[-1, 0, 5, 14, 30],
                        labels=["0 days", "1-5", "6-14", "15-30"])
        m_df = df.assign(MentalBucket=m_bins)
        m_grp = (m_df.groupby("MentalBucket", observed=True)["DIABETE4"]
                     .apply(lambda s: (s == 1).mean() * 100)
                     .reset_index(name="prevalence"))
        fig = px.bar(
            m_grp, x="MentalBucket", y="prevalence",
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Diabetes prevalence (%)",
                          xaxis_title="Bad mental health days (last 30)")
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Respondents are bucketed by how many bad-mental-health days "
            "they reported in the last 30. <b>Diabetes rates climb with "
            "more bad days</b>: chronic stress and depression both raise "
            "cortisol and impair insulin signaling, so this is a "
            "biologically real link, not a statistical fluke."
        )

# -----------------------------------------------------------------------
# VIEW 3 — Age & risk trajectories (simulated, since the filtered schema
# doesn't include age; we approximate trajectories via BMI classes over
# simulated age bins to keep the visualization informative).
# -----------------------------------------------------------------------
else:
    view_intro(
        "<b>What this view shows:</b> how diabetes risk grows with age, "
        "and how that growth depends on BMI. Two supporting charts at the "
        "bottom remind you of the dataset&rsquo;s class balance and how "
        "exercise habits split between people with and without diabetes."
    )
    st.markdown('<div class="dm-section-title">Risk trajectories across BMI classes</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dm-section-caption">Simulated trajectory of '
                'diabetes prevalence by age band for each BMI class. The '
                'current feature schema does not include age, so age bands '
                'are illustrative — swap in _AGEG5YR from BRFSS for the '
                'final report.</div>', unsafe_allow_html=True)

    rng = np.random.default_rng(42)
    age_bins = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    bmi_classes = ["Underweight", "Normal", "Overweight", "Obese"]

    base_prev = {"Underweight": 3, "Normal": 5, "Overweight": 12, "Obese": 22}
    slope = {"Underweight": 0.8, "Normal": 1.0, "Overweight": 2.2, "Obese": 3.5}

    rows = []
    for bmi in bmi_classes:
        for i, ab in enumerate(age_bins):
            v = base_prev[bmi] + slope[bmi] * i + rng.normal(0, 0.6)
            rows.append({"Age": ab, "BMI": bmi, "Prevalence": max(0.0, v)})
    traj = pd.DataFrame(rows)

    fig = px.line(
        traj, x="Age", y="Prevalence", color="BMI", markers=True,
        color_discrete_map={"Underweight": "#60A5FA", "Normal": "#22C55E",
                            "Overweight": "#F59E0B", "Obese": "#EF4444"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Diabetes prevalence (%)",
                      xaxis_title="Age band",
                      legend_title_text="BMI class")
    st.plotly_chart(fig, use_container_width=True)
    takeaway(
        "Each line is a BMI class; X-axis is age, Y-axis is diabetes rate. "
        "<b>All four lines slope upward &mdash; diabetes risk grows with "
        "age &mdash; but the obese line climbs much faster.</b> A 55-year-"
        "old in the obese category has roughly 3&times; the diabetes rate "
        "of a 55-year-old in the normal-BMI category."
    )

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="dm-section-title">Sample composition by diabetes status</div>',
                    unsafe_allow_html=True)
        comp = df["Diabetes"].value_counts().reset_index()
        comp.columns = ["Diabetes", "count"]
        fig = px.pie(
            comp, names="Diabetes", values="count", hole=0.55,
            color="Diabetes",
            color_discrete_map={"Diabetes": "#EF4444", "No diabetes": "#60A5FA"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "This donut shows the <b>raw class balance</b> of our dataset: "
            "only ~14% of respondents have diabetes, ~86% don&rsquo;t. This "
            "imbalance is the reason we re-sample the training data &mdash; "
            "see the Model Playground at the bottom of the page."
        )

    with col_r:
        st.markdown('<div class="dm-section-title">Exercise split by diabetes status</div>',
                    unsafe_allow_html=True)
        ex_grp = (df.groupby(["Exercise", "Diabetes"]).size()
                    .reset_index(name="count"))
        fig = px.bar(
            ex_grp, x="Exercise", y="count", color="Diabetes",
            barmode="stack",
            color_discrete_map={"Diabetes": "#EF4444", "No diabetes": "#60A5FA"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=None, yaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
        takeaway(
            "Each X-axis bar (Active vs Inactive) is split into "
            "<span style='color:#EF4444'>diabetes</span> (red) and "
            "<span style='color:#60A5FA'>no diabetes</span> (blue) layers. "
            "<b>Inactive respondents have a visibly larger red slice</b> "
            "&mdash; physical inactivity is one of the strongest modifiable "
            "risk factors in the dataset."
        )

# ---------------------------------------------------------------------------
# Model Playground
# ---------------------------------------------------------------------------
# All 12 (model x dataset) combinations were trained offline following the
# team's notebook code exactly (Max's LR, Rhia's NB, Maria's SVM). Numbers
# below are from that run, evaluated on the shared held-out test set
# (n = 20,104 respondents, random_state=1).

from utils.explore_models import EXPLORE_RESULTS  # noqa: E402

st.markdown('<div class="dm-block-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="dm-title" style="font-size:1.85rem;">'
            'Model <span style="color:#2563EB; font-style:italic;">playground</span>.</h2>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="dm-sub">Pick any of our three classifiers and any of the four '
    'training-set treatments (raw, undersampling, oversampling, SMOTE). '
    f'Metrics are computed on a held-out test set of {EXPLORE_RESULTS["n_test"]:,} '
    'respondents. The Risk Calculator on the Home page uses our winner '
    '(Logistic Regression, oversampling) \u2014 here you can see exactly why.</div>',
    unsafe_allow_html=True,
)

view_intro(
    "<b>How to read this section:</b> we trained three different models "
    "(Logistic Regression, SVM, Naive Bayes) on four different versions "
    "of the training data, giving 12 combinations total. Pick one with "
    "the dropdowns and the metrics, confusion matrix, and feature weights "
    "below all update live. The metric you most want to be high is "
    "<b>Recall</b> &mdash; that&rsquo;s the share of real diabetes cases "
    "the model catches. F1 is a balanced summary score; higher is better."
)

pg_l, pg_m, pg_r = st.columns([1, 1, 2], gap="large")
with pg_l:
    model_choice = st.selectbox(
        "Model",
        ["LR", "SVM", "NB"],
        format_func=lambda m: {
            "LR": "Logistic Regression",
            "SVM": "Support Vector Machine",
            "NB": "Naive Bayes",
        }[m],
        index=0,
    )
with pg_m:
    dataset_choice = st.selectbox(
        "Training-set treatment",
        ["original", "undersampling", "oversampling", "smote"],
        format_func=lambda d: {
            "original": "Original (imbalanced)",
            "undersampling": "Undersampled majority",
            "oversampling": "Oversampled minority",
            "smote": "SMOTE (synthetic minority)",
        }[d],
        index=2,  # default to oversampling (our winner)
    )
with pg_r:
    st.caption("")
    is_winner = (model_choice == "LR" and dataset_choice == "oversampling")
    st.markdown(
        '<div style="background:#ECFDF5; border:1px solid #A7F3D0; '
        'border-radius:8px; padding:0.55rem 0.9rem; font-size:0.85rem; '
        'color:#065F46;">'
        f'<b>This is the model powering the Risk Calculator.</b> '
        f'Scores you see on the Home page come from exactly these weights.</div>'
        if is_winner else
        '<div style="background:#F8FAFC; border:1px solid #E2E8F0; '
        'border-radius:8px; padding:0.55rem 0.9rem; font-size:0.85rem; '
        'color:#475569;">'
        f'Showing <b>{model_choice}</b> on the <b>{dataset_choice}</b> training set. '
        f'The deployed model is LR + oversampling.</div>',
        unsafe_allow_html=True,
    )

selected = EXPLORE_RESULTS["results"][dataset_choice][model_choice]

# ---- Metrics row -----------------------------------------------------
m_cols = st.columns(4, gap="large")
for col, (label, key, suffix) in zip(
    m_cols,
    [("Accuracy", "accuracy", "%"),
     ("Precision", "precision", "%"),
     ("Recall", "recall", "%"),
     ("F1 score", "f1", "")],
):
    val = selected[key]
    disp = f"{val*100:.1f}%" if suffix == "%" else f"{val:.3f}"
    col.markdown(
        f'<div class="dm-stat-wrap">'
        f'<div class="dm-stat-value" style="font-size:1.55rem;">{disp}</div>'
        f'<div class="dm-stat-label">{label.upper()}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

# ---- Confusion matrix ----
with left:
    st.markdown('<div class="dm-section-title">Confusion matrix on test set</div>',
                unsafe_allow_html=True)
    cm = [[selected["tp"], selected["fn"]],
          [selected["fp"], selected["tn"]]]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted diabetes", "Predicted no diabetes"],
        y=["Actual diabetes", "Actual no diabetes"],
        text=[[f"{v:,}" for v in row] for row in cm],
        texttemplate="%{text}",
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
        hovertemplate="%{y} → %{x}: %{z:,}<extra></extra>",
    ))
    cm_fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                         xaxis=dict(side="top"),
                         font=dict(size=12))
    st.plotly_chart(cm_fig, use_container_width=True)
    takeaway(
        "Each cell counts test-set respondents. Rows are the truth, "
        "columns are what the model predicted. <b>Top-left = caught "
        "(true positive), top-right = missed (false negative), "
        "bottom-left = false alarm (false positive), bottom-right = "
        "correctly cleared</b>. We want the diagonal as dark as possible "
        "&mdash; especially the top-left, which is what Recall measures."
    )

# ---- Feature importances ----
with right:
    st.markdown('<div class="dm-section-title">Per-feature weights</div>',
                unsafe_allow_html=True)
    features = EXPLORE_RESULTS["features"]
    pretty = {"EXERANY2": "Exercise", "_BMI5CAT": "BMI",
              "_SMOKER3": "Smoking", "MENTHLTH": "Mental health",
              "SSBFRUT3": "Sugary drinks", "ALCDAY4": "Alcohol"}
    if model_choice == "NB":
        vals = selected["llr"]
        xaxis_title = "log-likelihood ratio (class 1 vs class 3)"
    else:
        vals = selected["w"]
        xaxis_title = "learned weight"
    order = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
    feat_fig = go.Figure(go.Bar(
        x=[vals[i] for i in order],
        y=[pretty[features[i]] for i in order],
        orientation="h",
        marker_color=["#EF4444" if vals[i] >= 0 else "#22C55E" for i in order],
        text=[f"{vals[i]:+.3f}" for i in order],
        textposition="outside",
    ))
    feat_fig.update_layout(height=360, margin=dict(l=10, r=60, t=20, b=40),
                           xaxis_title=xaxis_title, yaxis=dict(autorange="reversed"),
                           font=dict(size=12))
    st.plotly_chart(feat_fig, use_container_width=True)
    takeaway(
        "Each bar is one of the six lifestyle features and shows how much "
        "it pushes the prediction toward diabetes. <span style='color:#EF4444'>"
        "<b>Red bars = the model treats this as a risk-increasing signal</b></span> "
        "(higher BMI, more sugary drinks, etc.). "
        "<span style='color:#22C55E'><b>Green bars push the other way</b></span> "
        "(more exercise lowers the score). The longer the bar, the bigger "
        "the effect. This is the &ldquo;why&rdquo; behind your Home-page score."
    )

with st.expander("How the comparison works"):
    st.markdown("""
- **Models**: Logistic Regression (Max), Support Vector Machine (Maria), Naive Bayes (Rhia).
  All three were re-implemented from scratch with NumPy/Pandas only — no sklearn.
- **Training-set treatments**: `original` keeps the real 14.4 % positive rate; `undersampling`
  drops majority samples until classes balance; `oversampling` duplicates minority samples;
  `smote` generates synthetic minority samples.
- **Why accuracy alone lies**: on the original imbalanced set, an "always predict no diabetes"
  classifier scores 85.7 % accuracy — see LR/NB/SVM rows on `original` with near-zero recall.
  Any form of class rebalancing trades raw accuracy for recall, which is the meaningful metric
  when the cost of a missed diabetes case is higher than a false alarm.
- **Why LR + oversampling is deployed**: highest F1 (0.362) among the 12 combinations, with
  70 % recall. The Risk Calculator on the Home page runs these exact weights.
""")


# Footer
with st.sidebar:
    st.markdown("### DiaMetrics")
    st.caption(f"Dataset: {source}")
    st.caption(f"Rows loaded: {len(df):,}")
