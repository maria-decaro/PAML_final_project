"""
DiaMetrics — Page 1: Personal Risk Calculator.

Mirrors the Figma mock: lifestyle inputs on the left, a risk gauge + a
ranked "contributing factors" breakdown on the right.
"""

import streamlit as st

from utils.predictor import (
    BMI_OPTIONS,
    EXERCISE_OPTIONS,
    SMOKING_OPTIONS,
    predict_risk,
)

# -----------------------------------------------------------------------
# Page config + styles
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="DiaMetrics — Personal Risk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Trim default Streamlit top padding so the title sits closer to top */
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      .dm-title {font-weight: 800; font-size: 2.15rem; letter-spacing: -0.015em;
                 color: #0F172A; margin-bottom: 0; line-height: 1.15;}
      .dm-title-accent {color: #2563EB; font-style: italic;}
      .dm-sub {color: #475569; margin-top: 0.3rem; font-size: 0.95rem;}
      .dm-section-head {font-size: 0.72rem; font-weight: 700; color: #64748B;
                        letter-spacing: 0.08em; text-transform: uppercase;
                        margin-bottom: 0.3rem; margin-top: 1.15rem;}
      .dm-card {background: #F8FAFC; border: 1px solid #E2E8F0;
                border-radius: 14px; padding: 1.2rem 1.4rem;}
      .dm-pill {display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
                font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
                text-transform: uppercase; margin-left: 0.5rem;}
      .pill-low  {background:#DCFCE7; color:#166534;}
      .pill-med  {background:#FEF3C7; color:#92400E;}
      .pill-high {background:#FEE2E2; color:#991B1B;}
      .dm-factor-row {display:flex; align-items:center; justify-content:space-between;
                      padding: 0.5rem 0; border-bottom:1px solid #E2E8F0;}
      .dm-factor-row:last-child {border-bottom: none;}
      .dm-factor-label {font-weight: 600; color:#0F172A; min-width:115px;}
      .dm-bar-wrap {flex:1; background:#E2E8F0; height:10px; border-radius:999px;
                    margin: 0 0.8rem; overflow:hidden;}
      .dm-bar-fill {height:100%; border-radius:999px;
                    transition: width 0.4s ease;}
      .dm-count-chip {background:#EEF2FF; color:#3730A3; font-weight:700;
                      font-size:0.75rem; padding:0.22rem 0.65rem; border-radius:999px;}
      div[data-testid="stSidebarNav"] li div a span {font-weight: 600;}
      /* Streamlit radio horizontal: tighten spacing */
      div[data-testid="stHorizontalBlock"] label {margin-right: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.markdown(
    '<div class="dm-title">Your Habits '
    '<span class="dm-title-accent">Mapped For The Future.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dm-sub">Six quick questions. See what is driving your '
    'score and what you can change.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

# =======================================================================
# LEFT COLUMN — lifestyle inputs
# =======================================================================
with left:
    # Counter placeholder — rendered now, filled in once we know the
    # final widget values (avoids the 1-tick lag bug).
    header_row = st.empty()

    st.markdown('<div class="dm-section-head">Exercise Frequency</div>',
                unsafe_allow_html=True)
    exer_label = st.radio(
        "Exercise Frequency",
        list(EXERCISE_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        index=None,
        key="exer_input",
    )

    st.markdown('<div class="dm-section-head">BMI Category</div>',
                unsafe_allow_html=True)
    bmi_label = st.radio(
        "BMI Category",
        list(BMI_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        index=None,
        key="bmi_input",
    )

    st.markdown('<div class="dm-section-head">Smoking Status</div>',
                unsafe_allow_html=True)
    smoker_label = st.radio(
        "Smoking Status",
        list(SMOKING_OPTIONS.keys()),
        horizontal=True,
        label_visibility="collapsed",
        index=None,
        key="smoker_input",
    )

    st.markdown('<div class="dm-section-head">Poor Mental Health Days / Month</div>',
                unsafe_allow_html=True)
    mental_days = st.slider(
        "Poor mental health days (last 30)",
        min_value=0, max_value=30, value=0, step=1,
        label_visibility="collapsed",
        key="mental_input",
    )

    st.markdown('<div class="dm-section-head">Sugary Drinks Per Day</div>',
                unsafe_allow_html=True)
    ssb = st.slider(
        "Sugar-sweetened drinks per day",
        min_value=0.0, max_value=5.0, value=0.0, step=0.5,
        label_visibility="collapsed",
        key="ssb_input",
    )

    st.markdown('<div class="dm-section-head">Alcohol Drinks Per Week</div>',
                unsafe_allow_html=True)
    alc = st.slider(
        "Alcoholic drinks per week",
        min_value=0, max_value=21, value=0, step=1,
        label_visibility="collapsed",
        key="alc_input",
    )

    # Sliders are always "answered" (they have a default); so count the
    # 3 radios plus 3 sliders = 6 total.
    completed = 3 + sum([
        exer_label is not None,
        bmi_label is not None,
        smoker_label is not None,
    ])

    # Now that we know the final count, render the header chip.
    header_row.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-top:0.2rem;">'
        f'<div class="dm-section-head" style="margin-top:0;">Your '
        f'Lifestyle Inputs</div>'
        f'<span class="dm-count-chip">{completed} / 6 completed</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.button(
        "Calculate my risk score  →",
        type="primary",
        use_container_width=True,
        disabled=(exer_label is None or bmi_label is None or smoker_label is None),
    )

    st.caption(
        "Disclaimer: This tool provides an estimate based on lifestyle factors. "
        "It is not a medical diagnosis."
    )

# =======================================================================
# RIGHT COLUMN — gauge + contributing factors
# =======================================================================
with right:
    st.markdown('<div class="dm-section-head">Your Risk Score</div>',
                unsafe_allow_html=True)

    ready = exer_label is not None and bmi_label is not None and smoker_label is not None

    if ready:
        pred = predict_risk(
            exercise=EXERCISE_OPTIONS[exer_label],
            bmi_category=BMI_OPTIONS[bmi_label],
            smoking_status=SMOKING_OPTIONS[smoker_label],
            mental_health_days=mental_days,
            sugary_drinks_per_day=ssb,
            alcohol_drinks_per_week=alc,
        )
    else:
        pred = None

    # ------ Gauge -------------------------------------------------------
    if pred is None:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            number={"suffix": "", "font": {"size": 46, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0,
                         "tickcolor": "#E2E8F0"},
                "bar": {"color": "#E2E8F0", "thickness": 0.25},
                "bgcolor": "#F8FAFC",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": "#F1F5F9"}],
            },
        ))
        fig.add_annotation(
            x=0.5, y=0.05, text="PLEASE INPUT YOUR DATA",
            showarrow=False, font=dict(size=11, color="#94A3B8"),
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        import plotly.graph_objects as go
        # Binary threshold at 50: <50 = green Low Risk, >=50 = red High Risk.
        is_high = pred.score >= 50
        color = "#EF4444" if is_high else "#22C55E"
        display_label = "High" if is_high else "Low"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred.score,
            number={"font": {"size": 54, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#CBD5E1"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#F8FAFC",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50],   "color": "#DCFCE7"},
                    {"range": [50, 100], "color": "#FEE2E2"},
                ],
            },
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        pill_cls = "pill-high" if is_high else "pill-low"
        st.markdown(
            f'<div style="text-align:center; margin-top:-20px;">'
            f'<span style="font-size:0.75rem; color:#64748B; '
            f'letter-spacing:0.08em;">RISK INDEX / 0-100</span><br/>'
            f'<span class="dm-pill {pill_cls}" style="font-size:0.9rem; '
            f'padding:0.35rem 0.9rem;">{display_label} risk</span></div>',
            unsafe_allow_html=True,
        )

    # Contributing factors panel removed by request — the binary risk pill
    # and gauge on their own are the deliverable for the Home page.

# Sidebar — tightened for the professional screenshot.
with st.sidebar:
    st.markdown("### DiaMetrics")
    st.caption("Lifestyle-based Type 2 diabetes risk screening.")
    st.markdown("---")
    st.markdown("**Pages**")
    st.caption("Home — Personal risk calculator")
    st.caption("Peek Through Our Data — Population dashboard")
    st.markdown("---")
    st.caption(
        "Source: BRFSS 2024 (CDC). Built for PAML final project, "
        "Spring 2026."
    )
