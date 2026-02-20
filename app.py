import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(
    page_title="BurnoutIQ — Risk Analyzer",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:       #07080d;
  --card:     #0e1018;
  --card2:    #12151f;
  --border:   #1b1f2e;
  --accent:   #ff5c35;
  --accent-d: #c94220;
  --green:    #22c55e;
  --orange:   #f97316;
  --red:      #ef4444;
  --text:     #e6e8f0;
  --muted:    #505670;
  --label:    #7a859a;
}

html, body, .stApp            { background: var(--bg) !important; color: var(--text); }
#MainMenu, footer, header     { visibility: hidden; }

.block-container {
  max-width: 740px !important;
  padding: 2.2rem 1.6rem 5rem !important;
  margin: 0 auto !important;
}

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 2.6rem 1rem 1.8rem;
  margin-bottom: 1.8rem;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  background: rgba(255,92,53,.09);
  border: 1px solid rgba(255,92,53,.22);
  color: var(--accent);
  font-size: .68rem; font-weight: 600; letter-spacing: .13em;
  text-transform: uppercase; padding: .26rem .85rem;
  border-radius: 100px; margin-bottom: 1.1rem;
}
.hero h1 {
  font-family: 'Syne', sans-serif !important;
  font-size: 2.4rem !important; font-weight: 800 !important;
  line-height: 1.1 !important; letter-spacing: -.025em;
  margin: 0 0 .65rem !important; color: var(--text) !important;
}
.hero h1 em { color: var(--accent); font-style: normal; }
.hero p {
  font-size: .87rem; color: var(--muted);
  line-height: 1.75; max-width: 400px; margin: 0 auto;
}

/* ── Section card ── */
.s-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.5rem 1.2rem;
  margin-bottom: 1rem;
}
.s-title {
  font-family: 'Syne', sans-serif;
  font-size: .68rem; font-weight: 700;
  letter-spacing: .11em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 1rem;
  padding-bottom: .45rem;
  border-bottom: 1px solid rgba(255,92,53,.13);
}

/* ── Widget resets ── */
div[data-testid="stForm"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
  background: var(--card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
  color: var(--text) !important;
  font-size: .86rem !important;
  transition: border-color .18s, box-shadow .18s;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within > input {
  border-color: rgba(255,92,53,.5) !important;
  box-shadow: 0 0 0 3px rgba(255,92,53,.08) !important;
}
div[data-baseweb="popover"] ul { background: var(--card2) !important; }
li[role="option"] { color: var(--text) !important; }
li[role="option"]:hover { background: rgba(255,92,53,.08) !important; }
label {
  color: var(--label) !important;
  font-size: .73rem !important;
  font-weight: 500 !important;
  letter-spacing: .05em !important;
  text-transform: uppercase !important;
}

/* ── Tooltip override ── */
.stTooltipIcon { color: var(--muted) !important; }

/* ── Submit button ── */
.stFormSubmitButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-d) 100%) !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-size: .92rem !important; font-weight: 700 !important;
  letter-spacing: .04em !important;
  border: none !important;
  border-radius: 11px !important;
  padding: .82rem 1.5rem !important;
  box-shadow: 0 4px 22px rgba(255,92,53,.28) !important;
  transition: all .2s !important;
}
.stFormSubmitButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 30px rgba(255,92,53,.42) !important;
}

/* ── Reset button ── */
.stButton > button {
  width: 100% !important;
  background: transparent !important;
  color: var(--label) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .84rem !important; font-weight: 500 !important;
  border: 1px solid var(--border) !important;
  border-radius: 11px !important;
  padding: .72rem 1.5rem !important;
  transition: all .18s !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: rgba(255,92,53,.05) !important;
}

/* ── Divider ── */
.gap { margin: .85rem 0; }

/* ── Result ── */
.result-wrap {
  border-radius: 16px; padding: 1.7rem 1.7rem 1.3rem;
  margin-top: 1.4rem; position: relative; overflow: hidden;
  animation: fadeUp .45s ease forwards;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(14px); }
  to   { opacity:1; transform:translateY(0); }
}
.res-low  { background:linear-gradient(135deg,rgba(34,197,94,.1),rgba(34,197,94,.03));  border:1px solid rgba(34,197,94,.22);  }
.res-mod  { background:linear-gradient(135deg,rgba(249,115,22,.1),rgba(249,115,22,.03)); border:1px solid rgba(249,115,22,.22); }
.res-high { background:linear-gradient(135deg,rgba(239,68,68,.12),rgba(239,68,68,.03)); border:1px solid rgba(239,68,68,.28); }

.res-glow {
  position:absolute; top:-70px; right:-70px;
  width:200px; height:200px; border-radius:50%;
  pointer-events:none; filter:blur(55px); opacity:.3;
}
.res-tag   { font-size:.67rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:.3rem; }
.res-score { font-family:'Syne',sans-serif; font-weight:800; font-size:3.5rem; line-height:1; letter-spacing:-.03em; }
.res-denom { font-size:1.2rem; font-weight:400; opacity:.38; }
.res-level { font-family:'Syne',sans-serif; font-weight:700; font-size:.95rem; margin-top:.45rem; letter-spacing:.05em; text-transform:uppercase; }

.bar-track { background:rgba(255,255,255,.06); border-radius:100px; height:7px; margin:1.1rem 0 .3rem; overflow:hidden; }
.bar-fill  { height:100%; border-radius:100px; transition:width .9s cubic-bezier(.16,1,.3,1); }
.bar-labels{ display:flex; justify-content:space-between; font-size:.68rem; color:var(--muted); }

.insight {
  display:flex; gap:.75rem; align-items:flex-start;
  background:rgba(255,255,255,.025); border:1px solid var(--border);
  border-radius:11px; padding:.95rem 1.1rem; margin-top:.75rem;
  animation: fadeUp .55s ease forwards;
}
.insight-icon { font-size:1rem; flex-shrink:0; margin-top:.05rem; }
.insight-body { font-size:.82rem; color:#8d97b0; line-height:1.6; }
.insight-body strong { color:var(--text); font-weight:600; }

/* ── Indicators row ── */
.indicators {
  display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem;
  margin-top:.75rem; animation: fadeUp .6s ease forwards;
}
.ind-box {
  background:rgba(255,255,255,.025); border:1px solid var(--border);
  border-radius:11px; padding:.85rem .9rem; text-align:center;
}
.ind-val { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; line-height:1; }
.ind-lbl { font-size:.67rem; color:var(--muted); margin-top:.25rem; letter-spacing:.05em; text-transform:uppercase; }

/* ── Footer ── */
.foot {
  text-align:center; margin-top:3rem; padding-top:1.3rem;
  border-top:1px solid var(--border);
  font-size:.7rem; color:var(--muted); line-height:2.2;
}
.foot strong { color:var(--accent); font-family:'Syne',sans-serif; letter-spacing:.03em; }

/* scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    job_role          = "Data Scientist",
    experience_years  = 5,
    deadline_pressure = "Low",
    work_life_balance = 5,
    ai_tool_hours     = 10.0,
    manual_work_hours = 20.0,
    meeting_hours     = 5.0,
    collab_hours      = 8.0,
    learning_time     = 3.0,
    focus_hours       = 4.0,
    tasks_automated   = 30.0,
    error_rate        = 3.0,
    task_complexity   = 5,
    show_result       = False,
    result_score      = 0.0,
)

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv("ai_productivity_features.csv")
    df = df.drop(columns=["Employee_ID"])
    df["deadline_pressure_level"] = pd.Categorical(
        df["deadline_pressure_level"], categories=["Low","Medium","High"], ordered=True
    ).codes
    df = pd.get_dummies(df, columns=["job_role"], drop_first=True)
    X = df.drop("burnout_risk_score", axis=1)
    y = df["burnout_risk_score"]
    num_cols = [
        "experience_years","ai_tool_usage_hours_per_week","tasks_automated_percent",
        "manual_work_hours_per_week","learning_time_hours_per_week","deadline_pressure_level",
        "meeting_hours_per_week","collaboration_hours_per_week","error_rate_percent",
        "task_complexity_score","focus_hours_per_day","work_life_balance_score",
    ]
    sc = StandardScaler()
    Xs = X.copy(); Xs[num_cols] = sc.fit_transform(X[num_cols])
    mdl = RandomForestRegressor(n_estimators=200, random_state=42)
    mdl.fit(Xs, y)
    return mdl, sc, X.columns.tolist(), num_cols

PRESSURE_MAP     = {"Low": 0, "Medium": 1, "High": 2}
JOB_ROLE_DUMMIES = ["Designer","Developer","HR","Manager","Marketing"]


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🔥 AI-Powered Analytics</div>
  <h1>Employee <em>Burnout</em><br>Risk Analyzer</h1>
  <p>Enter employee details below and get an instant burnout risk score powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────────────────────────────────────
with st.form("burnout_form", clear_on_submit=False):

    # — Employee Profile —
    st.markdown('<div class="s-card"><div class="s-title">👤 Employee Profile</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        job_role = st.selectbox("Job Role",
            ["Data Scientist","Designer","Developer","HR","Manager","Marketing"],
            index=["Data Scientist","Designer","Developer","HR","Manager","Marketing"].index(st.session_state.job_role))
        experience_years = st.number_input("Experience (Years)", 0, 40,
            value=st.session_state.experience_years)
    with r1c2:
        deadline_pressure = st.selectbox("Deadline Pressure",
            ["Low","Medium","High"],
            index=["Low","Medium","High"].index(st.session_state.deadline_pressure))
        work_life_balance = st.number_input("Work-Life Balance (1–10)", 1, 10,
            value=st.session_state.work_life_balance,
            help="1 = Very poor balance, 10 = Excellent balance")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # — Time Allocation —
    st.markdown('<div class="s-card"><div class="s-title">⏱ Weekly Time Allocation (hours)</div>', unsafe_allow_html=True)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        ai_tool_hours     = st.number_input("AI Tool Usage", 0.0, 60.0,
            value=st.session_state.ai_tool_hours, step=0.5,
            help="Hours/week spent using AI tools")
        manual_work_hours = st.number_input("Manual Work", 0.0, 60.0,
            value=st.session_state.manual_work_hours, step=0.5,
            help="Hours/week on manual, non-automated tasks")
    with r2c2:
        meeting_hours     = st.number_input("Meetings", 0.0, 30.0,
            value=st.session_state.meeting_hours, step=0.5)
        collab_hours      = st.number_input("Collaboration", 0.0, 30.0,
            value=st.session_state.collab_hours, step=0.5)
    with r2c3:
        learning_time     = st.number_input("Learning & Dev", 0.0, 20.0,
            value=st.session_state.learning_time, step=0.5)
        focus_hours       = st.number_input("Focus (hrs/day)", 0.0, 12.0,
            value=st.session_state.focus_hours, step=0.5,
            help="Uninterrupted deep work hours per day")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    # — Performance —
    st.markdown('<div class="s-card"><div class="s-title">📊 Performance Indicators</div>', unsafe_allow_html=True)
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        tasks_automated = st.number_input("Tasks Automated (%)", 0.0, 100.0,
            value=st.session_state.tasks_automated, step=1.0,
            help="% of weekly tasks handled by automation/AI")
    with r3c2:
        error_rate      = st.number_input("Error Rate (%)", 0.0, 20.0,
            value=st.session_state.error_rate, step=0.1,
            help="Percentage of tasks with errors or rework needed")
    with r3c3:
        task_complexity = st.number_input("Task Complexity (1–10)", 1, 10,
            value=st.session_state.task_complexity,
            help="1 = Very simple, 10 = Extremely complex tasks")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("⚡  Predict Burnout Risk Score", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reset button (outside form)
# ─────────────────────────────────────────────────────────────────────────────
def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.button("↺  Reset to Defaults", on_click=reset_all, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    # Save values to session
    st.session_state.job_role          = job_role
    st.session_state.experience_years  = experience_years
    st.session_state.deadline_pressure = deadline_pressure
    st.session_state.work_life_balance = work_life_balance
    st.session_state.ai_tool_hours     = ai_tool_hours
    st.session_state.manual_work_hours = manual_work_hours
    st.session_state.meeting_hours     = meeting_hours
    st.session_state.collab_hours      = collab_hours
    st.session_state.learning_time     = learning_time
    st.session_state.focus_hours       = focus_hours
    st.session_state.tasks_automated   = tasks_automated
    st.session_state.error_rate        = error_rate
    st.session_state.task_complexity   = task_complexity

    if not os.path.exists("ai_productivity_features.csv"):
        st.error("❌ `ai_productivity_features.csv` not found. Place it in the same folder as app.py.")
    else:
        with st.spinner("Analyzing employee data…"):
            mdl, sc, cols, num_cols = train_model()

        row = {
            "experience_years":              experience_years,
            "ai_tool_usage_hours_per_week":  ai_tool_hours,
            "tasks_automated_percent":       tasks_automated,
            "manual_work_hours_per_week":    manual_work_hours,
            "learning_time_hours_per_week":  learning_time,
            "deadline_pressure_level":       PRESSURE_MAP[deadline_pressure],
            "meeting_hours_per_week":        meeting_hours,
            "collaboration_hours_per_week":  collab_hours,
            "error_rate_percent":            error_rate,
            "task_complexity_score":         task_complexity,
            "focus_hours_per_day":           focus_hours,
            "work_life_balance_score":       work_life_balance,
        }
        for role in JOB_ROLE_DUMMIES:
            row[f"job_role_{role}"] = 1 if job_role == role else 0

        inp = pd.DataFrame([row])
        for col in cols:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[cols]
        inp[num_cols] = sc.transform(inp[num_cols])

        score = float(np.clip(mdl.predict(inp)[0], 0, 10))
        st.session_state.show_result  = True
        st.session_state.result_score = score

# ─────────────────────────────────────────────────────────────────────────────
# Result display (persists between reruns)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.show_result:
    score = st.session_state.result_score
    pct   = int(score * 10)

    # Metrics
    total_hrs   = st.session_state.meeting_hours + st.session_state.collab_hours + st.session_state.manual_work_hours + st.session_state.ai_tool_hours
    auto_ratio  = f"{int(st.session_state.tasks_automated)}%"
    balance_lbl = "Good" if st.session_state.work_life_balance >= 7 else ("Fair" if st.session_state.work_life_balance >= 4 else "Poor")

    if score < 4:
        res_cls  = "res-low";  color = "#22c55e"; glow = "#22c55e"
        level    = "Low Risk";  icon = "✅"
        advice   = ("Workload and lifestyle indicators look healthy. No immediate action needed — maintain current support and schedule a routine check-in next quarter.")
    elif score < 7:
        res_cls  = "res-mod";  color = "#f97316"; glow = "#f97316"
        level    = "Moderate Risk"; icon = "⚠️"
        advice   = ("Burnout signals are emerging. A workload review and a manager check-in are recommended within the next 30 days to prevent escalation.")
    else:
        res_cls  = "res-high"; color = "#ef4444"; glow = "#ef4444"
        level    = "High Risk"; icon = "🚨"
        advice   = ("Critical burnout risk detected. Immediate action required — schedule a 1-on-1, reduce workload, and connect the employee with mental health support.")

    st.markdown(f"""
    <div class="result-wrap {res_cls}">
      <div class="res-glow" style="background:{glow};"></div>
      <div class="res-tag">Predicted Burnout Score</div>
      <div class="res-score" style="color:{color};">
        {score:.2f}<span class="res-denom"> / 10</span>
      </div>
      <div class="res-level" style="color:{color};">{icon} &nbsp;{level}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct}%; background:linear-gradient(90deg,{color}99,{color});"></div>
      </div>
      <div class="bar-labels"><span>0 — No Risk</span><span>5 — Moderate</span><span>10 — Extreme</span></div>
    </div>

    <div class="indicators">
      <div class="ind-box">
        <div class="ind-val" style="color:{color};">{score:.1f}</div>
        <div class="ind-lbl">Risk Score</div>
      </div>
      <div class="ind-box">
        <div class="ind-val" style="color:#7a859a;">{auto_ratio}</div>
        <div class="ind-lbl">Automated</div>
      </div>
      <div class="ind-box">
        <div class="ind-val" style="color:#7a859a;">{balance_lbl}</div>
        <div class="ind-lbl">Work-Life</div>
      </div>
    </div>

    <div class="insight">
      <span class="insight-icon">{icon}</span>
      <div class="insight-body"><strong>{level} — Recommended Action:</strong><br>{advice}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <strong>BurnoutIQ</strong><br>
  Random Forest Regressor &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Streamlit<br>
  For internal HR analytics use only — not a substitute for professional assessment.
</div>
""", unsafe_allow_html=True)