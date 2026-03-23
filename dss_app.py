"""
BD6 — Stroke Rehabilitation Decision Support System
"""
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import signal

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BODY_PARTS = ["Hand", "Wrist", "Elbow", "Shoulder"]

@st.cache_data(show_spinner=False)
def load_signals():
    path = os.path.join(BASE_DIR, "signals.npz")
    if not os.path.exists(path):
        return {}
    return dict(np.load(path, allow_pickle=False))

def bandpass(data, fs=100, low=0.1, high=12.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, data, axis=0)

# ── Pre-computed classification results (hardcoded — instant load) ─────────
RESULTS = {
    "Alfaf": {"cup_to_lip": 274, "wrist_rotation": 301, "reach_retrieve": 273, "arm_swing": 349},
    "Max":   {"cup_to_lip": 325, "arm_swing": 233,      "wrist_rotation": 177, "reach_retrieve": 435},
    "Sara":  {"reach_retrieve": 272, "cup_to_lip": 217, "wrist_rotation": 283, "arm_swing": 284},
    "Yusuf": {"cup_to_lip": 362, "arm_swing": 246,      "wrist_rotation": 302, "reach_retrieve": 438},
}

TASK_LABELS = {
    "reach_retrieve": "Reach & Retrieve",
    "cup_to_lip":     "Cup to Lip",
    "arm_swing":      "Arm Swing",
    "wrist_rotation": "Wrist Rotation",
}
TASK_ICONS = {
    "reach_retrieve": "🤚",
    "cup_to_lip":     "☕",
    "arm_swing":      "💪",
    "wrist_rotation": "🔄",
}
TASK_COLORS = {
    "reach_retrieve": "#4C9BE8",
    "cup_to_lip":     "#E8844C",
    "arm_swing":      "#4CE87A",
    "wrist_rotation": "#E84C4C",
}
PARTICIPANTS = sorted(RESULTS.keys())

def compliance_status(tc):
    n = sum(1 for t in TASK_LABELS if tc.get(t, 0) > 0)
    if n == 4:  return "Compliant",          "#2ECC71", "🟢"
    if n >= 2:  return "Partially Compliant","#F1C40F", "🟡"
    return           "Non-Compliant",         "#E74C3C", "🔴"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="BD6 Rehab DSS", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .card { border-radius:12px; padding:1.2rem; margin-bottom:0.8rem;
          border:1px solid #e0e0e0; box-shadow:0 2px 6px rgba(0,0,0,0.08); }
  h1 { font-size:clamp(1.4rem,4vw,2rem); }
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Rehab DSS")
    st.markdown("**Stroke Rehabilitation**\nDecision Support System")
    st.divider()
    page = st.radio("Navigate", ["🏠 Patient Overview", "📊 Patient Detail", "📈 Movement Signals"])
    if page != "🏠 Patient Overview":
        patient = st.selectbox("Select Patient", PARTICIPANTS)
    else:
        patient = None
    st.divider()

# ── PAGE 1 — Overview ─────────────────────────────────────────────────────
if page == "🏠 Patient Overview":
    st.title("🏥 Stroke Rehabilitation Dashboard")
    st.markdown("Movement compliance tracking for all patients.")
    st.divider()

    n_compliant = sum(1 for p in PARTICIPANTS if compliance_status(RESULTS[p])[0] == "Compliant")
    n_partial   = sum(1 for p in PARTICIPANTS if "Partially" in compliance_status(RESULTS[p])[0])
    n_non       = sum(1 for p in PARTICIPANTS if compliance_status(RESULTS[p])[0] == "Non-Compliant")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",    len(PARTICIPANTS))
    c2.metric("🟢 Compliant",      n_compliant)
    c3.metric("🟡 Partial",        n_partial)
    c4.metric("🔴 Non-Compliant",  n_non)
    st.divider()

    for p in PARTICIPANTS:
        tc = RESULTS[p]
        status, color, icon = compliance_status(tc)
        total = sum(tc.values())
        st.markdown(f"""
        <div class="card" style="border-left:5px solid {color};">
            <h3 style="margin:0">{icon} {p}</h3>
            <p style="color:{color};font-weight:bold;margin:4px 0">{status}</p>
            <p style="margin:0;color:#666">Total movement windows detected: {total}</p>
        </div>""", unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (task, label) in zip(cols, TASK_LABELS.items()):
            count = tc.get(task, 0)
            col.metric(f"{TASK_ICONS[task]} {label}", f"{count} windows",
                       "✅ Done" if count > 0 else "❌ Not done")
        st.divider()

# ── PAGE 2 — Patient Detail ───────────────────────────────────────────────
elif page == "📊 Patient Detail":
    p  = patient
    tc = RESULTS[p]
    st.title(f"📊 {p} — Movement Detail")
    status, color, icon = compliance_status(tc)
    total = sum(tc.values())

    st.markdown(f"""
    <div style="background:{color}22;border:2px solid {color};border-radius:10px;
                padding:1rem;margin-bottom:1rem;">
        <h2 style="margin:0;color:{color}">{icon} {status}</h2>
        <p style="margin:0">Total windows detected: <b>{total}</b></p>
    </div>""", unsafe_allow_html=True)

    st.subheader("Movement Frequency")
    fig = go.Figure(go.Bar(
        x=[TASK_LABELS[t] for t in TASK_LABELS],
        y=[tc.get(t, 0) for t in TASK_LABELS],
        marker_color=[TASK_COLORS[t] for t in TASK_LABELS],
        text=[tc.get(t, 0) for t in TASK_LABELS],
        textposition="outside"))
    fig.update_layout(yaxis_title="Windows Detected", plot_bgcolor="white",
                      height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Movement Breakdown")
    cols = st.columns(2)
    for i, (task, label) in enumerate(TASK_LABELS.items()):
        count = tc.get(task, 0)
        done  = count > 0
        bg  = "#2ECC7122" if done else "#E74C3C22"
        bdr = "#2ECC71"   if done else "#E74C3C"
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {bdr};border-radius:10px;
                        padding:1rem;margin-bottom:0.8rem;text-align:center;">
                <h3 style="margin:0">{TASK_ICONS[task]} {label}</h3>
                <p style="margin:4px 0;font-size:1.3rem">{"✅ Performed" if done else "❌ Not Performed"}</p>
                <p style="margin:0;color:#555">{count} windows</p>
            </div>""", unsafe_allow_html=True)

    if total > 0:
        st.subheader("Movement Distribution")
        labels = [TASK_LABELS[t] for t in TASK_LABELS if tc.get(t, 0) > 0]
        vals   = [tc[t] for t in TASK_LABELS if tc.get(t, 0) > 0]
        clrs   = [TASK_COLORS[t] for t in TASK_LABELS if tc.get(t, 0) > 0]
        fig2   = go.Figure(go.Pie(labels=labels, values=vals,
                                   marker_colors=clrs, hole=0.4))
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 3 — Movement Signals ─────────────────────────────────────────────
elif page == "📈 Movement Signals":
    p = patient
    st.title(f"📈 {p} — Raw Movement Signals")
    signals = load_signals()

    available = [t for t in TASK_LABELS if f"{p}__{t}__Hand" in signals]
    if not available:
        st.warning("No signal data available for this patient.")
    else:
        task = st.selectbox("Select Movement", available,
                            format_func=lambda t: f"{TASK_ICONS[t]} {TASK_LABELS[t]}")
        bp   = st.selectbox("Select Sensor", BODY_PARTS)
        key  = f"{p}__{task}__{bp}"

        if key not in signals:
            st.warning(f"No data for {bp} sensor in this session.")
        else:
            arr  = signals[key]
            n    = min(len(arr), 3000)
            t_ax = np.arange(n) / 100.0
            axes_labels = ["Roll", "Pitch", "Yaw"]
            colors_rpy  = ["#E74C3C", "#3498DB", "#2ECC71"]

            st.subheader(f"Raw Signal — {bp}")
            fig = go.Figure()
            for i, (ax, col) in enumerate(zip(axes_labels, colors_rpy)):
                fig.add_trace(go.Scatter(x=t_ax, y=arr[:n, i], name=ax,
                                          line=dict(color=col, width=1.2)))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)",
                              plot_bgcolor="white", height=300,
                              legend=dict(orientation="h", y=1.1),
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Filtered Signal (0.1–12 Hz Butterworth)")
            filtered = bandpass(arr[:n])
            fig2 = go.Figure()
            for i, (ax, col) in enumerate(zip(axes_labels, colors_rpy)):
                fig2.add_trace(go.Scatter(x=t_ax, y=filtered[:, i], name=ax,
                                           line=dict(color=col, width=1.2)))
            fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)",
                               plot_bgcolor="white", height=300,
                               legend=dict(orientation="h", y=1.1),
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            count = RESULTS[p].get(task, 0)
            st.success(f"**Detected:** {TASK_ICONS[task]} {TASK_LABELS[task]} — {count} windows classified")
            jerk = float(np.mean([np.sqrt(np.mean((np.diff(filtered[:, i]) * 100) ** 2)) for i in range(3)]))
            quality = "Smooth 🟢" if jerk < 50 else ("Moderate 🟡" if jerk < 150 else "Jerky 🔴")
            st.info(f"**Movement quality:** {quality}  (jerk: {jerk:.1f}°/s²)")
