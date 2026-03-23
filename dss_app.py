"""
BD6 — Stroke Rehabilitation Decision Support System (DSS)
All ML is pre-computed. This app only loads and displays results.
"""

import os, json, warnings
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import signal

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BODY_PARTS = ['Hand', 'Wrist', 'Elbow', 'Shoulder']

TASK_LABELS = {'reach_retrieve':'Reach & Retrieve','cup_to_lip':'Cup to Lip',
               'arm_swing':'Arm Swing','wrist_rotation':'Wrist Rotation'}
TASK_ICONS  = {'reach_retrieve':'🤚','cup_to_lip':'☕','arm_swing':'💪','wrist_rotation':'🔄'}
TASK_COLORS = {'reach_retrieve':'#4C9BE8','cup_to_lip':'#E8844C',
               'arm_swing':'#4CE87A','wrist_rotation':'#E84C4C'}

# ── Load pre-computed data (instant) ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    with open(os.path.join(BASE_DIR, 'precomputed.json')) as f:
        precomputed = json.load(f)
    signals = np.load(os.path.join(BASE_DIR, 'signals.npz'), allow_pickle=False)
    return precomputed, dict(signals)

def bandpass(data, fs=100, low=0.1, high=12.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def compliance_status(task_counts):
    n = sum(1 for t in TASK_LABELS if task_counts.get(t, 0) > 0)
    if n == 4:  return 'Compliant',          '#2ECC71', '🟢'
    if n >= 2:  return 'Partially Compliant','#F1C40F', '🟡'
    return           'Non-Compliant',         '#E74C3C', '🔴'

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title='BD6 Rehab DSS', page_icon='🏥',
                   layout='wide', initial_sidebar_state='expanded')
st.markdown("""
<style>
    .patient-card { border-radius:12px; padding:1.2rem; margin-bottom:0.8rem;
                    border:1px solid #e0e0e0; box-shadow:0 2px 6px rgba(0,0,0,0.08); }
    h1 { font-size: clamp(1.4rem,4vw,2rem); }
    h2 { font-size: clamp(1.1rem,3vw,1.5rem); }
</style>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────
precomputed, signals = load_data()
PARTICIPANTS = sorted(precomputed.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏥 Rehab DSS')
    st.markdown('**Stroke Rehabilitation**\nDecision Support System')
    st.divider()
    page = st.radio('Navigate', ['🏠 Patient Overview','📊 Patient Detail','📈 Movement Signals'])
    if page != '🏠 Patient Overview':
        selected_patient = st.selectbox('Select Patient', PARTICIPANTS)
    else:
        selected_patient = None
    st.divider()

# ── PAGE 1 — Overview ─────────────────────────────────────────────────────
if page == '🏠 Patient Overview':
    st.title('🏥 Stroke Rehabilitation Dashboard')
    st.markdown('Movement compliance tracking for all patients.')
    st.divider()

    all_counts  = {p: precomputed[p] for p in PARTICIPANTS}
    n_compliant = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Compliant')
    n_partial   = sum(1 for p in PARTICIPANTS if 'Partially' in compliance_status(all_counts[p])[0])
    n_non       = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Non-Compliant')

    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Total Patients',   len(PARTICIPANTS))
    c2.metric('🟢 Compliant',     n_compliant)
    c3.metric('🟡 Partial',       n_partial)
    c4.metric('🔴 Non-Compliant', n_non)
    st.divider()

    st.subheader('Patient Status')
    for p in PARTICIPANTS:
        tc = all_counts[p]
        status, color, icon = compliance_status(tc)
        total = sum(tc.values())
        st.markdown(f"""
        <div class="patient-card" style="border-left:5px solid {color};">
            <h3 style="margin:0">{icon} {p}</h3>
            <p style="color:{color}; font-weight:bold; margin:4px 0">{status}</p>
            <p style="margin:0;color:#666">Total movement windows detected: {total}</p>
        </div>""", unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (task, label) in zip(cols, TASK_LABELS.items()):
            count = tc.get(task, 0)
            col.metric(f"{TASK_ICONS[task]} {label}", f"{count} windows",
                       '✅ Done' if count > 0 else '❌ Not done')
        st.divider()

# ── PAGE 2 — Patient Detail ───────────────────────────────────────────────
elif page == '📊 Patient Detail':
    p = selected_patient
    st.title(f'📊 {p} — Movement Detail')
    tc = precomputed[p]
    status, color, icon = compliance_status(tc)
    total = sum(tc.values())

    st.markdown(f"""
    <div style="background:{color}22;border:2px solid {color};border-radius:10px;
                padding:1rem;margin-bottom:1rem;">
        <h2 style="margin:0;color:{color}">{icon} {status}</h2>
        <p style="margin:0">Total windows detected: <b>{total}</b></p>
    </div>""", unsafe_allow_html=True)

    st.subheader('Movement Frequency')
    fig = go.Figure(go.Bar(
        x=[TASK_LABELS[t] for t in TASK_LABELS],
        y=[tc.get(t,0) for t in TASK_LABELS],
        marker_color=[TASK_COLORS[t] for t in TASK_LABELS],
        text=[tc.get(t,0) for t in TASK_LABELS], textposition='outside'))
    fig.update_layout(yaxis_title='Windows Detected', plot_bgcolor='white',
                      height=350, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Movement Breakdown')
    cols = st.columns(2)
    for i, (task, label) in enumerate(TASK_LABELS.items()):
        count = tc.get(task, 0)
        done  = count > 0
        with cols[i % 2]:
            bg  = '#2ECC7122' if done else '#E74C3C22'
            bdr = '#2ECC71'   if done else '#E74C3C'
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {bdr};border-radius:10px;
                        padding:1rem;margin-bottom:0.8rem;text-align:center;">
                <h3 style="margin:0">{TASK_ICONS[task]} {label}</h3>
                <p style="margin:4px 0;font-size:1.3rem">{'✅ Performed' if done else '❌ Not Performed'}</p>
                <p style="margin:0;color:#555">{count} windows</p>
            </div>""", unsafe_allow_html=True)

    if total > 0:
        st.subheader('Movement Distribution')
        labels = [TASK_LABELS[t] for t in TASK_LABELS if tc.get(t,0)>0]
        vals   = [tc[t] for t in TASK_LABELS if tc.get(t,0)>0]
        clrs   = [TASK_COLORS[t] for t in TASK_LABELS if tc.get(t,0)>0]
        fig2   = go.Figure(go.Pie(labels=labels,values=vals,marker_colors=clrs,hole=0.4))
        fig2.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 3 — Signals ──────────────────────────────────────────────────────
elif page == '📈 Movement Signals':
    p = selected_patient
    st.title(f'📈 {p} — Raw Movement Signals')
    tc = precomputed[p]

    available_tasks = [t for t in TASK_LABELS if f"{p}__{t}__Hand" in signals]
    if not available_tasks:
        st.warning('No signal data found for this patient.')
    else:
        selected_task = st.selectbox('Select Movement', available_tasks,
                                     format_func=lambda t: f"{TASK_ICONS[t]} {TASK_LABELS[t]}")
        selected_bp   = st.selectbox('Select Sensor', BODY_PARTS)
        sig_key       = f"{p}__{selected_task}__{selected_bp}"

        if sig_key not in signals:
            st.warning(f'No data for {selected_bp} sensor.')
        else:
            arr  = signals[sig_key]          # (n, 3) numpy array
            n    = min(len(arr), 3000)
            t    = np.arange(n) / 100
            axes_labels = ['Roll','Pitch','Yaw']
            colors_rpy  = ['#E74C3C','#3498DB','#2ECC71']

            st.subheader(f'Raw Signal — {selected_bp}')
            fig = go.Figure()
            for i,(ax,col) in enumerate(zip(axes_labels,colors_rpy)):
                fig.add_trace(go.Scatter(x=t,y=arr[:n,i],name=ax,
                                         line=dict(color=col,width=1.2)))
            fig.update_layout(xaxis_title='Time (s)',yaxis_title='Angle (°)',
                              plot_bgcolor='white',height=300,
                              legend=dict(orientation='h',y=1.1),margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader('Filtered Signal (0.1–12 Hz Butterworth)')
            filtered = bandpass(arr[:n])
            fig2 = go.Figure()
            for i,(ax,col) in enumerate(zip(axes_labels,colors_rpy)):
                fig2.add_trace(go.Scatter(x=t,y=filtered[:,i],name=ax,
                                           line=dict(color=col,width=1.2)))
            fig2.update_layout(xaxis_title='Time (s)',yaxis_title='Angle (°)',
                               plot_bgcolor='white',height=300,
                               legend=dict(orientation='h',y=1.1),margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

            count = tc.get(selected_task, 0)
            st.success(f"**Detected:** {TASK_ICONS[selected_task]} {TASK_LABELS[selected_task]} — {count} windows")
            jerk = np.mean([np.sqrt(np.mean((np.diff(filtered[:,i])*100)**2)) for i in range(3)])
            quality = 'Smooth 🟢' if jerk < 50 else ('Moderate 🟡' if jerk < 150 else 'Jerky 🔴')
            st.info(f"**Movement quality:** {quality}  (jerk: {jerk:.1f}°/s²)")
