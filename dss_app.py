"""
BD6 — Stroke Rehabilitation Decision Support System (DSS)
Works on laptop and phone (any browser).
Run with:  streamlit run dss_app.py
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from scipy import signal
from scipy.stats import kurtosis, skew, entropy as sp_entropy
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SAMPLING_RATE = 100
WINDOW_SIZE   = 200   # 2 seconds
STEP_SIZE     = 100   # 50% overlap

SENSORS_MAX_YUSUF = {
    '00B44876': 'Hand', '00B44805': 'Wrist',
    '00B44856': 'Elbow', '00B44877': 'Shoulder',
}
SENSORS_SARA_ALFAF_RR = {
    '00B447F7': 'Hand', '00B44804': 'Wrist',
    '00B4486D': 'Elbow', '00B44846': 'Shoulder',
}
SENSORS_SARA_ALFAF_OTHER = {
    '00B447FD': 'Hand', '00B447FA': 'Wrist',
    '00B447F1': 'Elbow', '00B44730': 'Shoulder',
}

TASK_LABELS = {
    'reach_retrieve': 'Reach & Retrieve',
    'cup_to_lip':     'Cup to Lip',
    'arm_swing':      'Arm Swing',
    'wrist_rotation': 'Wrist Rotation',
}
TASK_ICONS = {
    'reach_retrieve': '🤚',
    'cup_to_lip':     '☕',
    'arm_swing':      '💪',
    'wrist_rotation': '🔄',
}
TASK_COLORS = {
    'reach_retrieve': '#4C9BE8',
    'cup_to_lip':     '#E8844C',
    'arm_swing':      '#4CE87A',
    'wrist_rotation': '#E84C4C',
}
BODY_PARTS = ['Hand', 'Wrist', 'Elbow', 'Shoulder']

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def infer_task(path):
    p = path.lower()
    if any(k in p for k in ['reach and r','r and r','rr2','rr_new','/rr/','reach and retrieve','/reach/']):
        return 'reach_retrieve'
    if any(k in p for k in ['cup to lip','/cup/','cup\\']):
        return 'cup_to_lip'
    if any(k in p for k in ['wrist rotation','/wrist/']):
        return 'wrist_rotation'
    if any(k in p for k in ['arm swing','horizontal']):
        return 'arm_swing'
    return None

def infer_participant(path):
    p = path.lower()
    if 'yussuf' in p or 'yusuf' in p: return 'Yusuf'
    if 'max'   in p:                  return 'Max'
    if 'sara'  in p:                  return 'Sara'
    if 'alfaf' in p:                  return 'Alfaf'
    return 'Unknown'

def get_sensor_map(participant, task):
    if participant in ('Max','Yusuf'): return SENSORS_MAX_YUSUF
    if task == 'reach_retrieve':       return SENSORS_SARA_ALFAF_RR
    return SENSORS_SARA_ALFAF_OTHER

def read_file(filepath):
    device_id = None
    with open(filepath,'r') as f:
        for line in f:
            if 'DeviceId' in line:
                device_id = line.strip().split(':')[-1].strip()
                break
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=12, engine='python')
        df.columns = df.columns.str.strip()
        df = df[['Roll','Pitch','Yaw']].dropna().astype(float)
        return device_id, df
    except Exception:
        return device_id, None

@st.cache_data(show_spinner=False)
def load_all_data():
    data_path = os.path.join(BASE_DIR, 'bd6_data.pkl')
    if not os.path.exists(data_path):
        st.error("Data file not found. Please run:  python3 train_model.py")
        st.stop()
    with open(data_path, 'rb') as f:
        return pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def bandpass(data, fs=100, low=0.1, high=12.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def sliding_windows(arr, ws, ss):
    out, start = [], 0
    while start + ws <= len(arr):
        out.append(arr[start:start+ws])
        start += ss
    return out

def augment_jitter(w, sigma=0.5):
    return w + np.random.normal(0, sigma, w.shape)

def augment_scale(w):
    return w * np.random.uniform(0.9, 1.1)

def augment_time_warp(w, sigma=0.1):
    n = len(w)
    tt = np.linspace(0, 1, 4)
    wp = np.sort(np.clip(tt + np.random.normal(0, sigma, 4), 0, 1))
    wp[0], wp[-1] = 0.0, 1.0
    cs = CubicSpline(tt, wp)
    t0 = np.linspace(0, 1, n)
    tw = np.clip(cs(t0), 0, 1)
    return np.column_stack([np.interp(t0, tw, w[:,ch]) for ch in range(w.shape[1])])

def _entropy(x, n_bins=20):
    h, _ = np.histogram(x, bins=n_bins, density=True)
    return sp_entropy(h + 1e-12)

def _jerk(x, fs=100):
    return np.sqrt(np.mean((np.diff(x)*fs)**2))

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        peaks, _ = find_peaks(np.abs(x))
        pa = np.abs(x[peaks]) if len(peaks) > 0 else np.array([0.0])
        feats.extend([
            np.std(x), np.sqrt(np.mean(x**2)), _entropy(x), _jerk(x),
            len(peaks), np.max(pa), np.sum(np.abs(np.diff(x))),
            np.var(x)/(np.mean(np.abs(x))+1e-12), kurtosis(x), skew(x),
        ])
    return np.array(feats)

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(BASE_DIR, 'bd6_model.pkl')
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run:  python3 train_model.py")
        st.stop()
    with open(model_path, 'rb') as f:
        m = pickle.load(f)
    return m['svm'], m['scaler'], m['sfs'], m['le']

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFY A PATIENT'S SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
def classify_patient(participant, records, svm, scaler, sfs, le):
    """Return per-task window counts and a dict of raw signal DataFrames."""
    p_records = [r for r in records if r['participant'] == participant]
    sessions  = defaultdict(list)
    for r in p_records:
        key = (r['task'], os.path.dirname(r['path']))
        sessions[key].append(r)

    task_counts = defaultdict(int)
    raw_signals = {}   # task → {body_part: df}

    for (task, sdir), recs in sessions.items():
        part_map = {r['body_part']: r['data'] for r in recs}
        filt_map = {r['body_part']: bandpass(r['data'].values) for r in recs}
        if not all(bp in filt_map for bp in BODY_PARTS):
            continue
        min_len  = min(len(filt_map[bp]) for bp in BODY_PARTS)
        combined = np.hstack([filt_map[bp][:min_len] for bp in BODY_PARTS])
        wins     = sliding_windows(combined, WINDOW_SIZE, STEP_SIZE)
        if not wins: continue

        X = np.array([extract_features(w) for w in wins])
        X_sc  = scaler.transform(X)[:, sfs.get_support()]
        preds = le.inverse_transform(svm.predict(X_sc))

        for p in preds:
            task_counts[p] += 1

        # Store raw signal for the first session per task
        if task not in raw_signals:
            raw_signals[task] = {bp: part_map[bp] for bp in BODY_PARTS if bp in part_map}

    return dict(task_counts), raw_signals

# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def compliance_status(task_counts):
    """
    Green  = all 4 movements detected
    Yellow = 2–3 movements detected
    Red    = 0–1 movements detected
    """
    n = sum(1 for t in TASK_LABELS if task_counts.get(t, 0) > 0)
    if n == 4:    return 'Compliant',         '#2ECC71', '🟢'
    if n >= 2:    return 'Partially Compliant','#F1C40F', '🟡'
    return           'Non-Compliant',          '#E74C3C', '🔴'

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='BD6 Rehab DSS',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS for mobile friendliness ────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem; }
    .patient-card {
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-box {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem;
    }
    h1 { font-size: clamp(1.4rem, 4vw, 2rem); }
    h2 { font-size: clamp(1.1rem, 3vw, 1.5rem); }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data & train model ────────────────────────────────────────────────
with st.spinner('Loading data and model...'):
    records = load_all_data()
    svm, scaler, sfs, le = load_model()

PARTICIPANTS = sorted(set(r['participant'] for r in records))

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.image('https://img.icons8.com/color/96/physical-therapy.png', width=60)
    st.title('Rehab DSS')
    st.markdown('**Stroke Rehabilitation**\nDecision Support System')
    st.divider()

    page = st.radio('Navigate', ['🏠 Patient Overview', '📊 Patient Detail', '📈 Movement Signals'])
    st.divider()

    if page != '🏠 Patient Overview':
        selected_patient = st.selectbox('Select Patient', PARTICIPANTS)
    else:
        selected_patient = None

    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PATIENT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == '🏠 Patient Overview':
    st.title('🏥 Stroke Rehabilitation Dashboard')
    st.markdown('Real-time movement compliance tracking for all patients.')
    st.divider()

    # Summary row
    all_counts = {}
    for p in PARTICIPANTS:
        tc, _ = classify_patient(p, records, svm, scaler, sfs, le)
        all_counts[p] = tc

    n_compliant = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Compliant')
    n_partial   = sum(1 for p in PARTICIPANTS if 'Partially' in compliance_status(all_counts[p])[0])
    n_non       = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Non-Compliant')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Patients',       len(PARTICIPANTS))
    c2.metric('🟢 Compliant',         n_compliant)
    c3.metric('🟡 Partial',           n_partial)
    c4.metric('🔴 Non-Compliant',     n_non)
    st.divider()

    # Patient cards
    st.subheader('Patient Status')
    for p in PARTICIPANTS:
        tc = all_counts[p]
        status, color, icon = compliance_status(tc)
        total_windows = sum(tc.values())

        with st.container():
            st.markdown(f"""
            <div class="patient-card" style="border-left: 5px solid {color};">
                <h3 style="margin:0">{icon} {p}</h3>
                <p style="color:{color}; font-weight:bold; margin:4px 0">{status}</p>
                <p style="margin:0; color:#666">Total movement windows detected: {total_windows}</p>
            </div>
            """, unsafe_allow_html=True)

        cols = st.columns(4)
        for col, (task, label) in zip(cols, TASK_LABELS.items()):
            count = tc.get(task, 0)
            detected = '✅' if count > 0 else '❌'
            col.metric(f"{TASK_ICONS[task]} {label}", f"{count} windows", detected)
        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PATIENT DETAIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📊 Patient Detail':
    p = selected_patient
    st.title(f'📊 {p} — Movement Detail')

    with st.spinner(f'Classifying {p}\'s movements...'):
        task_counts, raw_signals = classify_patient(p, records, svm, scaler, sfs, le)

    status, color, icon = compliance_status(task_counts)
    total = sum(task_counts.values())

    # Status banner
    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color};
                border-radius:10px; padding:1rem; margin-bottom:1rem;">
        <h2 style="margin:0; color:{color}">{icon} {status}</h2>
        <p style="margin:0">Total movement windows detected today: <b>{total}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Movement frequency bar chart
    st.subheader('Movement Frequency')
    tasks  = [TASK_LABELS[t] for t in TASK_LABELS]
    counts = [task_counts.get(t, 0) for t in TASK_LABELS]
    colors = [TASK_COLORS[t] for t in TASK_LABELS]

    fig = go.Figure(go.Bar(
        x=tasks, y=counts,
        marker_color=colors,
        text=counts, textposition='outside',
    ))
    fig.update_layout(
        yaxis_title='Windows Detected',
        plot_bgcolor='white',
        height=350,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-movement compliance cards
    st.subheader('Movement Breakdown')
    cols = st.columns(2)
    for i, (task, label) in enumerate(TASK_LABELS.items()):
        count = task_counts.get(task, 0)
        done  = count > 0
        bg    = '#2ECC7122' if done else '#E74C3C22'
        bdr   = '#2ECC71'   if done else '#E74C3C'
        txt   = '✅ Performed' if done else '❌ Not Performed'
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {bdr};
                        border-radius:10px; padding:1rem; margin-bottom:0.8rem; text-align:center;">
                <h3 style="margin:0">{TASK_ICONS[task]} {label}</h3>
                <p style="margin:4px 0; font-size:1.3rem">{txt}</p>
                <p style="margin:0; color:#555">{count} windows</p>
            </div>
            """, unsafe_allow_html=True)

    # Pie chart
    if total > 0:
        st.subheader('Movement Distribution')
        labels = [TASK_LABELS[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        vals   = [task_counts[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        clrs   = [TASK_COLORS[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        fig2 = go.Figure(go.Pie(labels=labels, values=vals, marker_colors=clrs,
                                hole=0.4))
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MOVEMENT SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📈 Movement Signals':
    p = selected_patient
    st.title(f'📈 {p} — Raw Movement Signals')

    with st.spinner('Loading signals...'):
        task_counts, raw_signals = classify_patient(p, records, svm, scaler, sfs, le)

    if not raw_signals:
        st.warning('No signal data found for this patient.')
    else:
        selected_task = st.selectbox(
            'Select Movement',
            options=list(raw_signals.keys()),
            format_func=lambda t: f"{TASK_ICONS[t]} {TASK_LABELS[t]}"
        )
        selected_bp = st.selectbox('Select Sensor Location', BODY_PARTS)

        sig_dict = raw_signals.get(selected_task, {})
        if selected_bp not in sig_dict:
            st.warning(f'No data for {selected_bp} in this session.')
        else:
            df_sig = sig_dict[selected_bp]
            # Show only first 30 seconds for performance
            n_show = min(len(df_sig), 30 * SAMPLING_RATE)
            t = np.arange(n_show) / SAMPLING_RATE

            # Raw signal
            st.subheader(f'Raw Signal — {selected_bp} sensor')
            fig = go.Figure()
            colors_rpy = ['#E74C3C','#3498DB','#2ECC71']
            for ax, col in zip(['Roll','Pitch','Yaw'], colors_rpy):
                fig.add_trace(go.Scatter(x=t, y=df_sig[ax].values[:n_show],
                                         name=ax, line=dict(color=col, width=1.2)))
            fig.update_layout(xaxis_title='Time (s)', yaxis_title='Angle (°)',
                              plot_bgcolor='white', height=320,
                              legend=dict(orientation='h', y=1.1),
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Filtered signal
            st.subheader('Filtered Signal (0.1–12 Hz Butterworth)')
            filtered = bandpass(df_sig.values)
            fig2 = go.Figure()
            for i, (ax, col) in enumerate(zip(['Roll','Pitch','Yaw'], colors_rpy)):
                fig2.add_trace(go.Scatter(x=t, y=filtered[:n_show, i],
                                           name=ax, line=dict(color=col, width=1.2)))
            fig2.update_layout(xaxis_title='Time (s)', yaxis_title='Angle (°)',
                               plot_bgcolor='white', height=320,
                               legend=dict(orientation='h', y=1.1),
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            # Classification result for this task
            st.subheader('Classification Result')
            pred_label = TASK_LABELS.get(selected_task, selected_task)
            count      = task_counts.get(selected_task, 0)
            st.success(f"**Detected movement:** {TASK_ICONS[selected_task]} {pred_label} — {count} windows classified")

            # Movement quality indicator (smoothness via jerk)
            jerk_vals = [_jerk(filtered[:, i]) for i in range(3)]
            avg_jerk  = np.mean(jerk_vals)
            quality   = 'Smooth 🟢' if avg_jerk < 50 else ('Moderate 🟡' if avg_jerk < 150 else 'Jerky 🔴')
            st.info(f"**Movement quality:** {quality}  (jerk metric: {avg_jerk:.1f}°/s²)")
