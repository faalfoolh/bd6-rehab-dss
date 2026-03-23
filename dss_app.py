"""
BD6 — Stroke Rehabilitation Decision Support System (DSS)
Works on laptop and phone (any browser).
Run with:  streamlit run dss_app.py
"""

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
from scipy import signal
from scipy.stats import kurtosis, skew, entropy as sp_entropy
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SAMPLING_RATE = 100
WINDOW_SIZE   = 200
STEP_SIZE     = 100
BODY_PARTS    = ['Hand', 'Wrist', 'Elbow', 'Shoulder']

SENSORS_MAX_YUSUF        = {'00B44876':'Hand','00B44805':'Wrist','00B44856':'Elbow','00B44877':'Shoulder'}
SENSORS_SARA_ALFAF_RR    = {'00B447F7':'Hand','00B44804':'Wrist','00B4486D':'Elbow','00B44846':'Shoulder'}
SENSORS_SARA_ALFAF_OTHER = {'00B447FD':'Hand','00B447FA':'Wrist','00B447F1':'Elbow','00B44730':'Shoulder'}

TASK_LABELS  = {'reach_retrieve':'Reach & Retrieve','cup_to_lip':'Cup to Lip',
                'arm_swing':'Arm Swing','wrist_rotation':'Wrist Rotation'}
TASK_ICONS   = {'reach_retrieve':'🤚','cup_to_lip':'☕','arm_swing':'💪','wrist_rotation':'🔄'}
TASK_COLORS  = {'reach_retrieve':'#4C9BE8','cup_to_lip':'#E8844C',
                'arm_swing':'#4CE87A','wrist_rotation':'#E84C4C'}

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
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
    if 'max'   in p: return 'Max'
    if 'sara'  in p: return 'Sara'
    if 'alfaf' in p: return 'Alfaf'
    return 'Unknown'

def get_sensor_map(participant, task):
    if participant in ('Max','Yusuf'): return SENSORS_MAX_YUSUF
    if task == 'reach_retrieve':       return SENSORS_SARA_ALFAF_RR
    return SENSORS_SARA_ALFAF_OTHER

def read_file(filepath):
    device_id = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'DeviceId' in line:
                device_id = line.strip().split(':')[-1].strip(); break
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=12, engine='python')
        df.columns = df.columns.str.strip()
        df = df[['Roll','Pitch','Yaw']].dropna().astype(float)
        return device_id, df
    except Exception:
        return device_id, None

def bandpass(data, fs=100, low=0.1, high=12.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def sliding_windows(arr, ws, ss):
    out, start = [], 0
    while start + ws <= len(arr):
        out.append(arr[start:start+ws]); start += ss
    return out

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

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA + TRAIN MODEL (cached — runs once per deployment)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Setting up — this takes about 1 minute on first load...')
def load_and_train():
    np.random.seed(42)

    # Load all txt files
    records = []
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if not fname.endswith('.txt'): continue
            fpath = os.path.join(root, fname)
            task  = infer_task(fpath)
            if not task: continue
            participant = infer_participant(fpath)
            device_id, df = read_file(fpath)
            if df is None or len(df) < WINDOW_SIZE: continue
            sensor_map = get_sensor_map(participant, task)
            body_part  = sensor_map.get(device_id, 'Unknown') if device_id else 'Unknown'
            records.append({'participant':participant,'task':task,'body_part':body_part,
                            'device_id':device_id,'path':fpath,'data':df})

    # Build sessions and windows
    sessions = defaultdict(list)
    for r in records:
        sessions[(r['participant'], r['task'], os.path.dirname(r['path']))].append(r)

    X_wins, y_wins = [], []
    for (participant, task, sdir), recs in sessions.items():
        part_map = {r['body_part']: bandpass(r['data'].values) for r in recs}
        if not all(bp in part_map for bp in BODY_PARTS): continue
        min_len  = min(len(part_map[bp]) for bp in BODY_PARTS)
        combined = np.hstack([part_map[bp][:min_len] for bp in BODY_PARTS])
        for w in sliding_windows(combined, WINDOW_SIZE, STEP_SIZE):
            # Original + 3 augmentations
            for aug in [w,
                        w + np.random.normal(0, 0.5, w.shape),
                        w * np.random.uniform(0.9, 1.1)]:
                X_wins.append(aug); y_wins.append(task)

    X_feat = np.array([extract_features(w) for w in X_wins])
    y      = np.array(y_wins)
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_feat)

    # Fast linear SVM (no SFS needed — works well with all 120 features)
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42))
    clf.fit(X_sc, y_enc)

    # Pre-compute results per patient
    participants = sorted(set(r['participant'] for r in records))
    precomputed  = {}
    for p in participants:
        p_sessions = defaultdict(list)
        for r in records:
            if r['participant'] != p: continue
            p_sessions[(r['task'], os.path.dirname(r['path']))].append(r)

        task_counts = defaultdict(int)
        raw_signals = {}

        for (task, sdir), recs in p_sessions.items():
            part_map = {r['body_part']: r['data'] for r in recs}
            filt_map = {r['body_part']: bandpass(r['data'].values) for r in recs}
            if not all(bp in filt_map for bp in BODY_PARTS): continue
            min_len  = min(len(filt_map[bp]) for bp in BODY_PARTS)
            combined = np.hstack([filt_map[bp][:min_len] for bp in BODY_PARTS])
            wins     = sliding_windows(combined, WINDOW_SIZE, STEP_SIZE)
            if not wins: continue
            X     = np.array([extract_features(w) for w in wins])
            X_sc2 = scaler.transform(X)
            preds = le.inverse_transform(clf.predict(X_sc2))
            for pred in preds:
                task_counts[pred] += 1
            if task not in raw_signals:
                raw_signals[task] = {bp: part_map[bp].iloc[:3000].reset_index(drop=True)
                                     for bp in BODY_PARTS if bp in part_map}

        precomputed[p] = {'task_counts': dict(task_counts), 'raw_signals': raw_signals}

    return precomputed, sorted(participants)

# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────
def compliance_status(task_counts):
    n = sum(1 for t in TASK_LABELS if task_counts.get(t, 0) > 0)
    if n == 4:  return 'Compliant',          '#2ECC71', '🟢'
    if n >= 2:  return 'Partially Compliant','#F1C40F', '🟡'
    return           'Non-Compliant',         '#E74C3C', '🔴'

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='BD6 Rehab DSS', page_icon='🏥',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
    .patient-card { border-radius:12px; padding:1.2rem; margin-bottom:0.8rem;
                    border:1px solid #e0e0e0; box-shadow:0 2px 6px rgba(0,0,0,0.08); }
    h1 { font-size: clamp(1.4rem, 4vw, 2rem); }
    h2 { font-size: clamp(1.1rem, 3vw, 1.5rem); }
    .stButton>button { width:100%; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
precomputed, PARTICIPANTS = load_and_train()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == '🏠 Patient Overview':
    st.title('🏥 Stroke Rehabilitation Dashboard')
    st.markdown('Movement compliance tracking for all patients.')
    st.divider()

    all_counts  = {p: precomputed[p]['task_counts'] for p in PARTICIPANTS}
    n_compliant = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Compliant')
    n_partial   = sum(1 for p in PARTICIPANTS if 'Partially' in compliance_status(all_counts[p])[0])
    n_non       = sum(1 for p in PARTICIPANTS if compliance_status(all_counts[p])[0] == 'Non-Compliant')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Patients', len(PARTICIPANTS))
    c2.metric('🟢 Compliant',   n_compliant)
    c3.metric('🟡 Partial',     n_partial)
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
            <p style="margin:0; color:#666">Total movement windows detected: {total}</p>
        </div>""", unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (task, label) in zip(cols, TASK_LABELS.items()):
            count = tc.get(task, 0)
            col.metric(f"{TASK_ICONS[task]} {label}", f"{count} windows",
                       '✅ Done' if count > 0 else '❌ Not done')
        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PATIENT DETAIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📊 Patient Detail':
    p = selected_patient
    st.title(f'📊 {p} — Movement Detail')
    task_counts = precomputed[p]['task_counts']
    status, color, icon = compliance_status(task_counts)
    total = sum(task_counts.values())

    st.markdown(f"""
    <div style="background:{color}22; border:2px solid {color};
                border-radius:10px; padding:1rem; margin-bottom:1rem;">
        <h2 style="margin:0; color:{color}">{icon} {status}</h2>
        <p style="margin:0">Total movement windows detected: <b>{total}</b></p>
    </div>""", unsafe_allow_html=True)

    st.subheader('Movement Frequency')
    tasks  = [TASK_LABELS[t] for t in TASK_LABELS]
    counts = [task_counts.get(t, 0) for t in TASK_LABELS]
    colors = [TASK_COLORS[t] for t in TASK_LABELS]
    fig = go.Figure(go.Bar(x=tasks, y=counts, marker_color=colors,
                           text=counts, textposition='outside'))
    fig.update_layout(yaxis_title='Windows Detected', plot_bgcolor='white',
                      height=350, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Movement Breakdown')
    cols = st.columns(2)
    for i, (task, label) in enumerate(TASK_LABELS.items()):
        count = task_counts.get(task, 0)
        done  = count > 0
        bg  = '#2ECC7122' if done else '#E74C3C22'
        bdr = '#2ECC71'   if done else '#E74C3C'
        txt = '✅ Performed' if done else '❌ Not Performed'
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {bdr}; border-radius:10px;
                        padding:1rem; margin-bottom:0.8rem; text-align:center;">
                <h3 style="margin:0">{TASK_ICONS[task]} {label}</h3>
                <p style="margin:4px 0; font-size:1.3rem">{txt}</p>
                <p style="margin:0; color:#555">{count} windows</p>
            </div>""", unsafe_allow_html=True)

    if total > 0:
        st.subheader('Movement Distribution')
        labels = [TASK_LABELS[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        vals   = [task_counts[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        clrs   = [TASK_COLORS[t] for t in TASK_LABELS if task_counts.get(t,0)>0]
        fig2 = go.Figure(go.Pie(labels=labels, values=vals,
                                marker_colors=clrs, hole=0.4))
        fig2.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
elif page == '📈 Movement Signals':
    p = selected_patient
    st.title(f'📈 {p} — Raw Movement Signals')
    task_counts = precomputed[p]['task_counts']
    raw_signals = precomputed[p]['raw_signals']

    if not raw_signals:
        st.warning('No signal data found for this patient.')
    else:
        selected_task = st.selectbox('Select Movement', list(raw_signals.keys()),
                                     format_func=lambda t: f"{TASK_ICONS[t]} {TASK_LABELS[t]}")
        selected_bp   = st.selectbox('Select Sensor', BODY_PARTS)
        sig_dict      = raw_signals.get(selected_task, {})

        if selected_bp not in sig_dict:
            st.warning(f'No data for {selected_bp} sensor in this session.')
        else:
            df_sig = sig_dict[selected_bp]
            n_show = min(len(df_sig), 30 * SAMPLING_RATE)
            t      = np.arange(n_show) / SAMPLING_RATE
            colors_rpy = ['#E74C3C','#3498DB','#2ECC71']

            st.subheader(f'Raw Signal — {selected_bp}')
            fig = go.Figure()
            for ax, col in zip(['Roll','Pitch','Yaw'], colors_rpy):
                fig.add_trace(go.Scatter(x=t, y=df_sig[ax].values[:n_show],
                                         name=ax, line=dict(color=col, width=1.2)))
            fig.update_layout(xaxis_title='Time (s)', yaxis_title='Angle (°)',
                              plot_bgcolor='white', height=300,
                              legend=dict(orientation='h', y=1.1),
                              margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader('Filtered Signal (0.1–12 Hz Butterworth)')
            filtered = bandpass(df_sig.values)
            fig2 = go.Figure()
            for i, (ax, col) in enumerate(zip(['Roll','Pitch','Yaw'], colors_rpy)):
                fig2.add_trace(go.Scatter(x=t, y=filtered[:n_show, i],
                                           name=ax, line=dict(color=col, width=1.2)))
            fig2.update_layout(xaxis_title='Time (s)', yaxis_title='Angle (°)',
                               plot_bgcolor='white', height=300,
                               legend=dict(orientation='h', y=1.1),
                               margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

            count   = task_counts.get(selected_task, 0)
            st.success(f"**Detected:** {TASK_ICONS[selected_task]} {TASK_LABELS[selected_task]} — {count} windows classified")
            jerk_avg = np.mean([_jerk(filtered[:, i]) for i in range(3)])
            quality  = 'Smooth 🟢' if jerk_avg < 50 else ('Moderate 🟡' if jerk_avg < 150 else 'Jerky 🔴')
            st.info(f"**Movement quality:** {quality}  (jerk: {jerk_avg:.1f}°/s²)")
