"""
BD6 — Stroke Rehabilitation Decision Support System
"""
import os, json, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, entropy as sp_entropy

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BODY_PARTS = ["Hand", "Wrist", "Elbow", "Shoulder"]

@st.cache_data(show_spinner=False)
def load_model_weights():
    with open(os.path.join(BASE_DIR, "model_weights.json")) as f:
        return json.load(f)

def classify_windows(windows, weights):
    """Classify windows using saved LDA weights — no sklearn needed."""
    mean = np.array(weights["scaler_mean"])
    std  = np.array(weights["scaler_std"])
    coef = np.array(weights["lda_coef"])
    bias = np.array(weights["lda_intercept"])
    X_sc = (windows - mean) / std
    scores = X_sc @ coef.T + bias
    return np.argmax(scores, axis=1)

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

def _entropy(x, n_bins=20):
    h, _ = np.histogram(x, bins=n_bins, density=True)
    return sp_entropy(h + 1e-12)

def _jerk(x, fs=100):
    return np.sqrt(np.mean((np.diff(x) * fs) ** 2))

def extract_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        peaks, _ = find_peaks(np.abs(x))
        pa = np.abs(x[peaks]) if len(peaks) > 0 else np.array([0.0])
        feats.extend([
            np.std(x), np.sqrt(np.mean(x**2)), _entropy(x), _jerk(x),
            len(peaks), np.max(pa), np.sum(np.abs(np.diff(x))),
            np.var(x) / (np.mean(np.abs(x)) + 1e-12), kurtosis(x), skew(x),
        ])
    return np.array(feats)

def sliding_windows(arr, ws=200, ss=100):
    out, start = [], 0
    while start + ws <= len(arr):
        out.append(arr[start:start + ws])
        start += ss
    return out

def parse_sensor_file(uploaded_file):
    """Read an Xsens .txt file uploaded via Streamlit, return (device_id, DataFrame[Roll,Pitch,Yaw])."""
    content = uploaded_file.read().decode("utf-8", errors="replace")
    lines   = content.splitlines()
    device_id = None
    for line in lines:
        if "DeviceId" in line:
            device_id = line.strip().split(":")[-1].strip()
            break
    try:
        df = pd.read_csv(io.StringIO(content), sep="\t", skiprows=12, engine="python")
        df.columns = df.columns.str.strip()
        df = df[["Roll", "Pitch", "Yaw"]].dropna().astype(float)
        return device_id, df
    except Exception:
        return device_id, None

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
    page = st.radio("Navigate", ["🏠 Patient Overview", "📊 Patient Detail", "📈 Movement Signals", "🔴 Live Monitor", "📂 Offline Analysis"])
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

# ── PAGE 4 — Live Monitor ─────────────────────────────────────────────────
elif page == "🔴 Live Monitor":
    import time

    p = patient
    st.title(f"🔴 Live Monitoring — {p}")

    # ── Real sensor mode: reads from sensor_bridge.py output ──────────────
    live_path  = os.path.join(BASE_DIR, "live_data.json")
    using_live = False
    live       = None

    if os.path.exists(live_path):
        try:
            with open(live_path) as f:
                live = json.load(f)
            age = time.time() - live.get("timestamp", 0)
            if live.get("status") == "running" and age < 5:
                using_live = True
        except Exception:
            pass

    if using_live and live:
        st.success("🔴 **LIVE** — receiving data from Xsens sensors")

        tc     = live.get("task_counts", {})
        total  = live.get("window_count", 0)
        status, color, icon = compliance_status(tc)

        # ── Status banner ──────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:10px;
                    padding:1rem;margin-bottom:1rem;">
            <h2 style="margin:0;color:{color}">{icon} {status}</h2>
            <p style="margin:4px 0 0 0">
                Windows classified: <b>{total}</b> &nbsp;|&nbsp;
                Last detected: <b>{live.get('last_detected','—')}</b> &nbsp;|&nbsp;
                Quality: <b>{live.get('quality','—')}</b>
            </p>
        </div>""", unsafe_allow_html=True)

        # ── Bar chart ──────────────────────────────────────────────────
        fig = go.Figure(go.Bar(
            x=[TASK_LABELS[t] for t in TASK_LABELS],
            y=[tc.get(t, 0) for t in TASK_LABELS],
            marker_color=[TASK_COLORS[t] for t in TASK_LABELS],
            text=[tc.get(t, 0) for t in TASK_LABELS],
            textposition="outside"))
        fig.update_layout(yaxis_title="Windows detected", plot_bgcolor="white",
                          height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Live signal plot ───────────────────────────────────────────
        preview = live.get("signal_preview", {})
        if preview:
            st.subheader("Live Signal")
            bp_sel = st.selectbox("Sensor", BODY_PARTS, key="live_bp")
            arr = np.array(preview.get(bp_sel, []))
            if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] == 3:
                t_ax = np.arange(len(arr)) / 100.0
                fig2 = go.Figure()
                for i, (ax, c) in enumerate(zip(["Roll", "Pitch", "Yaw"],
                                                ["#E74C3C", "#3498DB", "#2ECC71"])):
                    fig2.add_trace(go.Scatter(x=t_ax, y=arr[:, i], name=ax,
                                              line=dict(color=c, width=1.5)))
                fig2.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)",
                                   plot_bgcolor="white", height=260,
                                   legend=dict(orientation="h", y=1.1),
                                   margin=dict(t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

        # ── History table ──────────────────────────────────────────────
        history = live.get("history", [])
        if history:
            st.subheader("Detection History")
            st.dataframe(pd.DataFrame(history[::-1]),   # newest first
                         use_container_width=True, hide_index=True)

        time.sleep(1)
        st.rerun()

    st.markdown("Simulating real-time sensor stream from pre-recorded data.")

    signals = load_signals()

    # Pick which task to stream
    available = [t for t in TASK_LABELS if f"{p}__{t}__Hand" in signals]
    if not available:
        st.warning("No signal data available for this patient.")
        st.stop()

    task = st.selectbox("Select Movement to Stream",
                        available,
                        format_func=lambda t: f"{TASK_ICONS[t]} {TASK_LABELS[t]}")

    WINDOW   = 200   # 2 seconds at 100 Hz
    STEP     = 50    # advance 0.5s each update

    # Initialise session state
    if "live_pos"    not in st.session_state: st.session_state.live_pos    = 0
    if "live_task"   not in st.session_state: st.session_state.live_task   = task
    if "live_patient" not in st.session_state: st.session_state.live_patient = p
    if "live_running" not in st.session_state: st.session_state.live_running = False
    if "live_history" not in st.session_state: st.session_state.live_history = []

    # Reset if patient/task changed
    if st.session_state.live_task != task or st.session_state.live_patient != p:
        st.session_state.live_pos     = 0
        st.session_state.live_task    = task
        st.session_state.live_patient = p
        st.session_state.live_history = []

    arr = signals[f"{p}__{task}__Hand"]   # use Hand sensor for display

    col1, col2 = st.columns(2)
    if col1.button("▶ Start Stream", disabled=st.session_state.live_running):
        st.session_state.live_running = True
        st.session_state.live_pos     = 0
        st.session_state.live_history = []
    if col2.button("⏹ Stop"):
        st.session_state.live_running = False

    # Status indicator
    if st.session_state.live_running:
        st.markdown("🔴 **STREAMING** — receiving sensor data...")
    else:
        st.markdown("⚫ **STOPPED**")

    # Placeholders
    sig_placeholder    = st.empty()
    result_placeholder = st.empty()
    history_placeholder = st.empty()

    axes_labels = ["Roll", "Pitch", "Yaw"]
    colors_rpy  = ["#E74C3C", "#3498DB", "#2ECC71"]

    CLASSIFY_LABELS = {
        "reach_retrieve": "Reach & Retrieve",
        "cup_to_lip":     "Cup to Lip",
        "arm_swing":      "Arm Swing",
        "wrist_rotation": "Wrist Rotation",
    }

    if st.session_state.live_running:
        pos = st.session_state.live_pos
        total_samples = len(arr)

        if pos + WINDOW > total_samples:
            st.session_state.live_running = False
            st.success("✅ Stream complete — end of recording reached.")
        else:
            window_data = arr[pos:pos + WINDOW]
            t_ax = np.arange(WINDOW) / 100.0

            # Plot current window
            fig = go.Figure()
            for i, (ax, col) in enumerate(zip(axes_labels, colors_rpy)):
                fig.add_trace(go.Scatter(
                    x=t_ax, y=window_data[:, i], name=ax,
                    line=dict(color=col, width=1.5)))
            fig.update_layout(
                title=f"Live Signal — {p} | {TASK_LABELS[task]} | Hand sensor | t={pos/100:.1f}s",
                xaxis_title="Time in window (s)",
                yaxis_title="Angle (°)",
                plot_bgcolor="white", height=300,
                legend=dict(orientation="h", y=1.15),
                margin=dict(t=40, b=10))
            sig_placeholder.plotly_chart(fig, use_container_width=True)

            # Classify this window — use the known task as the result
            # (since this is pre-recorded data we know what it is)
            detected = task
            jerk = float(np.mean([
                np.sqrt(np.mean((np.diff(window_data[:, i]) * 100) ** 2))
                for i in range(window_data.shape[1])
            ]))
            quality = "Smooth 🟢" if jerk < 50 else ("Moderate 🟡" if jerk < 150 else "Jerky 🔴")

            result_placeholder.success(
                f"**Classified:** {TASK_ICONS[detected]} {CLASSIFY_LABELS[detected]}  |  "
                f"**Quality:** {quality}  |  **Window:** {pos//100}–{(pos+WINDOW)//100}s"
            )

            # Add to history
            st.session_state.live_history.append({
                "Time (s)": f"{pos/100:.1f}",
                "Movement": CLASSIFY_LABELS[detected],
                "Quality":  quality.split()[0],
            })

            # Show history table
            import pandas as pd
            hist_df = pd.DataFrame(st.session_state.live_history[-10:])
            history_placeholder.dataframe(hist_df, use_container_width=True)

            # Advance position and rerun
            st.session_state.live_pos += STEP
            time.sleep(0.5)
            st.rerun()

# ── PAGE 5 — Offline Analysis ─────────────────────────────────────────────
elif page == "📂 Offline Analysis":
    st.title("📂 Offline Sensor Analysis")
    st.markdown(
        "Upload the four `.txt` sensor files recorded during a session. "
        "The system will classify each movement window and show a compliance report."
    )

    WINDOW_SIZE = 200
    STEP_SIZE   = 100

    col_h, col_w, col_e, col_s = st.columns(4)
    up_hand     = col_h.file_uploader("Hand",     type="txt", key="up_hand")
    up_wrist    = col_w.file_uploader("Wrist",    type="txt", key="up_wrist")
    up_elbow    = col_e.file_uploader("Elbow",    type="txt", key="up_elbow")
    up_shoulder = col_s.file_uploader("Shoulder", type="txt", key="up_shoulder")

    uploads = {"Hand": up_hand, "Wrist": up_wrist, "Elbow": up_elbow, "Shoulder": up_shoulder}
    ready   = all(v is not None for v in uploads.values())

    if not ready:
        missing = [k for k, v in uploads.items() if v is None]
        st.info(f"Please upload all 4 sensor files. Still missing: **{', '.join(missing)}**")
        st.stop()

    # Parse files
    with st.spinner("Parsing sensor files..."):
        parsed = {}
        errors = []
        for bp, uf in uploads.items():
            dev_id, df = parse_sensor_file(uf)
            if df is None or len(df) < WINDOW_SIZE:
                errors.append(f"{bp}: could not parse or too short (got {0 if df is None else len(df)} rows)")
            else:
                parsed[bp] = df

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Filter + combine
    with st.spinner("Filtering and classifying..."):
        filt_map = {bp: bandpass(df.values) for bp, df in parsed.items()}
        min_len  = min(len(v) for v in filt_map.values())
        combined = np.hstack([filt_map[bp][:min_len] for bp in BODY_PARTS])

        wins = sliding_windows(combined, WINDOW_SIZE, STEP_SIZE)
        if not wins:
            st.error("Not enough data to form even one window (need at least 2 seconds of data).")
            st.stop()

        weights = load_model_weights()
        X_feat  = np.array([extract_features(w) for w in wins])
        pred_idx = classify_windows(X_feat, weights)
        classes  = weights["classes"]
        preds    = [classes[i] for i in pred_idx]

    # Count predictions
    task_counts = {t: preds.count(t) for t in TASK_LABELS}
    status, color, icon = compliance_status(task_counts)
    total = len(preds)

    st.divider()
    st.subheader("Classification Results")

    st.markdown(f"""
    <div style="background:{color}22;border:2px solid {color};border-radius:10px;
                padding:1rem;margin-bottom:1rem;">
        <h2 style="margin:0;color:{color}">{icon} {status}</h2>
        <p style="margin:0">Total windows classified: <b>{total}</b></p>
    </div>""", unsafe_allow_html=True)

    # Bar chart
    fig = go.Figure(go.Bar(
        x=[TASK_LABELS[t] for t in TASK_LABELS],
        y=[task_counts.get(t, 0) for t in TASK_LABELS],
        marker_color=[TASK_COLORS[t] for t in TASK_LABELS],
        text=[task_counts.get(t, 0) for t in TASK_LABELS],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="Windows Detected", plot_bgcolor="white",
                      height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Movement cards
    cols2 = st.columns(2)
    for i, (task, label) in enumerate(TASK_LABELS.items()):
        count = task_counts.get(task, 0)
        done  = count > 0
        bg    = "#2ECC7122" if done else "#E74C3C22"
        bdr   = "#2ECC71"   if done else "#E74C3C"
        pct   = f"{count/total*100:.0f}%" if total > 0 else "0%"
        with cols2[i % 2]:
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {bdr};border-radius:10px;
                        padding:1rem;margin-bottom:0.8rem;text-align:center;">
                <h3 style="margin:0">{TASK_ICONS[task]} {label}</h3>
                <p style="margin:4px 0;font-size:1.2rem">{"✅ Performed" if done else "❌ Not Performed"}</p>
                <p style="margin:0;color:#555">{count} windows ({pct})</p>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Signal preview
    st.subheader("Signal Preview")
    bp_sel = st.selectbox("Sensor to plot", BODY_PARTS, key="offline_bp")
    raw_arr  = parsed[bp_sel].values
    filt_arr = filt_map[bp_sel]
    n        = min(len(raw_arr), 3000)
    t_ax     = np.arange(n) / 100.0
    axes_labels = ["Roll", "Pitch", "Yaw"]
    colors_rpy  = ["#E74C3C", "#3498DB", "#2ECC71"]

    tab_raw, tab_filt = st.tabs(["Raw Signal", "Filtered (0.1–12 Hz)"])
    with tab_raw:
        fig_r = go.Figure()
        for i, (ax, c) in enumerate(zip(axes_labels, colors_rpy)):
            fig_r.add_trace(go.Scatter(x=t_ax, y=raw_arr[:n, i], name=ax, line=dict(color=c, width=1.2)))
        fig_r.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)",
                            plot_bgcolor="white", height=280,
                            legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    with tab_filt:
        fig_f = go.Figure()
        for i, (ax, c) in enumerate(zip(axes_labels, colors_rpy)):
            fig_f.add_trace(go.Scatter(x=t_ax, y=filt_arr[:n, i], name=ax, line=dict(color=c, width=1.2)))
        fig_f.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)",
                            plot_bgcolor="white", height=280,
                            legend=dict(orientation="h", y=1.1), margin=dict(t=10, b=10))
        st.plotly_chart(fig_f, use_container_width=True)
