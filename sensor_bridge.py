"""
sensor_bridge.py — Live Xsens MTW2 sensor reader + classifier
Run this on the Windows laptop BEFORE opening the DSS in the browser:
    python sensor_bridge.py

It reads Roll/Pitch/Yaw from the 4 MTW2 sensors via the Awinda Station,
classifies movement windows in real-time using the saved LDA weights,
and writes results to live_data.json so the Streamlit app can read them.
"""

import os, sys, json, time, collections, threading
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, entropy as sp_entropy

# ── Xsens SDK ────────────────────────────────────────────────────────────────
# The xsensdeviceapi files must be in the same folder as this script.
# Copy them from:
#   C:\Program Files\Xsens\MT Software Suite X.X\MT SDK\Examples\xda_python\
try:
    import xsensdeviceapi as xda
except ImportError:
    print("ERROR: xsensdeviceapi not found.")
    print("Copy the xda_python folder contents into this project folder.")
    sys.exit(1)

# ── Sensor ID → Body part mapping (auto-detected from connected sensor IDs) ──
# All known sensor sets — participant and task are identified automatically
ALL_SENSOR_MAPS = {
    "Max / Yusuf": {
        "00B44876": "Hand", "00B44805": "Wrist",
        "00B44856": "Elbow", "00B44877": "Shoulder",
    },
    "Sara / Alfaf (Reach & Retrieve)": {
        "00B447F7": "Hand", "00B44804": "Wrist",
        "00B4486D": "Elbow", "00B44846": "Shoulder",
    },
    "Sara / Alfaf (Other movements)": {
        "00B447FD": "Hand", "00B447FA": "Wrist",
        "00B447F1": "Elbow", "00B44730": "Shoulder",
    },
}

def auto_detect_sensor_map(connected_ids):
    """Match connected sensor IDs to the correct participant sensor set."""
    connected_set = set(connected_ids)
    best_map  = None
    best_name = "Unknown"
    best_overlap = 0
    for name, smap in ALL_SENSOR_MAPS.items():
        overlap = len(connected_set & set(smap.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_map     = smap
            best_name    = name
    return best_name, best_map

BODY_PARTS  = ["Hand", "Wrist", "Elbow", "Shoulder"]
WINDOW_SIZE = 200   # 2 seconds at 100 Hz
STEP_SIZE   = 100   # 1 second step
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_FILE    = os.path.join(BASE_DIR, "live_data.json")

# ── Load LDA weights ──────────────────────────────────────────────────────────
weights_path = os.path.join(BASE_DIR, "model_weights.json")
if not os.path.exists(weights_path):
    print("ERROR: model_weights.json not found. Run train_model.py first.")
    sys.exit(1)

with open(weights_path) as f:
    weights = json.load(f)

CLASSES     = weights["classes"]
SC_MEAN     = np.array(weights["scaler_mean"])
SC_STD      = np.array(weights["scaler_std"])
LDA_COEF    = np.array(weights["lda_coef"])
LDA_BIAS    = np.array(weights["lda_intercept"])

TASK_LABELS = {
    "reach_retrieve": "Reach & Retrieve",
    "cup_to_lip":     "Cup to Lip",
    "arm_swing":      "Arm Swing",
    "wrist_rotation": "Wrist Rotation",
}

# ── Signal processing helpers ─────────────────────────────────────────────────
def bandpass(data, fs=100, low=0.1, high=12.0, order=3):
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=0)

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

MOTION_THRESHOLD   = 3.0   # degrees std — below this = sensor is still
CONFIDENCE_THRESHOLD = 0.55  # softmax probability — below this = uncertain

def classify_window(window):
    """Returns (label, confidence) or (None, 0) if not moving / uncertain."""
    # Motion gate — skip if sensor is barely moving
    motion = float(np.mean([np.std(window[:, i]) for i in range(window.shape[1])]))
    if motion < MOTION_THRESHOLD:
        return None, 0.0   # idle

    feat   = extract_features(window).reshape(1, -1)
    X_sc   = (feat - SC_MEAN) / SC_STD
    scores = (X_sc @ LDA_COEF.T + LDA_BIAS)[0]

    # Softmax confidence
    exp_s  = np.exp(scores - scores.max())
    probs  = exp_s / exp_s.sum()
    best   = int(np.argmax(probs))
    conf   = float(probs[best])

    if conf < CONFIDENCE_THRESHOLD:
        return None, conf  # uncertain

    return CLASSES[best], conf

# ── Rolling buffers — one per body part ──────────────────────────────────────
BUFFERS       = {bp: collections.deque(maxlen=WINDOW_SIZE * 4) for bp in BODY_PARTS}
sample_counts = {bp: 0 for bp in BODY_PARTS}

task_counts      = {t: 0 for t in TASK_LABELS}
window_count     = 0
last_detection   = "Waiting..."
last_quality     = "—"
history          = []
participant_name = "Detecting..."

def compute_quality(filtered_window):
    """Compute jerk-based quality label from a filtered window."""
    jerk_val = float(np.mean([
        _jerk(filtered_window[:, i]) for i in range(filtered_window.shape[1])
    ]))
    if jerk_val < 50:   return "Smooth 🟢",   jerk_val
    if jerk_val < 150:  return "Moderate 🟡", jerk_val
    return               "Jerky 🔴",          jerk_val

def write_live_data(status="running"):
    data = {
        "status":        status,
        "participant":   participant_name,
        "task_counts":   task_counts,
        "window_count":  window_count,
        "last_detected": last_detection,
        "quality":       last_quality,
        "timestamp":     time.time(),
        "history":       history[-20:],   # last 20 detections
        "signal_preview": {
            bp: [list(row) for row in list(BUFFERS[bp])[-200:]]
            for bp in BODY_PARTS if len(BUFFERS[bp]) > 0
        },
    }
    with open(OUT_FILE, "w") as f:
        json.dump(data, f)

# ── Xsens callback ────────────────────────────────────────────────────────────
class XsCallback(xda.XsCallback):
    def __init__(self, sensor_map):
        super().__init__()
        self.sensor_map      = sensor_map
        self.connected_mtws  = set()
        self._lock           = threading.Lock()

    def onConnectivityChanged(self, dev, new_state):
        dev_id = str(dev.deviceId())
        print(f"\n  [DEBUG] Connectivity: {dev_id} state={new_state}")
        with self._lock:
            # Any non-zero state means the device is present/connected
            if new_state != 0:
                self.connected_mtws.add(dev_id)
                bp = self.sensor_map.get(dev_id, "Unknown")
                print(f"  + Sensor present: {dev_id} ({bp})")
            else:
                self.connected_mtws.discard(dev_id)
                print(f"  - Sensor disconnected: {dev_id}")

    def onLiveDataAvailable(self, dev, packet):
        if not packet.containsOrientation():
            return
        dev_id = str(dev.deviceId())
        body_part = self.sensor_map.get(dev_id)
        if body_part is None:
            return

        euler = packet.orientationEuler()
        row   = [euler.roll(), euler.pitch(), euler.yaw()]
        BUFFERS[body_part].append(row)
        sample_counts[body_part] += 1

        # Classify when all buffers have enough data
        global window_count, last_detection, last_quality, participant_name
        if all(len(BUFFERS[bp]) >= WINDOW_SIZE for bp in BODY_PARTS):
            if sample_counts["Hand"] % STEP_SIZE == 0:
                try:
                    win = np.hstack([
                        np.array(list(BUFFERS[bp]))[-WINDOW_SIZE:]
                        for bp in BODY_PARTS
                    ])
                    filt              = bandpass(win)
                    label, conf       = classify_window(filt)
                    quality_str, jerk_val = compute_quality(filt)

                    if label is None:
                        # Still or uncertain — don't count, but update display
                        motion = float(np.mean([np.std(filt[:, i]) for i in range(filt.shape[1])]))
                        idle_msg = "Idle (no movement)" if motion < MOTION_THRESHOLD else f"Uncertain ({conf:.0%})"
                        last_detection = idle_msg
                        last_quality   = quality_str
                        write_live_data()
                        print(f"  Window  --- → {idle_msg}")
                    else:
                        task_counts[label] += 1
                        window_count       += 1
                        last_detection      = TASK_LABELS.get(label, label)
                        last_quality        = f"{quality_str}  ({conf:.0%} confident)"
                        history.append({
                            "Window":      window_count,
                            "Movement":    last_detection,
                            "Quality":     quality_str,
                            "Confidence":  f"{conf:.0%}",
                            "Jerk":        round(jerk_val, 1),
                        })
                        write_live_data()
                        print(f"  Window {window_count:4d} → {last_detection}  [{quality_str}  {conf:.0%}]")
                except Exception as e:
                    print(f"  Classification error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nBD6 Live Sensor Bridge — Auto-detecting participant from sensor IDs")

    control  = xda.XsControl.construct()
    # Start with empty sensor map — will be filled after auto-detection
    callback = XsCallback({})

    # Scan for Awinda Station
    print("Scanning for Awinda Station...")
    port_info_array = xda.XsScanner.scanPorts()
    awinda_port = None
    for port_info in port_info_array:
        if port_info.deviceId().isAwinda2():
            awinda_port = port_info
            break

    if awinda_port is None:
        print("ERROR: Awinda Station not found. Make sure it is plugged in.")
        sys.exit(1)

    print(f"Found Awinda on port {awinda_port.portName()} at {awinda_port.baudrate()} baud")

    if not control.openPort(awinda_port.portName(), awinda_port.baudrate()):
        print("ERROR: Could not open port.")
        sys.exit(1)

    master_id = awinda_port.deviceId()
    master    = control.device(master_id)

    # Add callback to both control and master device
    control.addCallbackHandler(callback)
    master.addCallbackHandler(callback)

    print(f"Master device: {master_id}")

    # Make sure Awinda is in config mode and radio is on
    master.gotoConfig()
    try:
        master.enableRadio(19)   # channel 19 — change if sensors don't connect
        print("Radio enabled on channel 19")
    except Exception as e:
        print(f"  (enableRadio not available or already on: {e})")

    print("Waiting for MTW sensors to connect (turn them on now)...")
    print("  (Turn on the 4 MTW sensors now...)")

    # Wait until 4 sensors connect, then auto-detect which participant they belong to
    while True:
        with callback._lock:
            n_connected = len(callback.connected_mtws)
            connected_ids = list(callback.connected_mtws)
        print(f"  Connected {n_connected}/4    ", end="\r")
        if n_connected >= 4:
            break
        time.sleep(1)

    # Auto-detect participant from sensor IDs
    global participant_name
    participant_name, sensor_map = auto_detect_sensor_map(connected_ids)
    callback.sensor_map = sensor_map
    print(f"\nAll 4 sensors connected!")
    print(f"Auto-detected: {participant_name}")
    print(f"Sensor map: {sensor_map}")

    # Go to measurement mode
    if not master.gotoMeasurement():
        print("ERROR: Could not start measurement.")
        sys.exit(1)

    print("Streaming... Open the DSS in your browser and go to Live Monitor.")
    print("Press Ctrl+C to stop.\n")

    write_live_data(status="running")

    try:
        while True:
            time.sleep(0.5)
            write_live_data(status="running")
    except KeyboardInterrupt:
        print("\nStopping...")

    master.gotoConfig()
    control.closePort(awinda_port.portName())
    control.destruct()
    write_live_data(status="stopped")
    print("Done.")

if __name__ == "__main__":
    main()
