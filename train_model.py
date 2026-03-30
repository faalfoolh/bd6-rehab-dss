"""
Run this ONCE to train and save the model to disk.
After this, the DSS app loads instantly.
  python3 train_model.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import signal
from scipy.stats import kurtosis, skew, entropy as sp_entropy
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SAMPLING_RATE = 100
WINDOW_SIZE   = 200
STEP_SIZE     = 100
BODY_PARTS    = ['Hand', 'Wrist', 'Elbow', 'Shoulder']

SENSORS_MAX_YUSUF       = {'00B44876':'Hand','00B44805':'Wrist','00B44856':'Elbow','00B44877':'Shoulder'}
SENSORS_SARA_ALFAF_RR   = {'00B447F7':'Hand','00B44804':'Wrist','00B4486D':'Elbow','00B44846':'Shoulder'}
SENSORS_SARA_ALFAF_OTHER= {'00B447FD':'Hand','00B447FA':'Wrist','00B447F1':'Elbow','00B44730':'Shoulder'}

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
    with open(filepath,'r') as f:
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

def augment_time_warp(w, sigma=0.1):
    n = len(w)
    tt = np.linspace(0,1,4)
    wp = np.sort(np.clip(tt + np.random.normal(0,sigma,4), 0, 1))
    wp[0], wp[-1] = 0.0, 1.0
    cs = CubicSpline(tt, wp)
    t0 = np.linspace(0,1,n)
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

# ── Load all files ──────────────────────────────────────────────────────────
print("Loading data files...")
records = []
for root, dirs, files in os.walk(BASE_DIR):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
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

print(f"  Loaded {len(records)} recordings")

# ── Build windows + augment ─────────────────────────────────────────────────
print("Building windows and augmenting...")
sessions = defaultdict(list)
for r in records:
    key = (r['participant'], r['task'], os.path.dirname(r['path']))
    sessions[key].append(r)

X_wins, y_wins = [], []
for (participant, task, sdir), recs in sessions.items():
    part_map = {r['body_part']: bandpass(r['data'].values) for r in recs}
    if not all(bp in part_map for bp in BODY_PARTS): continue
    min_len  = min(len(part_map[bp]) for bp in BODY_PARTS)
    combined = np.hstack([part_map[bp][:min_len] for bp in BODY_PARTS])
    for w in sliding_windows(combined, WINDOW_SIZE, STEP_SIZE):
        for aug in [w,
                    w + np.random.normal(0, 0.5, w.shape),
                    w * np.random.uniform(0.9, 1.1),
                    augment_time_warp(w)]:
            X_wins.append(aug); y_wins.append(task)

print(f"  {len(X_wins)} windows after augmentation")

# ── Feature extraction ──────────────────────────────────────────────────────
print("Extracting features...")
X_feat = np.array([extract_features(w) for w in X_wins])
y      = np.array(y_wins)

le      = LabelEncoder()
y_enc   = le.fit_transform(y)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_feat)

# ── Feature selection (SFS) ─────────────────────────────────────────────────
print("Running feature selection (SFS) — this takes ~2 mins, only once...")
sfs = SequentialFeatureSelector(LDA(), n_features_to_select=30,
                                direction='forward', scoring='accuracy',
                                cv=5, n_jobs=-1)
sfs.fit(X_sc, y_enc)
X_sel = X_sc[:, sfs.get_support()]
print(f"  Selected {sfs.get_support().sum()} features")

# ── Train SVM ───────────────────────────────────────────────────────────────
print("Training SVM classifier...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_sel, y_enc)
print("  Done!")

# ── Save model ──────────────────────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, 'bd6_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'svm': svm, 'scaler': scaler, 'sfs': sfs, 'le': le}, f, protocol=4)
print(f"Model saved to: {model_path}")

# ── Pre-compute classification results for all patients ──────────────────────
print("Pre-computing classification results for all patients...")

BODY_PARTS = ['Hand', 'Wrist', 'Elbow', 'Shoulder']

def classify_patient_offline(participant, records, svm, scaler, sfs, le):
    from collections import defaultdict
    p_records = [r for r in records if r['participant'] == participant]
    sessions  = defaultdict(list)
    for r in p_records:
        key = (r['task'], os.path.dirname(r['path']))
        sessions[key].append(r)

    task_counts = defaultdict(int)
    raw_signals = {}

    for (task, sdir), recs in sessions.items():
        part_map = {r['body_part']: r['data'] for r in recs}
        filt_map = {r['body_part']: bandpass(r['data'].values) for r in recs}
        if not all(bp in filt_map for bp in BODY_PARTS):
            continue
        min_len  = min(len(filt_map[bp]) for bp in BODY_PARTS)
        combined = np.hstack([filt_map[bp][:min_len] for bp in BODY_PARTS])
        wins     = sliding_windows(combined, WINDOW_SIZE, STEP_SIZE)
        if not wins: continue

        X     = np.array([extract_features(w) for w in wins])
        X_sc  = scaler.transform(X)[:, sfs.get_support()]
        preds = le.inverse_transform(svm.predict(X_sc))
        for p in preds:
            task_counts[p] += 1

        if task not in raw_signals:
            # Store only first 30 seconds (3000 samples) to keep file small
            raw_signals[task] = {
                bp: part_map[bp].iloc[:3000].reset_index(drop=True)
                for bp in BODY_PARTS if bp in part_map
            }

    return dict(task_counts), raw_signals

participants = sorted(set(r['participant'] for r in records))
precomputed  = {}
for p in participants:
    print(f"  Computing {p}...")
    task_counts, raw_signals = classify_patient_offline(p, records, svm, scaler, sfs, le)
    precomputed[p] = {'task_counts': task_counts, 'raw_signals': raw_signals}

# ── Save everything ──────────────────────────────────────────────────────────
print("Saving processed data...")
# Strip raw DataFrames from records to reduce file size — app only needs precomputed
lightweight_records = [
    {k: v for k, v in r.items() if k != 'data'}
    for r in records
]

data_path = os.path.join(BASE_DIR, 'bd6_data.pkl')
with open(data_path, 'wb') as f:
    pickle.dump({'records': lightweight_records, 'precomputed': precomputed}, f, protocol=4)
print(f"Data saved to: {data_path}")

print("\nAll done! Now run:  streamlit run dss_app.py")
