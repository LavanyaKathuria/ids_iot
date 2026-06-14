"""Train and save the Stage-1 benign anomaly gate (Isolation Forest + scaler).

This DEFAULT gate is trained on the benign captures we have. In production it
MUST be re-calibrated on the deployment network's own benign traffic
(see IDSModel.calibrate_benign in deploy/predict.py) — cross-network benign is
only ~52%, but on-site-calibrated is ~95%.
"""
from __future__ import annotations

import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from fast_transform import signed_log

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
OUT = os.path.join(C.MODELS_DIR, "pathA", "anomaly_gate.joblib")
CONTAM = 0.05


def train_gate(benign_X: np.ndarray):
    """Return a dict {scaler, iso} fitted on benign feature rows (raw 73-col matrix)."""
    sc = StandardScaler().fit(signed_log(benign_X))
    iso = IsolationForest(n_estimators=50, contamination=CONTAM,   # 50 trees for Pi latency
                          random_state=C.RANDOM_SEED, n_jobs=-1).fit(sc.transform(signed_log(benign_X)))
    return {"scaler": sc, "iso": iso, "contamination": CONTAM}


def main():
    bcsv = os.path.join(C.ARTIFACTS, "pcap_features", "BenignTraffic.csv")
    df = pd.read_csv(bcsv)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    gate = train_gate(X)
    joblib.dump(gate, OUT, compress=3)
    print(f"saved DEFAULT benign gate ({len(df):,} benign rows) -> {OUT}  "
          f"({os.path.getsize(OUT)/1e6:.2f} MB)")
    print("NOTE: re-calibrate on the deployment network for ~95% benign (default cross-network ~52%).")


if __name__ == "__main__":
    main()
