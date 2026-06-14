"""Raspberry-Pi inference — TWO-STAGE IDS (Path A).

Stage 1 — benign anomaly gate (Isolation Forest): "is this normal for THIS
          network?"  -> calibrate on-site for ~95% benign accuracy.
Stage 2 — attack-type classifier (Path A DecisionTree): labels the flows the
          gate flags as anomalous.

Rationale: a supervised benign class does NOT generalise across networks
(~14% benign recall). The gate, calibrated on the deployment network's own
benign, reaches ~95% while still flagging ~92% of attacks (see
src/extractor/anomaly_gate_test.py).

Deploy: copy config.py, fast_transform.py, feature_engineering.py, this file,
and artifacts/models/pathA/. Your tcpdump -> CICFlowMeter pipeline yields the
raw 46 CICIoT2023 columns.

    from predict import IDSModel
    ids = IDSModel()                       # 8-class attack types + benign gate
    ids.calibrate_benign(local_benign_df)  # ON-SITE: tune the gate to this network
    labels = ids.predict(flow_df)          # 'BenignTraffic' or an attack name
    detail = ids.predict_detailed(flow_df) # is_anomaly / attack_type / label
"""
from __future__ import annotations

import json, os, sys, time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from fast_transform import build_matrix, signed_log  # noqa: E402


class IDSModel:
    def __init__(self, models_dir: str | None = None, model: str = "attack"):
        # model (attacks-only Stage-2, benign handled by the gate):
        #   "attack"    -> 18 fine-grained classes, only DoS+DDoS merged to Flood (default)
        #   "attackcat" -> 6 broad attack categories
        #   "8class"/"34class" -> legacy, include a Benign class
        models_dir = models_dir or os.path.join(C.MODELS_DIR, "pathA")
        self._dir = models_dir
        self.model = joblib.load(os.path.join(models_dir, f"pathA_dtree_{model}.joblib"))
        enc = {"attack": "label_encoder_attack.joblib",
               "attackcat": "label_encoder_attackcat.joblib",
               "8class": "label_encoder_8.joblib",
               "34class": "label_encoder_34.joblib"}[model]
        self.le = joblib.load(os.path.join(models_dir, enc))
        with open(os.path.join(models_dir, "feature_list.json")) as fh:
            self.features = json.load(fh)
        self.gate = joblib.load(os.path.join(models_dir, "anomaly_gate.joblib"))
        if hasattr(self.model, "feature_names_in_"):
            del self.model.feature_names_in_

    # ---- shared ----
    def _matrix(self, flow_df: pd.DataFrame) -> np.ndarray:
        return build_matrix(flow_df, self.features)

    def _is_anomaly(self, X: np.ndarray) -> np.ndarray:
        g = self.gate
        return g["iso"].predict(g["scaler"].transform(signed_log(X))) == -1   # True = anomalous

    # ---- inference ----
    def predict(self, flow_df: pd.DataFrame) -> np.ndarray:
        """Stage1 gate -> 'BenignTraffic' for normal; Stage2 attack name for anomalous."""
        X = self._matrix(flow_df)
        anom = self._is_anomaly(X)
        out = np.full(len(X), "BenignTraffic", dtype=object)
        if anom.any():
            out[anom] = self.le.inverse_transform(self.model.predict(X[anom]))
        return out

    def predict_detailed(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        X = self._matrix(flow_df)
        anom = self._is_anomaly(X)
        atype = np.full(len(X), "", dtype=object)
        if anom.any():
            atype[anom] = self.le.inverse_transform(self.model.predict(X[anom]))
        return pd.DataFrame({"is_anomaly": anom, "attack_type": atype,
                             "label": np.where(anom, atype, "BenignTraffic")})

    # ---- on-site calibration (the key deployment step) ----
    def calibrate_benign(self, benign_flow_df: pd.DataFrame, save: bool = True):
        """Re-fit Stage-1 on THIS network's own benign traffic. Run once on a
        clean benign capture from the deployment network."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        X = self._matrix(benign_flow_df)
        contam = self.gate.get("contamination", 0.05)
        sc = StandardScaler().fit(signed_log(X))
        iso = IsolationForest(n_estimators=50, contamination=contam,   # 50 trees for Pi latency
                              random_state=C.RANDOM_SEED, n_jobs=-1).fit(sc.transform(signed_log(X)))
        self.gate = {"scaler": sc, "iso": iso, "contamination": contam}
        if save:
            joblib.dump(self.gate, os.path.join(self._dir, "anomaly_gate.joblib"), compress=3)
        return self


def _benchmark(csv_path: str) -> None:
    ids = IDSModel()
    df = pd.read_csv(csv_path)
    for c in (C.LABEL_COL, "label", "source", "fp"):
        if c in df.columns:
            df = df.drop(columns=[c])
    ids.predict(df.head(64))                                  # warm-up
    for batch in (1, 32, 256, len(df)):
        s = df.head(batch); n = 50 if batch <= 256 else 5
        t0 = time.perf_counter()
        for _ in range(n):
            ids.predict(s)
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"  batch={batch:>6}: {dt:8.3f} ms/batch ({dt/batch*1000:7.2f} us/flow)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("latency benchmark (two-stage):")
        _benchmark(sys.argv[1])
    else:
        print("usage: python predict.py <flows.csv>")
