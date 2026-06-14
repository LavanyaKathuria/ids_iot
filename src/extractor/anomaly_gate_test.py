"""Two-stage architecture validation: can a one-class anomaly gate (Isolation
Forest) recognise benign across captures better than the ~14% supervised result?

Stage-1 gate is trained ONLY on benign (it learns "normal for this network").
We test two scenarios:
  * CROSS-NETWORK : train on benign captures A+B, test benign on a held-out
    capture C  -> does a benign gate transfer to a *different* network?
  * ON-SITE (same-network): train on 80% of one benign capture, test on the
    held-out 20% of the SAME capture -> the realistic on-site-calibration case.
Both also report how many ATTACKS the gate flags as anomalous (its real job).

Anomaly detectors are distance-based -> features are signed-log compressed and
StandardScaler-normalised (a scaler IS needed here, unlike the tree models).
"""
from __future__ import annotations

import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
CONTAM = 0.05          # ~5% of training benign treated as boundary


def slog(a):           # signed log compresses the huge-range features
    return np.sign(a) * np.log1p(np.abs(a))


def gate(train_X, testB_X, attack_X):
    sc = StandardScaler().fit(slog(train_X))
    iso = IsolationForest(n_estimators=200, contamination=CONTAM,
                         random_state=42, n_jobs=-1).fit(sc.transform(slog(train_X)))
    bn = np.mean(iso.predict(sc.transform(slog(testB_X))) == 1) * 100   # benign called normal
    aa = np.mean(iso.predict(sc.transform(slog(attack_X))) == -1) * 100  # attack flagged anomalous
    return bn, aa


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    lab = df["label"].to_numpy(); src = df["source"].to_numpy()
    is_b = lab == "BenignTraffic"
    rng = np.random.default_rng(42)
    attack_X = X[~is_b][rng.choice((~is_b).sum(), 80_000, replace=False)]
    bsrc = sorted(set(src[is_b]))
    print(f"benign captures: {bsrc}\n")
    print("scenario            benign-as-normal   attack-as-anomaly")

    # CROSS-NETWORK: hold out benign.pcap
    held = "benign.pcap"
    trB = X[is_b & (src != held)]; teB = X[is_b & (src == held)]
    bn, aa = gate(trB, teB, attack_X)
    print(f"  CROSS-NETWORK      {bn:5.1f}%             {aa:5.1f}%   (supervised was ~14%)")

    # ON-SITE: train/test within the same capture
    for cap in bsrc:
        idx = np.where(is_b & (src == cap))[0]
        rng.shuffle(idx); cut = int(.8*len(idx))
        bn, aa = gate(X[idx[:cut]], X[idx[cut:]], attack_X)
        print(f"  ON-SITE [{cap[:18]:18s}] {bn:5.1f}%             {aa:5.1f}%")


if __name__ == "__main__":
    main()
