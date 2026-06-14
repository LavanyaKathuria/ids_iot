"""Focused cross-capture benign test: does any model generalize benign better
than the DecisionTree's 14%? Train binary (attack vs benign) with benign from 2
captures, test benign recall on a held-out 3rd capture (+ attack recall)."""
from __future__ import annotations

import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
HELD = "benign.pcap"          # held-out benign capture (matches earlier 14% test)


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    lab = df["label"].to_numpy(); src = df["source"].to_numpy()
    is_b = lab == "BenignTraffic"; is_att = ~is_b
    is_held_b = is_b & (src == HELD)
    print(f"benign captures: {sorted(set(src[is_b]))}  held-out={HELD}")
    print(f"benign train rows={int((is_b&~is_held_b).sum()):,}  held-out benign={int(is_held_b.sum()):,}\n")

    rng = np.random.default_rng(42)
    att = np.where(is_att)[0]
    att_tr = rng.choice(att, min(len(att), 300_000), replace=False)
    tr = np.zeros(len(df), bool); tr[att_tr] = True; tr |= (is_b & ~is_held_b)
    y = is_att.astype(int)
    Xb = X[is_held_b]                       # held-out benign
    att_te = rng.choice(att, 50_000, replace=False)

    candidates = {
        "DecisionTree": DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                          class_weight="balanced", random_state=42),
        "RandomForest_50": RandomForestClassifier(n_estimators=50, max_depth=30,
                          min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample",
                          random_state=42),
        "RF_40_leaf3000": RandomForestClassifier(n_estimators=40, max_leaf_nodes=3000,
                          min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample",
                          random_state=42),
    }
    print("model            benign-recall(unseen)   attack-recall")
    for name, m in candidates.items():
        t=time.time(); m.fit(X[tr], y[tr])
        br = np.mean(m.predict(Xb) == 0) * 100
        ar = np.mean(m.predict(X[att_te]) == 1) * 100
        print(f"  {name:16s} {br:5.1f}%                  {ar:5.1f}%   ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
