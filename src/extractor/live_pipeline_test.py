"""Simulate live traffic through the FULL two-stage pipeline and report accuracy.

Unlike train_final_merged.py (which scores Stage-2 alone on attacks), this runs
the deployed pipeline end to end: Stage-1 benign gate -> Stage-2 26-class. An
attack flow is only 'correct' if the gate flags it anomalous AND Stage-2 gives
the right merged class; a benign flow is correct if the gate passes it.

Honest setup: train on 80%, draw the 'live' flows from the held-out 20% (no
leakage). Same model config + scheme as the deployed artifacts.

Caveat: these flows come from the same captures as training, so this is an
IN-DISTRIBUTION upper bound. Real-world generalisation is the cross-capture
numbers (floods ~97%+, benign weak cross-network -> needs on-site gate
calibration, Web/brute weak). 'Live' here = unseen rows, not a new network.
"""
from __future__ import annotations

import glob, os, sys
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from fast_transform import signed_log

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
SEED = C.RANDOM_SEED
PER_CLASS = 100          # 'a few flows' sampled per class to replicate live arrivals


def main():
    df = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(C.ARTIFACTS, "pcap_features", "*.csv")))],
                   ignore_index=True)
    X = (engineer_features(df)[FEATS].astype("float32")
         .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy())
    raw = df["label"].to_numpy()
    truth = np.array([l if l == C.BENIGN_LABEL else C.to_final_merged(l) for l in raw])
    benign = truth == C.BENIGN_LABEL

    tr, te = train_test_split(np.arange(len(df)), test_size=.2,
                              random_state=SEED, stratify=truth)
    tr_set = set(tr)

    # ---- train deployed-config models on the 80% ----
    le = LabelEncoder().fit(truth[tr][~benign[tr]])
    clf = DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                 class_weight="balanced", random_state=SEED)
    clf.fit(X[tr][~benign[tr]], le.transform(truth[tr][~benign[tr]]))
    sc = StandardScaler().fit(signed_log(X[tr][benign[tr]]))
    iso = IsolationForest(n_estimators=50, contamination=0.05, random_state=SEED,
                          n_jobs=-1).fit(sc.transform(signed_log(X[tr][benign[tr]])))

    def pipeline(Xrows):
        anom = iso.predict(sc.transform(signed_log(Xrows))) == -1
        out = np.full(len(Xrows), C.BENIGN_LABEL, dtype=object)
        if anom.any():
            out[anom] = le.inverse_transform(clf.predict(Xrows[anom]))
        return out

    # ---- sample 'live' flows from the held-out 20%, per class ----
    rng = np.random.default_rng(SEED)
    rows = []
    for cls in sorted(set(truth)):
        idx = np.array([i for i in te if truth[i] == cls])
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(PER_CLASS, len(idx)), replace=False)
        pred = pipeline(X[pick])
        acc = float(np.mean(pred == cls))
        rows.append((cls, len(pick), acc))

    rep = pd.DataFrame(rows, columns=["true_class", "n_flows", "accuracy"]).set_index("true_class")
    rep = rep.sort_values("accuracy", ascending=False)
    print(rep.round(3).to_string())

    # ---- headline metrics on the sampled live stream ----
    allpick = np.concatenate([rng.choice(np.array([i for i in te if truth[i] == c]),
                              size=min(PER_CLASS, sum(truth[te] == c)), replace=False)
                              for c in sorted(set(truth)) if (truth[te] == c).any()])
    pred_all = pipeline(X[allpick])
    tru_all = truth[allpick]
    is_atk = tru_all != C.BENIGN_LABEL
    print(f"\nsampled flows: {len(allpick)}  ({is_atk.sum()} attack / {(~is_atk).sum()} benign)")
    print(f"overall accuracy           : {np.mean(pred_all == tru_all):.3f}")
    print(f"benign pass-rate (gate)    : {np.mean(pred_all[~is_atk] == C.BENIGN_LABEL):.3f}")
    print(f"attack catch-rate (gate)   : {np.mean(pred_all[is_atk] != C.BENIGN_LABEL):.3f}")
    print(f"attack exact-class accuracy: {np.mean(pred_all[is_atk] == tru_all[is_atk]):.3f}")


if __name__ == "__main__":
    main()
