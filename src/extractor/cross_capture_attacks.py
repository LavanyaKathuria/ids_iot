"""Cross-capture (leave-one-session-out) generalization test for attack classes.

For every class with >1 capture session: hold out ONE whole session, train on
everything else (including the class's OTHER sessions), and measure recall on
the held-out session. Compare to within-capture recall (random split). If
cross-capture << within-capture, that attack does NOT generalize across capture
sessions — the same failure we found for benign.
"""
from __future__ import annotations

import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES


def dt():
    return DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                  class_weight="balanced", random_state=C.RANDOM_SEED)


def per_class_recall(y_true, y_pred, classes):
    out = {}
    for ci, c in enumerate(classes):
        m = y_true == ci
        out[c] = np.mean(y_pred[m] == ci) if m.any() else np.nan
    return out


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    le = LabelEncoder(); y = le.fit_transform(df["label"]); src = df["source"].to_numpy()
    print(f"{len(df):,} rows, {len(le.classes_)} classes\n")

    # (1) within-capture recall (random 80/20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=C.RANDOM_SEED, stratify=y)
    p = dt().fit(Xtr, ytr).predict(Xte)
    within = per_class_recall(yte, p, le.classes_)

    # (2) cross-capture: hold out one whole session per multi-session class
    test_mask = np.zeros(len(df), bool); held = {}
    for c in le.classes_:
        s = sorted(set(src[df["label"].to_numpy() == c]))
        if len(s) > 1:
            held[c] = s[-1]
            test_mask |= (df["label"].to_numpy() == c) & (src == s[-1])
    pcr = dt().fit(X[~test_mask], y[~test_mask]).predict(X[test_mask])
    yte2 = y[test_mask]
    cross = {}
    for c in held:
        ci = le.transform([c])[0]; m = yte2 == ci
        cross[c] = np.mean(pcr[m] == ci) if m.any() else np.nan

    rows = []
    for c in held:
        rows.append((c, len(set(src[df["label"].to_numpy()==c])), within[c], cross[c],
                     within[c]-cross[c]))
    tbl = pd.DataFrame(rows, columns=["class","sessions","within_recall","cross_recall","gap"]).sort_values("cross_recall")
    pd.set_option("display.width", 160)
    print("CROSS-CAPTURE (leave-one-session-out) vs WITHIN-CAPTURE recall:")
    print(tbl.round(3).to_string(index=False))
    print(f"\n  mean within={tbl.within_recall.mean():.3f}  mean cross={tbl.cross_recall.mean():.3f}  "
          f"mean gap={tbl['gap'].mean():.3f}")
    print(f"  classes that GENERALIZE (cross>=0.8): {int((tbl.cross_recall>=.8).sum())}/{len(tbl)}")
    print(f"  classes that COLLAPSE  (cross<0.5):  {int((tbl.cross_recall<.5).sum())}/{len(tbl)}")


if __name__ == "__main__":
    main()
