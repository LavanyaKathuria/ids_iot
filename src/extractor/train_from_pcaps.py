"""Path A: train/evaluate a model on features produced by OUR extractor.

Reads artifacts/pcap_features/<label>.csv (one per class, with 'label' and
'source' columns). Training and serving both use pcap_to_features.py -> zero
train/serve skew, no IAT artifact. The honest end-to-end estimate.

Reports: 34-class, 8-class, binary metrics + per-class F1, a per-class
row-count vs F1 table (to flag which classes need more pcaps), and a
cross-capture benign generalization test if two benign captures exist.
"""
from __future__ import annotations

import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEAT_DIR = os.path.join(C.ARTIFACTS, "pcap_features")
FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES


def load():
    parts = []
    for f in sorted(glob.glob(os.path.join(FEAT_DIR, "*.csv"))):
        d = pd.read_csv(f)
        if "label" not in d:
            d["label"] = os.path.basename(f)[:-4]
        if "source" not in d:
            d["source"] = os.path.basename(f)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def matrix(df):
    X = engineer_features(df)[FEATS].astype("float32")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_eval(name, X, y, classes, weight="balanced"):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2,
                                          random_state=C.RANDOM_SEED, stratify=y)
    m = DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                               class_weight=weight, random_state=C.RANDOM_SEED).fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"\n=== {name} ===  acc={accuracy_score(yte,p):.4f} "
          f"macroF1={f1_score(yte,p,average='macro',zero_division=0):.4f} "
          f"weightedF1={f1_score(yte,p,average='weighted',zero_division=0):.4f}")
    return m, yte, p


def main():
    df = load()
    print(f"loaded {len(df):,} rows, {df['label'].nunique()} classes")
    X = matrix(df)
    counts = df["label"].value_counts()

    # ---- 34-class ----
    le = LabelEncoder(); y34 = le.fit_transform(df["label"])
    m, yte, p = fit_eval("34-class (extractor features)", X, y34, le.classes_)
    per = dict(zip(le.classes_, f1_score(yte, p, average=None, zero_division=0)))
    tbl = pd.DataFrame({"rows": counts, "F1": pd.Series(per)}).sort_values("F1")
    print("\nper-class F1 vs row count (sorted weakest first):")
    print(tbl.to_string())

    # ---- 8-class ----
    y8 = LabelEncoder().fit_transform(df["label"].map(C.CLASS_TO_CATEGORY))
    cats = sorted(set(df["label"].map(C.CLASS_TO_CATEGORY)))
    _, y8t, p8 = fit_eval("8-class (categories)", X, y8, cats)
    for c, s in sorted(zip(LabelEncoder().fit(df['label'].map(C.CLASS_TO_CATEGORY)).classes_,
                           f1_score(y8t, p8, average=None, zero_division=0)), key=lambda kv:-kv[1]):
        print(f"    {c:12s} F1={s:.3f}")

    # ---- binary ----
    yb = (df["label"] != "BenignTraffic").astype(int).to_numpy()
    fit_eval("binary attack/benign", X, yb, ["Benign", "Attack"])

    # ---- cross-capture benign generalization ----
    bsrc = df.loc[df["label"] == "BenignTraffic", "source"].unique()
    if len(bsrc) >= 2:
        cap2 = bsrc[-1]
        is_b2 = ((df["label"] == "BenignTraffic") & (df["source"] == cap2)).to_numpy()
        is_att = (df["label"] != "BenignTraffic").to_numpy()
        rng = np.random.default_rng(C.RANDOM_SEED)
        att = np.where(is_att)[0]; att_te = np.zeros(len(df), bool)
        att_te[rng.choice(att, int(.2*len(att)), replace=False)] = True
        tr = (~att_te & is_att) | (~is_att & ~is_b2)
        mc = DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                    class_weight="balanced", random_state=C.RANDOM_SEED).fit(X[tr], is_att[tr].astype(int))
        b2p = mc.predict(X[is_b2])
        print(f"\n=== CROSS-CAPTURE benign (train on other capture, test on '{cap2}') ===")
        print(f"  unseen-capture benign correctly called Benign: {np.mean(b2p==0)*100:.1f}%"
              f"  (false-alarm: {np.mean(b2p==1)*100:.1f}%)")


if __name__ == "__main__":
    main()
