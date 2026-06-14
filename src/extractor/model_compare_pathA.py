"""Model bake-off on Path A (extractor) multi-session features.

Two questions:
  (1) Which model is best on extractor features? (acc / macro-F1 / size)
  (2) Which generalizes BENIGN best across captures? (train benign on 2
      captures, test on a held-out 3rd -> benign recall) -- the metric that
      matters for the benign problem.
"""
from __future__ import annotations

import glob, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

FEAT_DIR = os.path.join(C.ARTIFACTS, "pcap_features")
FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
SUB = 500_000


def size_mb(m):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib").name
    joblib.dump(m, f, compress=3); mb = os.path.getsize(f)/1e6; os.remove(f); return mb


def models():
    d = {
        "DecisionTree": DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                          class_weight="balanced", random_state=42),
        "RandomForest_50": RandomForestClassifier(n_estimators=50, max_depth=30,
                          min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample",
                          random_state=42),
        "RF_40_leaf3000": RandomForestClassifier(n_estimators=40, max_leaf_nodes=3000,
                          min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample",
                          random_state=42),
        "ExtraTrees_50": ExtraTreesClassifier(n_estimators=50, max_depth=30,
                          min_samples_leaf=3, n_jobs=-1, class_weight="balanced_subsample",
                          random_state=42),
        "HistGBDT": HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63,
                          learning_rate=0.2, random_state=42),
    }
    if HAVE_XGB:
        d["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.3,
                          tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
    return d


def main():
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FEAT_DIR, "*.csv")))]
    df = pd.concat(parts, ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0)
    print(f"{len(df):,} rows\n")

    # subsample for the multiclass bake-off
    idx = (df.groupby("label", group_keys=False).sample(frac=min(1,SUB/len(df)),
                                                        random_state=42).index)
    Xs, ys_lab = X.loc[idx], df.loc[idx, "label"]
    le = LabelEncoder(); ys = le.fit_transform(ys_lab)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=.2, random_state=42, stratify=ys)

    print("=== (1) 34-class bake-off (extractor features) ===")
    for name, m in models().items():
        t=time.time()
        try:
            m.fit(Xtr, ytr); p=m.predict(Xte); dt=time.time()-t
            print(f"  {name:16s} acc={accuracy_score(yte,p):.4f} "
                  f"macroF1={f1_score(yte,p,average='macro',zero_division=0):.4f} "
                  f"size={size_mb(m):5.1f}MB ({dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name:16s} FAILED: {e}", flush=True)

    # (2) cross-capture benign: train benign on cap1+cap2, test on held-out cap3
    print("\n=== (2) CROSS-CAPTURE benign recall per model ===")
    bsrc = sorted(df.loc[df.label=="BenignTraffic","source"].unique())
    held = bsrc[0]
    print(f"  benign captures: {bsrc}  | held-out test = {held}\n")
    is_b = (df.label=="BenignTraffic").to_numpy()
    is_held_b = is_b & (df.source==held).to_numpy()
    is_att = ~is_b
    rng=np.random.default_rng(42)
    att_idx=np.where(is_att)[0]
    att_sub=rng.choice(att_idx,min(len(att_idx),300_000),replace=False)
    train_mask=np.zeros(len(df),bool); train_mask[att_sub]=True
    train_mask |= (is_b & ~is_held_b)        # benign cap1+cap2 in train
    yb=is_att.astype(int)
    for name,m in models().items():
        try:
            t=time.time(); m.fit(X[train_mask], yb[train_mask]); dt=time.time()-t
            bp=m.predict(X[is_held_b])
            ap=m.predict(X[is_att][rng.choice(is_att.sum(),50000,replace=False)])
            print(f"  {name:16s} benign-recall(unseen cap)={np.mean(bp==0)*100:5.1f}%  "
                  f"attack-recall={np.mean(ap==1)*100:5.1f}%  ({dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name:16s} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
