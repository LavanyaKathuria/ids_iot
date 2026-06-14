"""Crack the LightGBM-vs-DecisionTree mystery on raw46.

Key questions:
  1. Is LightGBM UNDERFITTING? -> compare TRAIN vs TEST accuracy.
  2. Is it multiclass-specific? -> try binary and 8-class.
  3. Threading bug? -> n_jobs=1.
  4. Does aggressive optimization (more rounds / higher lr) help?
  5. What does it actually predict? -> prediction class histogram.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import engineer_features

N = 150_000


def tt(name, m, Xtr, ytr, Xte, yte):
    t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
    tr = accuracy_score(ytr, m.predict(Xtr))
    te = accuracy_score(yte, m.predict(Xte))
    print(f"  {name:28s} TRAIN={tr:.4f} TEST={te:.4f} ({dt:.1f}s)")
    return m


def main():
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    df = (df.groupby("label_34", group_keys=False)
            .sample(frac=min(1.0, N / len(df)), random_state=C.RANDOM_SEED)
            .reset_index(drop=True))
    df = engineer_features(df)
    X = df[C.RAW_FEATURES].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y34 = LabelEncoder().fit_transform(df["label_34"])
    y8 = LabelEncoder().fit_transform(df["label_8"])
    ybin = (df["label_binary"] == "Attack").astype(int).to_numpy()
    print(f"rows={len(df):,}  raw46\n")

    Xtr, Xte, ytr, yte, y8tr, y8te, ybtr, ybte = train_test_split(
        X, y34, y8, ybin, test_size=0.25, random_state=C.RANDOM_SEED, stratify=y34)

    common = dict(n_jobs=-1, random_state=C.RANDOM_SEED, verbose=-1)

    print("-- reference DecisionTree (34-class) --")
    tt("dtree_depth25", DecisionTreeClassifier(max_depth=25, min_samples_leaf=20,
                                               random_state=C.RANDOM_SEED),
       Xtr, ytr, Xte, yte)

    print("\n-- LightGBM 34-class: underfit check --")
    m = tt("lgbm_31_lr0.1_100", lgb.LGBMClassifier(
        n_estimators=100, num_leaves=31, learning_rate=0.1, **common),
        Xtr, ytr, Xte, yte)
    pc = pd.Series(m.predict(Xte)).value_counts()
    print(f"    predicts {len(pc)} classes; top-5 predicted idx counts: "
          f"{pc.head(5).to_dict()}")
    print(f"    true distribution top-5: {pd.Series(yte).value_counts().head(5).to_dict()}")

    print("\n-- aggressive optimization --")
    tt("lgbm_255_lr0.3_300", lgb.LGBMClassifier(
        n_estimators=300, num_leaves=255, learning_rate=0.3, **common),
        Xtr, ytr, Xte, yte)

    print("\n-- threading: n_jobs=1 --")
    tt("lgbm_31_njobs1", lgb.LGBMClassifier(
        n_estimators=100, num_leaves=31, learning_rate=0.1,
        n_jobs=1, random_state=C.RANDOM_SEED, verbose=-1),
        Xtr, ytr, Xte, yte)

    print("\n-- easier targets --")
    tt("lgbm_BINARY", lgb.LGBMClassifier(
        n_estimators=100, num_leaves=31, **common), Xtr, ybtr, Xte, ybte)
    tt("lgbm_8class", lgb.LGBMClassifier(
        n_estimators=100, num_leaves=31, **common), Xtr, y8tr, Xte, y8te)


if __name__ == "__main__":
    main()
