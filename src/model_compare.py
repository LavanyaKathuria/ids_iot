"""34-class model bake-off (LightGBM excluded: its multiclass collapses here).

Ranks candidates by accuracy / macro-F1 / weighted-F1 AND by saved model size,
so we can pick the best model that fits the <10 MB Pi budget.
"""
from __future__ import annotations

import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
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

N = 800_000


def size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
        path = f.name
    joblib.dump(model, path, compress=3)
    mb = os.path.getsize(path) / 1e6
    os.remove(path)
    return mb


def evaluate(name, model, Xtr, ytr, Xte, yte):
    t = time.time(); model.fit(Xtr, ytr); dt = time.time() - t
    p = model.predict(Xte)
    row = dict(model=name,
               accuracy=accuracy_score(yte, p),
               bal_acc=balanced_accuracy_score(yte, p),
               f1_macro=f1_score(yte, p, average="macro", zero_division=0),
               f1_weighted=f1_score(yte, p, average="weighted", zero_division=0),
               size_mb=size_mb(model), train_s=dt)
    print(f"  {name:24s} acc={row['accuracy']:.4f} macroF1={row['f1_macro']:.4f} "
          f"wF1={row['f1_weighted']:.4f} balAcc={row['bal_acc']:.4f} "
          f"size={row['size_mb']:.1f}MB ({dt:.0f}s)")
    return row


def main():
    print("loading + subsampling ...")
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    df = (df.groupby("label_34", group_keys=False)
            .sample(frac=min(1.0, N / len(df)), random_state=C.RANDOM_SEED)
            .reset_index(drop=True))
    df = engineer_features(df)
    feat = C.RAW_FEATURES + ENGINEERED_FEATURES
    X = df[feat].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = LabelEncoder().fit_transform(df["label_34"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=C.RANDOM_SEED, stratify=y)
    print(f"  train={len(Xtr):,} test={len(Xte):,}  ({len(np.unique(y))} classes)\n")

    rows = []
    rows.append(evaluate("DecisionTree", DecisionTreeClassifier(
        max_depth=30, min_samples_leaf=5, random_state=C.RANDOM_SEED),
        Xtr, ytr, Xte, yte))

    rows.append(evaluate("RandomForest_40", RandomForestClassifier(
        n_estimators=40, max_depth=30, min_samples_leaf=3, n_jobs=-1,
        random_state=C.RANDOM_SEED), Xtr, ytr, Xte, yte))

    rows.append(evaluate("RF_30_leaf2000", RandomForestClassifier(
        n_estimators=30, max_leaf_nodes=2000, min_samples_leaf=3, n_jobs=-1,
        random_state=C.RANDOM_SEED), Xtr, ytr, Xte, yte))

    rows.append(evaluate("ExtraTrees_40", ExtraTreesClassifier(
        n_estimators=40, max_depth=30, min_samples_leaf=3, n_jobs=-1,
        random_state=C.RANDOM_SEED), Xtr, ytr, Xte, yte))

    rows.append(evaluate("HistGBDT", HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=63, learning_rate=0.2,
        random_state=C.RANDOM_SEED), Xtr, ytr, Xte, yte))

    if HAVE_XGB:
        rows.append(evaluate("XGBoost_hist", XGBClassifier(
            n_estimators=400, max_depth=10, learning_rate=0.3,
            tree_method="hist", objective="multi:softmax",
            n_jobs=-1, random_state=C.RANDOM_SEED, verbosity=0),
            Xtr, ytr, Xte, yte))
    else:
        print("  (xgboost unavailable)")

    print("\n=== ranked by macro-F1 ===")
    res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
