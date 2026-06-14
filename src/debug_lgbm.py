"""Fast, controlled diagnosis of why LightGBM collapses on this data.

Small subsample + few trees so each cell runs in seconds. We isolate:
  A) raw 46 features, LightGBM DEFAULT params
  B) raw 46 features, MY params (num_leaves48, feature_fraction/bagging 0.8)
  C) all 64 features, DEFAULT params
  D) all 64 features, signed-log transform on heavy-tailed cols, DEFAULT params

We also print train time and the number of distinct predicted classes (a
collapsed model predicts very few).
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

N = 200_000
N_EST = 80


def signed_log(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def run(name, Xtr, Xte, ytr, yte, **params):
    t = time.time()
    m = lgb.LGBMClassifier(n_estimators=N_EST, n_jobs=-1,
                           random_state=C.RANDOM_SEED, verbose=-1, **params)
    m.fit(Xtr, ytr)
    dt = time.time() - t
    p = m.predict(Xte)
    acc = accuracy_score(yte, p)
    mf1 = f1_score(yte, p, average="macro", zero_division=0)
    ndist = len(np.unique(p))
    print(f"  {name:26s} acc={acc:.4f} macroF1={mf1:.4f} "
          f"predClasses={ndist:2d}/{len(np.unique(yte))} ({dt:.1f}s)")


def main():
    print("loading + subsampling ...")
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    df = (df.groupby("label_34", group_keys=False)
            .sample(frac=min(1.0, N / len(df)), random_state=C.RANDOM_SEED)
            .reset_index(drop=True))
    df = engineer_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df["label_34"])
    print(f"  rows={len(df):,}  classes={len(le.classes_)}\n")

    raw = C.RAW_FEATURES
    allf = C.RAW_FEATURES + ENGINEERED_FEATURES

    def split(cols, transform=False):
        X = df[cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if transform:
            X = X.apply(signed_log)
        return train_test_split(X, y, test_size=0.25,
                                random_state=C.RANDOM_SEED, stratify=y)

    my = dict(num_leaves=48, max_depth=11, learning_rate=0.12,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              min_child_samples=50)

    print("A) raw46, DEFAULT params")
    Xtr, Xte, ytr, yte = split(raw)
    run("raw46_default", Xtr, Xte, ytr, yte)

    print("B) raw46, MY params")
    run("raw46_myparams", Xtr, Xte, ytr, yte, **my)

    print("C) all64, DEFAULT params")
    Xtr, Xte, ytr, yte = split(allf)
    run("all64_default", Xtr, Xte, ytr, yte)

    print("D) all64 + signed-log, DEFAULT params")
    Xtr, Xte, ytr, yte = split(allf, transform=True)
    run("all64_log_default", Xtr, Xte, ytr, yte)


if __name__ == "__main__":
    main()
