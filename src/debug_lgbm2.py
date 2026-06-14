"""Find the winning LightGBM recipe: log-transform + expressiveness + precision.

All cells use the EXACT same split on the 34-class task.
Root cause so far: extreme-range features (float32) wreck LightGBM's histogram
bins. Signed-log compresses them. We now combine that with deeper trees and
test whether float64 alone also helps.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import engineer_features

N = 200_000


def signed_log(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df) * np.log1p(np.abs(df))


def score(name, m, Xtr, ytr, Xte, yte):
    t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
    p = m.predict(Xte)
    print(f"  {name:34s} acc={accuracy_score(yte,p):.4f} "
          f"macroF1={f1_score(yte,p,average='macro',zero_division=0):.4f} "
          f"pred={len(np.unique(p))}/{len(np.unique(yte))} ({dt:.1f}s)")


def main():
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    df = (df.groupby("label_34", group_keys=False)
            .sample(frac=min(1.0, N / len(df)), random_state=C.RANDOM_SEED)
            .reset_index(drop=True))
    df = engineer_features(df)
    feats = C.RAW_FEATURES  # raw only, to isolate the transform/expressiveness
    Xraw = df[feats].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xlog = signed_log(Xraw).astype("float32")
    y = LabelEncoder().fit_transform(df["label_34"])
    print(f"rows={len(df):,}  raw46 features\n")

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=C.RANDOM_SEED, stratify=y)

    def sp(X):
        return X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te], y[idx_te]

    print("-- reference --")
    Xtr, ytr, Xte, yte = sp(Xraw)
    score("dtree_depth25 (raw)", DecisionTreeClassifier(
        max_depth=25, min_samples_leaf=20, random_state=C.RANDOM_SEED),
        Xtr, ytr, Xte, yte)

    common = dict(n_estimators=100, learning_rate=0.1, n_jobs=-1,
                  random_state=C.RANDOM_SEED, verbose=-1)

    print("\n-- raw float32 (baseline failure) --")
    score("lgbm_raw_leaves255", lgb.LGBMClassifier(num_leaves=255, **common),
          Xtr, ytr, Xte, yte)

    print("\n-- raw as float64 (precision hypothesis) --")
    Xtr64, _, Xte64, _ = sp(Xraw.astype("float64"))
    score("lgbm_raw64_leaves255", lgb.LGBMClassifier(num_leaves=255, **common),
          Xtr64, ytr, Xte64, yte)

    print("\n-- signed-log + expressiveness --")
    Xtr, ytr, Xte, yte = sp(Xlog)
    score("lgbm_log_leaves255", lgb.LGBMClassifier(num_leaves=255, **common),
          Xtr, ytr, Xte, yte)
    score("lgbm_log_leaves512", lgb.LGBMClassifier(num_leaves=512, **common),
          Xtr, ytr, Xte, yte)
    score("lgbm_log_leaves1024_mcs20",
          lgb.LGBMClassifier(num_leaves=1024, min_child_samples=20, **common),
          Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    main()
