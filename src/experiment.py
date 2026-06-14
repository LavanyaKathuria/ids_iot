"""Fast config search on a subsample to fix the LightGBM weighting problem.

Loads the sampled parquet, takes a stratified subsample, reuses the saved
feature list, and compares weighting schemes for LightGBM against the
DecisionTree reference. Prints an accuracy / macro-F1 / weighted-F1 / time
table so we can pick the right config before a full retrain.
"""
from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

SUBSAMPLE = 800_000
N_EST = 150


def softened_weights(y_enc: np.ndarray, power: float, cap: float) -> np.ndarray:
    """Per-sample weights = (N / (K * n_c))**power, normalised, capped."""
    counts = np.bincount(y_enc)
    K = len(counts)
    N = len(y_enc)
    w_per_class = (N / (K * counts)) ** power
    w_per_class = np.clip(w_per_class, 1.0 / cap, cap)
    w_per_class = w_per_class / w_per_class.mean()
    return w_per_class[y_enc]


def metrics(name, y, p, dt):
    row = {
        "config": name,
        "accuracy": accuracy_score(y, p),
        "bal_acc": balanced_accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
        "train_s": dt,
    }
    print(f"  {name:28s} acc={row['accuracy']:.4f} balacc={row['bal_acc']:.4f} "
          f"macroF1={row['f1_macro']:.4f} wF1={row['f1_weighted']:.4f} "
          f"({dt:.0f}s)")
    return row


def main():
    print("loading parquet ...")
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    # stratified subsample (proportional). groupby.sample keeps the group col,
    # unlike groupby.apply in pandas 3.0.
    frac = min(1.0, SUBSAMPLE / len(df))
    df = (df.groupby("label_34", group_keys=False)
            .sample(frac=frac, random_state=C.RANDOM_SEED)
            .reset_index(drop=True))
    print(f"  subsample rows: {len(df):,}")

    df = engineer_features(df)
    with open(os.path.join(C.MODELS_DIR, "feature_list.json")) as fh:
        feat_cols = json.load(fh)
    X = df[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    le = LabelEncoder()
    y = le.fit_transform(df["label_34"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=C.RANDOM_SEED, stratify=y)
    print(f"  train={len(X_tr):,} test={len(X_te):,}  ({len(le.classes_)} classes)\n")

    base = dict(objective="multiclass", n_estimators=N_EST, num_leaves=48,
                max_depth=11, learning_rate=0.12, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, min_child_samples=50,
                n_jobs=-1, random_state=C.RANDOM_SEED, verbose=-1)
    rows = []

    # A: no weighting
    t = time.time(); m = lgb.LGBMClassifier(**base).fit(X_tr, y_tr)
    rows.append(metrics("lgbm_none", y_te, m.predict(X_te), time.time()-t))

    # B: balanced (reproduce the failure)
    t = time.time(); m = lgb.LGBMClassifier(class_weight="balanced", **base).fit(X_tr, y_tr)
    rows.append(metrics("lgbm_balanced", y_te, m.predict(X_te), time.time()-t))

    # C: softened sqrt weights, capped 10x
    sw = softened_weights(y_tr, power=0.5, cap=10.0)
    t = time.time(); m = lgb.LGBMClassifier(**base).fit(X_tr, y_tr, sample_weight=sw)
    rows.append(metrics("lgbm_sqrt_cap10", y_te, m.predict(X_te), time.time()-t))

    # D: softened weights power=0.3, cap 6x
    sw = softened_weights(y_tr, power=0.3, cap=6.0)
    t = time.time(); m = lgb.LGBMClassifier(**base).fit(X_tr, y_tr, sample_weight=sw)
    rows.append(metrics("lgbm_p0.3_cap6", y_te, m.predict(X_te), time.time()-t))

    # E: DecisionTree balanced reference
    t = time.time()
    dt = DecisionTreeClassifier(max_depth=25, min_samples_leaf=20,
                                class_weight="balanced",
                                random_state=C.RANDOM_SEED).fit(X_tr, y_tr)
    rows.append(metrics("dtree_balanced", y_te, dt.predict(X_te), time.time()-t))

    print("\n=== summary ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
