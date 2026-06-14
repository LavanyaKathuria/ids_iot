"""Same DecisionTree pipeline as train.py, but targeting the 8 categories
(label_8) instead of the 34 fine-grained classes.

Reuses train.py's feature pruning, candidate configs, and metric helper so the
only difference is the target column. Reports accuracy, macro-F1, and per-class
F1 for the 8 classes.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from train import CONFIGS, metric_block, prune_features

TARGET = "label_8"


def main():
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)

    print(f"loading {C.SAMPLED_PARQUET} ...")
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    print(f"  {len(df):,} rows")
    df = engineer_features(df)
    X = df[C.RAW_FEATURES + ENGINEERED_FEATURES].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[TARGET]

    print("pruning features ...")
    feat_cols = prune_features(X)
    X = X[feat_cols]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"{len(le.classes_)} classes: {list(le.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=C.RANDOM_SEED, stratify=y_enc)
    print(f"train={len(X_tr):,} test={len(X_te):,}\n")

    best, best_model, best_cfg = None, None, None
    for cfg in CONFIGS:
        t = time.time()
        m = DecisionTreeClassifier(random_state=C.RANDOM_SEED, **cfg)
        m.fit(X_tr, y_tr)
        res = metric_block(str(cfg), y_te, m.predict(X_te))
        res["train_s"] = time.time() - t
        if best is None or res["f1_macro"] > best["f1_macro"]:
            best, best_model, best_cfg = res, m, cfg
    print(f"\nbest config (by macro-F1): {best_cfg}")

    pred = best_model.predict(X_te)
    print("\n=== 8-class results (best model) ===")
    print(f"  accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"  macro-F1 : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted-F1 : {f1_score(y_te, pred, average='weighted', zero_division=0):.4f}")

    per_f1 = f1_score(y_te, pred, average=None, zero_division=0)
    print("\n  per-class F1:")
    for cls, s in sorted(zip(le.classes_, per_f1), key=lambda kv: -kv[1]):
        print(f"    {cls:12s} {s:.4f}")

    rep = classification_report(y_te, pred, target_names=le.classes_,
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(os.path.join(C.REPORTS_DIR, "final8_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_te, pred), index=le.classes_,
                 columns=le.classes_).to_csv(
        os.path.join(C.REPORTS_DIR, "final8_confusion_matrix.csv"))

    model_path = os.path.join(C.MODELS_DIR, "dtree_8class.joblib")
    joblib.dump(best_model, model_path, compress=3)
    joblib.dump(le, os.path.join(C.MODELS_DIR, "label_encoder_8.joblib"))
    # 8-class shares the same engineered+pruned feature pipeline, so the
    # existing feature_list.json applies unchanged.
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"\nsaved -> {model_path} ({size_mb:.2f} MB, depth={best_model.get_depth()}, "
          f"leaves={best_model.get_n_leaves()})")

    with open(os.path.join(C.REPORTS_DIR, "metrics8.json"), "w") as fh:
        json.dump({"model": "DecisionTree", "target": TARGET, "config": best_cfg,
                   "n_classes": len(le.classes_), "n_features": len(feat_cols),
                   "model_size_mb": size_mb, "metrics": best}, fh, indent=2)
    print("metrics -> artifacts/reports/metrics8.json")


if __name__ == "__main__":
    main()
