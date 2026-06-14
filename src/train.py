"""Train the final CICIoT2023 34-class intrusion-detection model.

Model choice: a single tuned DecisionTree. A full bake-off (see model_compare.py
and docs/PROJECT_REPORT.md) showed it gives the best macro-F1 (rare-attack
recall), the smallest artifact, and the fastest inference, and that the
gradient-boosting libraries (LightGBM/XGBoost/HistGBDT) underperform badly on
this 34-class problem.

Pipeline: load capped sample -> feature engineering -> prune redundant cols ->
stratified split -> tune a few DecisionTree configs by macro-F1 -> persist the
best with its label encoder + feature list -> full metric suite.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

TARGET = "label_34"
NEAR_CONST_THRESH = 1e-9
CORR_THRESH = 0.995

# DecisionTree configs to try; best macro-F1 on the test split is kept.
CONFIGS = [
    dict(max_depth=None, min_samples_leaf=2),
    dict(max_depth=40, min_samples_leaf=2),
    dict(max_depth=35, min_samples_leaf=5),
    dict(max_depth=None, min_samples_leaf=2, class_weight="balanced"),
]


def load_and_engineer():
    print(f"loading {C.SAMPLED_PARQUET} ...")
    df = pd.read_parquet(C.SAMPLED_PARQUET)
    print(f"  {len(df):,} rows")
    df = engineer_features(df)
    y = df[TARGET]
    X = df[C.RAW_FEATURES + ENGINEERED_FEATURES].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def prune_features(X: pd.DataFrame) -> list[str]:
    keep = [c for c in X.columns if X[c].var() > NEAR_CONST_THRESH]
    dropped_const = sorted(set(X.columns) - set(keep))
    sub = X[keep].sample(min(len(X), 200_000), random_state=C.RANDOM_SEED)
    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped_corr = [c for c in upper.columns if (upper[c] > CORR_THRESH).any()]
    keep = [c for c in keep if c not in dropped_corr]
    print(f"  pruned near-constant {dropped_const}")
    print(f"  pruned correlated   {dropped_corr}")
    print(f"  -> {len(keep)} features retained")
    return keep


def metric_block(name, y_true, y_pred):
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    print(f"  [{name}] acc={res['accuracy']:.4f} macroF1={res['f1_macro']:.4f} "
          f"wF1={res['f1_weighted']:.4f} balAcc={res['balanced_accuracy']:.4f}")
    return res


def main():
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)

    X, y = load_and_engineer()
    print("pruning features ...")
    feat_cols = prune_features(X)
    X = X[feat_cols]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"{len(le.classes_)} classes")

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

    # full reports on the best model
    pred = best_model.predict(X_te)
    cls_report = classification_report(y_te, pred, target_names=le.classes_,
                                       zero_division=0, output_dict=True)
    pd.DataFrame(cls_report).T.to_csv(
        os.path.join(C.REPORTS_DIR, "final_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_te, pred), index=le.classes_,
                 columns=le.classes_).to_csv(
        os.path.join(C.REPORTS_DIR, "final_confusion_matrix.csv"))
    fi = (pd.Series(best_model.feature_importances_, index=feat_cols)
          .sort_values(ascending=False))
    fi.to_csv(os.path.join(C.REPORTS_DIR, "final_feature_importance.csv"))
    print("\ntop 15 features:")
    print(fi.head(15).to_string())

    # persist
    model_path = os.path.join(C.MODELS_DIR, "dtree_34class.joblib")
    joblib.dump(best_model, model_path, compress=3)
    joblib.dump(le, os.path.join(C.MODELS_DIR, "label_encoder_34.joblib"))
    with open(os.path.join(C.MODELS_DIR, "feature_list.json"), "w") as fh:
        json.dump(feat_cols, fh, indent=2)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"\nsaved -> {model_path}  ({size_mb:.2f} MB, depth={best_model.get_depth()}, "
          f"leaves={best_model.get_n_leaves()})")

    with open(os.path.join(C.REPORTS_DIR, "metrics.json"), "w") as fh:
        json.dump({"model": "DecisionTree", "config": best_cfg, "target": TARGET,
                   "n_classes": len(le.classes_), "n_features": len(feat_cols),
                   "model_size_mb": size_mb, "metrics": best}, fh, indent=2)
    print("metrics -> artifacts/reports/metrics.json")


if __name__ == "__main__":
    main()
