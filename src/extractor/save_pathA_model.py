"""Train the FINAL Path A models on ALL multi-session extractor features and
save every artifact needed for Raspberry-Pi deployment.

No scaler is produced or needed: the model is a DecisionTree (scale-invariant)
and the preprocessing is only feature-engineering + inf/NaN -> 0. The full
pipeline on the Pi is:
    raw 46 features (CICFlowMeter)
      -> fast_transform.build_matrix(df, feature_list)   # adds 27 engineered, selects, cleans
      -> model.predict()                                 # -> class index
      -> label_encoder.inverse_transform()               # -> attack name
"""
from __future__ import annotations

import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEAT_DIR = os.path.join(C.ARTIFACTS, "pcap_features")
OUT = os.path.join(C.MODELS_DIR, "pathA")
FEATURES = C.RAW_FEATURES + ENGINEERED_FEATURES      # full set, no pruning


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(FEAT_DIR, "*.csv")))]
    df = pd.concat(parts, ignore_index=True)
    print(f"training on ALL {len(df):,} multi-session rows, {df['label'].nunique()} classes")

    X = engineer_features(df)[FEATURES].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # feature list (single source of truth for the Pi)
    json.dump(FEATURES, open(os.path.join(OUT, "feature_list.json"), "w"), indent=2)

    saved = []
    # 34-class
    le34 = LabelEncoder(); y34 = le34.fit_transform(df["label"])
    m34 = DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                 class_weight="balanced", random_state=C.RANDOM_SEED).fit(X, y34)
    joblib.dump(m34, os.path.join(OUT, "pathA_dtree_34class.joblib"), compress=3)
    joblib.dump(le34, os.path.join(OUT, "label_encoder_34.joblib"))
    saved += ["pathA_dtree_34class.joblib", "label_encoder_34.joblib"]

    # 8-class (categories)
    le8 = LabelEncoder(); y8 = le8.fit_transform(df["label"].map(C.CLASS_TO_CATEGORY))
    m8 = DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                class_weight="balanced", random_state=C.RANDOM_SEED).fit(X, y8)
    joblib.dump(m8, os.path.join(OUT, "pathA_dtree_8class.joblib"), compress=3)
    joblib.dump(le8, os.path.join(OUT, "label_encoder_8.joblib"))
    saved += ["pathA_dtree_8class.joblib", "label_encoder_8.joblib"]

    manifest = {
        "trained_on": "extractor features (Path A), all multi-session pcaps",
        "rows": int(len(df)), "n_features": len(FEATURES),
        "scaler": "NONE (DecisionTree is scale-invariant; preprocessing = feature engineering + NaN/inf->0)",
        "pipeline": ["raw 46 features", "fast_transform.build_matrix(df, feature_list)",
                     "model.predict", "label_encoder.inverse_transform"],
        "pi_files_needed": [
            "src/config.py", "src/feature_engineering.py", "src/fast_transform.py",
            "src/deploy/predict.py",
            "artifacts/models/pathA/pathA_dtree_8class.joblib (or 34class)",
            "artifacts/models/pathA/label_encoder_8.joblib (or 34)",
            "artifacts/models/pathA/feature_list.json",
        ],
        "caveat": "Attack-TYPE classifier. Benign does NOT generalize across networks "
                  "(cross-capture benign ~14%); calibrate benign on the deployment network.",
        "files": saved + ["feature_list.json"],
    }
    json.dump(manifest, open(os.path.join(OUT, "DEPLOYMENT_manifest.json"), "w"), indent=2)

    print("\nsaved to artifacts/models/pathA/:")
    for f in saved + ["feature_list.json", "DEPLOYMENT_manifest.json"]:
        p = os.path.join(OUT, f)
        print(f"  {f:32s} {os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
