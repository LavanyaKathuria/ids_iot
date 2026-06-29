"""Final Stage-2 scheme: protocol-wise flood merge (incl. SynonymousIP->SYN)
PLUS recon-scan merge (PingSweep+PortScan+OSScan -> Recon-Scanning).
Attacks only (benign handled by the gate). -> 26 attack classes.

Reports per-class precision/recall/F1 and saves the deployable model. Supersedes
train_proto_merged.py as the canonical Stage-2 trainer.
"""
from __future__ import annotations

import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
OUT = os.path.join(C.MODELS_DIR, "pathA")


def dt():
    return DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                  class_weight="balanced", random_state=C.RANDOM_SEED)


def main():
    df = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(C.ARTIFACTS, "pcap_features", "*.csv")))],
                   ignore_index=True)
    X = (engineer_features(df)[FEATS].astype("float32")
         .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy())
    lab = df["label"].to_numpy()
    att = lab != C.BENIGN_LABEL
    merged = np.array([C.to_final_merged(l) for l in lab])
    le = LabelEncoder().fit(merged[att])
    print(f"{len(le.classes_)} attack classes (proto-flood + recon-scan merge):")
    print("  " + ", ".join(le.classes_) + "\n")

    Xa, ya = X[att], le.transform(merged[att])
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=.2,
                                          random_state=C.RANDOM_SEED, stratify=ya)
    rep = pd.DataFrame(classification_report(
        yte, dt().fit(Xtr, ytr).predict(Xte), target_names=le.classes_,
        digits=4, output_dict=True, zero_division=0)).T
    rep["support"] = rep["support"].astype(int)
    cls = rep.loc[~rep.index.isin(["accuracy", "macro avg", "weighted avg"])
                  ].sort_values("f1-score", ascending=False)
    print(cls[["precision", "recall", "f1-score", "support"]].round(4).to_string())
    print("\n" + rep.loc[["accuracy", "macro avg", "weighted avg"]]
          [["precision", "recall", "f1-score", "support"]].round(4).to_string())
    cls[["precision", "recall", "f1-score", "support"]].to_csv(
        os.path.join(C.REPORTS_DIR, "final_merged_per_class.csv"))

    # save final on ALL attack rows
    mfinal = dt().fit(Xa, ya)
    joblib.dump(mfinal, os.path.join(OUT, "pathA_dtree_attack.joblib"), compress=3)
    joblib.dump(le, os.path.join(OUT, "label_encoder_attack.joblib"))
    print(f"\nsaved -> pathA_dtree_attack.joblib "
          f"({os.path.getsize(os.path.join(OUT,'pathA_dtree_attack.joblib'))/1e6:.2f} MB), "
          f"{len(le.classes_)} classes")


if __name__ == "__main__":
    main()
