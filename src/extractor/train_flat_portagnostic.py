"""Flat, port-agnostic Stage-2 model (NO anomaly gate).

One DecisionTree over the 62 BEHAVIOURAL features (port-identity features
dropped, see config.PORT_FEATURES) that classifies every window directly as
Benign or one of the 26 merged attack classes (27 total). Use this when on-site
benign calibration for the anomaly gate is not available.

DecisionTree chosen over an MLP: marginally lower in-distribution macro-F1 but
markedly more ROBUST on out-of-distribution attack pcaps (the MLP mislabels
clean SYN floods as Recon-Scanning) and ~3.6x faster per flow.

Real-capture augmentation: own-network pcaps listed in captures/train_manifest.csv
(file,label) are extracted and appended (x AUG_REPS) to the training data. This
closed the UDP-flood domain-shift hole found in cross-laptop testing (real
Wi-Fi UDP floods were mislabelled until their own captures were added). Verified
no regression on any non-web class (>0.02 rule). The manifest pcaps are
git-ignored; deployment keeps them locally. Add rows to extend to other classes.

Saves:  flat_dtree_portagnostic.joblib, label_encoder_flat.joblib,
        feature_list_behavioral.json   (under artifacts/models/pathA/)
"""
from __future__ import annotations

import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from fast_transform import build_matrix
from pcap_to_features import extract_features

FEATS = [f for f in C.RAW_FEATURES + ENGINEERED_FEATURES if f not in C.PORT_FEATURES]
OUT = os.path.join(C.MODELS_DIR, "pathA")
MANIFEST = os.path.join(C.ROOT, "captures", "train_manifest.csv")
AUG_REPS = 3                      # upweight real own-network captures vs ~1.3M CICIoT rows
CAP_PER_CLASS = 6000             # cap manifest windows per class -> balanced (no UDP bias)


def dt():
    return DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                  class_weight="balanced", random_state=C.RANDOM_SEED)


def load_manifest_aug(le):
    """Extract own-network pcaps from the manifest, balanced per class (capped to
    CAP_PER_CLASS) so no single class dominates -> (X_aug, y_aug encoded)."""
    if not os.path.exists(MANIFEST):
        return None, None
    man = pd.read_csv(MANIFEST)
    by_label = {}
    for _, r in man.iterrows():
        p = os.path.join(C.ROOT, "captures", r["file"])
        if not os.path.exists(p):
            print(f"  manifest: MISSING {r['file']} (skipped)"); continue
        M = build_matrix(extract_features(p), FEATS)
        by_label.setdefault(r["label"], []).append(M)
        print(f"  manifest: {r['file']} -> {len(M)} windows as {r['label']}")
    if not by_label:
        return None, None
    rng = np.random.default_rng(C.RANDOM_SEED)
    mats, labs = [], []
    for lbl, ms in by_label.items():
        M = np.vstack(ms)
        if len(M) > CAP_PER_CLASS:
            M = M[rng.choice(len(M), CAP_PER_CLASS, replace=False)]
        mats.append(M); labs += [lbl] * len(M)
        print(f"  balanced: {lbl} -> {len(M)} windows")
    return np.vstack(mats), le.transform(np.array(labs))


def main():
    print(f"{len(FEATS)} behavioural features (dropped {len(C.PORT_FEATURES)} port features)")
    df = pd.concat([pd.read_csv(f) for f in
                    sorted(glob.glob(os.path.join(C.ARTIFACTS, "pcap_features", "*.csv")))],
                   ignore_index=True)
    lab = df["label"].to_numpy()
    y_lab = np.array([l if l == C.BENIGN_LABEL else C.to_deploy_merged(l) for l in lab])
    X = (engineer_features(df)[FEATS].astype("float32")
         .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy())
    le = LabelEncoder().fit(y_lab)
    y = le.transform(y_lab)
    print(f"{len(X):,} CICIoT rows, {len(le.classes_)} classes (Benign + {len(le.classes_)-1} attacks)")

    print("loading real-capture augmentation:")
    Xa, ya = load_manifest_aug(le)
    aug = (Xa is not None)
    if aug:
        print(f"  total {len(Xa):,} augmentation windows x{AUG_REPS}\n")

    def stack(Xb, yb):
        if not aug:
            return Xb, yb
        return np.vstack([Xb] + [Xa] * AUG_REPS), np.concatenate([yb] + [ya] * AUG_REPS)

    # held-out report (augmentation in TRAIN only; report on CICIoT test)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2,
                                          random_state=C.RANDOM_SEED, stratify=y)
    Xtr2, ytr2 = stack(Xtr, ytr)
    rep = pd.DataFrame(classification_report(
        yte, dt().fit(Xtr2, ytr2).predict(Xte), target_names=le.classes_,
        digits=4, output_dict=True, zero_division=0)).T
    rep["support"] = rep["support"].astype(int)
    cls = rep.loc[~rep.index.isin(["accuracy", "macro avg", "weighted avg"])
                  ].sort_values("f1-score", ascending=False)
    print(cls[["precision", "recall", "f1-score", "support"]].round(4).to_string())
    print("\n" + rep.loc[["accuracy", "macro avg", "weighted avg"]]
          [["precision", "recall", "f1-score", "support"]].round(4).to_string())
    cls[["precision", "recall", "f1-score", "support"]].to_csv(
        os.path.join(C.REPORTS_DIR, "flat_portagnostic_per_class.csv"))

    # final model on ALL rows + augmentation
    Xall, yall = stack(X, y)
    model = dt().fit(Xall, yall)
    joblib.dump(model, os.path.join(OUT, "flat_dtree_portagnostic.joblib"), compress=3)
    joblib.dump(le, os.path.join(OUT, "label_encoder_flat.joblib"))
    with open(os.path.join(OUT, "feature_list_behavioral.json"), "w") as fh:
        json.dump(FEATS, fh, indent=2)
    sz = os.path.getsize(os.path.join(OUT, "flat_dtree_portagnostic.joblib")) / 1e6
    print(f"\nsaved -> flat_dtree_portagnostic.joblib ({sz:.2f} MB), "
          f"{len(le.classes_)} classes, {len(FEATS)} features, aug={aug}")


if __name__ == "__main__":
    main()
