"""Finalize the deployable two-stage system:

  * Stage-2 attack-type classifier: ATTACKS ONLY (no Benign), DoS+DDoS merged
    into one 'Flood' category -> 6 attack categories. Saves model + encoder.
  * Reports within-capture per-category F1 and the cross-capture recall of a
    held-out DoS-UDP session as 'Flood' (confirms the merge fixed the DoS
    generalization failure: it was 0.11 as a separate class).
  * Re-saves the anomaly gate at 50 trees (Pi latency).
"""
from __future__ import annotations

import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from save_anomaly_gate import train_gate

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
OUT = os.path.join(C.MODELS_DIR, "pathA")


def dt():
    return DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                                  class_weight="balanced", random_state=C.RANDOM_SEED)


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    lab = df["label"].to_numpy(); src = df["source"].to_numpy()
    att = lab != C.BENIGN_LABEL
    cat = np.array([C.to_category_merged(l) for l in lab])
    le = LabelEncoder().fit(C.ATTACK_CATEGORIES_MERGED)

    # ---- within-capture metrics (attacks only, merged) ----
    Xa, ya = X[att], le.transform(cat[att])
    Xtr,Xte,ytr,yte = train_test_split(Xa, ya, test_size=.2, random_state=C.RANDOM_SEED, stratify=ya)
    p = dt().fit(Xtr,ytr).predict(Xte)
    print(f"Stage-2 attacks-only ({len(le.classes_)} merged categories): "
          f"acc={accuracy_score(yte,p):.4f} macroF1={f1_score(yte,p,average='macro',zero_division=0):.4f}")
    for c,f in sorted(zip(le.classes_, f1_score(yte,p,average=None,zero_division=0)), key=lambda kv:-kv[1]):
        print(f"    {c:12s} F1={f:.3f}")

    # ---- cross-capture: held-out DoS-UDP session -> should now be 'Flood' ----
    dossrc = sorted(set(src[lab=="DoS-UDP_Flood"]))
    held = dossrc[-1]
    test = (lab=="DoS-UDP_Flood") & (src==held)
    train = att & ~test
    m = dt().fit(X[train], le.transform(cat[train]))
    flood_recall = np.mean(le.inverse_transform(m.predict(X[test]))=="Flood")*100
    print(f"\ncross-capture held-out DoS-UDP session -> recall as 'Flood': {flood_recall:.1f}%  "
          f"(was 0.11 as a separate DoS-UDP class)")

    # ---- save final Stage-2 (trained on ALL attack rows) ----
    mfinal = dt().fit(Xa, ya)
    joblib.dump(mfinal, os.path.join(OUT, "pathA_dtree_attackcat.joblib"), compress=3)
    joblib.dump(le, os.path.join(OUT, "label_encoder_attackcat.joblib"))
    print(f"\nsaved Stage-2 -> pathA_dtree_attackcat.joblib "
          f"({os.path.getsize(os.path.join(OUT,'pathA_dtree_attackcat.joblib'))/1e6:.2f} MB)")

    # ---- re-save gate at 50 trees ----
    bX = X[~att]
    joblib.dump(train_gate(bX), os.path.join(OUT, "anomaly_gate.joblib"), compress=3)
    print(f"re-saved anomaly gate (50 trees) on {int((~att).sum()):,} benign rows")


if __name__ == "__main__":
    main()
