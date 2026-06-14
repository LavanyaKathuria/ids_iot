"""Stage-2 with PROTOCOL-WISE flood merge (max granularity):
DoS-X_Flood + DDoS-X_Flood -> X_Flood (UDP/SYN/TCP/HTTP); everything else stays
as the original 34 classes. Attacks only (benign handled by the gate).

Reports class count, per-class precision/recall/F1, saves the model, and runs a
leave-one-session-out cross-capture test on the previously-collapsing DoS flood
sessions to confirm the merge fixes generalization.
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
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()
    lab = df["label"].to_numpy(); src = df["source"].to_numpy()
    att = lab != C.BENIGN_LABEL
    merged = np.array([C.to_proto_flood_merged(l) for l in lab])
    le = LabelEncoder().fit(merged[att])
    print(f"{len(le.classes_)} attack classes (protocol-wise flood merge):")
    print("  " + ", ".join(le.classes_) + "\n")

    Xa, ya = X[att], le.transform(merged[att])
    Xtr,Xte,ytr,yte = train_test_split(Xa, ya, test_size=.2, random_state=C.RANDOM_SEED, stratify=ya)
    rep = pd.DataFrame(classification_report(yte, dt().fit(Xtr,ytr).predict(Xte),
                       target_names=le.classes_, digits=4, output_dict=True, zero_division=0)).T
    rep["support"] = rep["support"].astype(int)
    cls = rep.loc[~rep.index.isin(["accuracy","macro avg","weighted avg"])].sort_values("f1-score", ascending=False)
    print(cls[["precision","recall","f1-score","support"]].round(4).to_string())
    print("\n" + rep.loc[["accuracy","macro avg","weighted avg"]][["precision","recall","f1-score","support"]].round(4).to_string())

    # ---- cross-capture: hold out one whole session of each merged flood constituent ----
    print("\nCROSS-CAPTURE (leave-one-session-out) on merged flood classes:")
    for orig in ["DoS-UDP_Flood","DoS-SYN_Flood","DoS-TCP_Flood","DoS-HTTP_Flood",
                 "DDoS-UDP_Flood","DDoS-SYN_Flood","DDoS-SynonymousIP_Flood"]:
        s = sorted(set(src[lab==orig]))
        if len(s) < 2:
            print(f"  {orig:18s} -> {C.to_proto_flood_merged(orig):12s}: (only 1 session, skipped)")
            continue
        held = s[-1]
        test = (lab==orig) & (src==held)
        train = att & ~test
        m = dt().fit(X[train], le.transform(merged[train]))
        tgt = C.to_proto_flood_merged(orig)
        rec = np.mean(le.inverse_transform(m.predict(X[test]))==tgt)*100
        print(f"  held-out {orig:16s} -> recall as {tgt:11s}: {rec:5.1f}%")

    # ---- save final on ALL attack rows ----
    mfinal = dt().fit(Xa, ya)
    joblib.dump(mfinal, os.path.join(OUT, "pathA_dtree_attack.joblib"), compress=3)
    joblib.dump(le, os.path.join(OUT, "label_encoder_attack.joblib"))
    print(f"\nsaved -> pathA_dtree_attack.joblib "
          f"({os.path.getsize(os.path.join(OUT,'pathA_dtree_attack.joblib'))/1e6:.2f} MB), "
          f"{len(le.classes_)} classes")


if __name__ == "__main__":
    main()
