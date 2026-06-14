"""Compare interpolation-based oversampling for the weak app-layer classes.

Conditions (same train/test split, 28-class proto-merged, attacks only):
  baseline   : current approach = DecisionTree(class_weight='balanced'), no oversampling
  SMOTE      : oversample only the weak classes, DT (no class_weight)
  Borderline : BorderlineSMOTE (boundary-focused)
  ADASYN     : ADASYN (density/difficulty-weighted)

Reports per-class F1 for the 6 weak classes AND a strong-class summary
(to check the boundary methods don't hurt the strong classes). NO generative
synthesis — interpolation only.
"""
from __future__ import annotations

import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import config as C
from feature_engineering import ENGINEERED_FEATURES, engineer_features
from fast_transform import signed_log

FEATS = C.RAW_FEATURES + ENGINEERED_FEATURES
WEAK = ["XSS","SqlInjection","CommandInjection","BrowserHijacking","Backdoor_Malware","Uploading_Attack"]
CAP = 20_000          # cap majority classes in TRAIN (keeps neighbour search tractable)
TARGET = 12_000       # oversample each weak class up to this


def dt(**kw):
    return DecisionTreeClassifier(max_depth=35, min_samples_leaf=5, random_state=C.RANDOM_SEED, **kw)


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(C.ARTIFACTS,"pcap_features","*.csv")))],
                   ignore_index=True)
    lab = df["label"].to_numpy(); att = lab != C.BENIGN_LABEL
    X = engineer_features(df)[FEATS].astype("float32").replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy()[att]
    y_lab = np.array([C.to_proto_flood_merged(l) for l in lab])[att]
    le = LabelEncoder().fit(y_lab); y = le.transform(y_lab)
    # scale (signed-log + standardise) so SMOTE neighbours are sensible; DT is invariant
    sc = StandardScaler()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=C.RANDOM_SEED, stratify=y)
    Xtr = sc.fit_transform(signed_log(Xtr)); Xte = sc.transform(signed_log(Xte))

    # cap majority classes in train
    rng = np.random.default_rng(C.RANDOM_SEED); keep = []
    for c in np.unique(ytr):
        idx = np.where(ytr==c)[0]
        keep.append(rng.choice(idx, min(len(idx), CAP), replace=False))
    keep = np.concatenate(keep); Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"train={len(ytr):,} (capped {CAP}/class)  test={len(yte):,}  {len(le.classes_)} classes")

    weak_idx = [le.transform([w])[0] for w in WEAK]
    cur = pd.Series(ytr).value_counts()
    strat = {c: TARGET for c in weak_idx if cur.get(c,0) < TARGET}
    print(f"oversampling weak classes -> {TARGET} each: {WEAK}\n")

    samplers = {
        "baseline(cls_wt)": None,
        "SMOTE":      SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=C.RANDOM_SEED),
        "Borderline": BorderlineSMOTE(sampling_strategy=strat, k_neighbors=5, m_neighbors=10, random_state=C.RANDOM_SEED),
        "ADASYN":     ADASYN(sampling_strategy=strat, n_neighbors=5, random_state=C.RANDOM_SEED),
    }
    res = {}
    for name, smp in samplers.items():
        t = time.time()
        if smp is None:
            m = dt(class_weight="balanced").fit(Xtr, ytr)
        else:
            Xr, yr = smp.fit_resample(Xtr, ytr)
            m = dt().fit(Xr, yr)
        f1 = f1_score(yte, m.predict(Xte), average=None, zero_division=0)
        res[name] = dict(zip(le.classes_, f1))
        print(f"  {name:16s} done ({time.time()-t:.0f}s)")

    tab = pd.DataFrame(res)
    print("\n=== per-class F1: WEAK classes ===")
    print(tab.loc[WEAK].round(4).to_string())
    print(f"\n  weak-class MEAN F1:  " + "  ".join(f"{k}={tab.loc[WEAK,k].mean():.4f}" for k in res))
    strong = [c for c in le.classes_ if c not in WEAK]
    print("\n=== STRONG classes (check no degradation) ===")
    print(f"  strong MEAN F1: " + "  ".join(f"{k}={tab.loc[strong,k].mean():.4f}" for k in res))
    print(f"  strong MIN  F1: " + "  ".join(f"{k}={tab.loc[strong,k].min():.4f}" for k in res))


if __name__ == "__main__":
    main()
