"""Validate the model on REALISTIC class proportions (the full 46M distribution),
not the capped training distribution.

Two independent estimates:

(A) Exact prior-reweighting of the held-out (capped) confusion matrix.
    Per-class recall is invariant to class priors; precision is not. Reweighting
    the held-out confusion rates by the true full-dataset priors gives the exact
    realistic-proportion precision / recall / F1 with no sampling noise.

(B) Empirical held-out realistic sample. Build a test set whose proportions match
    the full 46M distribution, using ONLY rows the model never trained on:
      * dominant classes (>1M rows): genuinely unseen full-dataset rows
        (excluded from the training sample by row hash);
      * kept-in-full classes (<=1M rows): the held-out 20% test split.
    Then run the model and score.

If (A) and (B) agree, the realistic macro-F1 is trustworthy.
"""
from __future__ import annotations

import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config as C
from fast_transform import build_matrix

TEST_TOTAL = 2_000_000   # size of the empirical realistic test set
CHUNK = 400_000


def load_full_counts() -> pd.Series:
    dist = pd.read_csv(os.path.join(C.REPORTS_DIR, "class_distribution.csv"),
                       index_col=0)
    return dist["total"]


# ---------------------------------------------------------------------------
# (A) exact reweighting of the held-out confusion matrix
# ---------------------------------------------------------------------------
def reweighted_metrics(full_counts: pd.Series) -> None:
    cm = pd.read_csv(os.path.join(C.REPORTS_DIR, "final_confusion_matrix.csv"),
                     index_col=0)
    classes = list(cm.index)
    C_mat = cm.to_numpy(dtype="float64")            # C[k,c] true k, pred c
    N_k = C_mat.sum(axis=1, keepdims=True)          # held-out support per class
    R = C_mat / np.clip(N_k, 1, None)               # P(pred c | true k)
    pi = full_counts.reindex(classes).to_numpy("float64")
    pi = pi / pi.sum()                              # realistic priors
    J = R * pi[:, None]                             # realistic joint P(true k,pred c)
    recall = np.diag(R)                             # prior-invariant
    col = J.sum(axis=0)
    precision = np.divide(np.diag(J), col, out=np.zeros_like(col),
                          where=col > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=(precision + recall) > 0)
    acc = np.diag(J).sum()
    print("=== (A) exact reweighting to full-46M proportions ===")
    print(f"  accuracy      : {acc:.4f}")
    print(f"  macro-F1      : {f1.mean():.4f}")
    print(f"  weighted-F1   : {(f1 * pi).sum():.4f}")
    print(f"  macro recall  : {recall.mean():.4f}")
    print(f"  macro prec.   : {precision.mean():.4f}")
    worst = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1},
                         index=classes).sort_values("f1").head(8)
    print("  weakest 8 classes (realistic):")
    print(worst.round(3).to_string().replace("\n", "\n    "))
    return


# ---------------------------------------------------------------------------
# (B) empirical held-out realistic sample
# ---------------------------------------------------------------------------
def hash_rows(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[C.RAW_FEATURES], index=False).to_numpy()


def build_realistic_test(full_counts: pd.Series) -> pd.DataFrame:
    total_all = full_counts.sum()
    targets = (full_counts / total_all * TEST_TOTAL).round().astype(int)

    print("\n=== (B) building empirical held-out realistic test ===")
    samp = pd.read_parquet(C.SAMPLED_PARQUET)
    samp = samp.rename(columns={})  # no-op, keep names
    # reconstruct the exact held-out 20% split (same seed/stratify as train.py)
    y34 = samp["label_34"].to_numpy()
    idx_all = np.arange(len(samp))
    _, test_idx = train_test_split(idx_all, test_size=0.2,
                                   random_state=C.RANDOM_SEED, stratify=y34)
    held = samp.iloc[test_idx]
    sampled_hashes = np.sort(hash_rows(samp))

    capped = [c for c in full_counts.index if full_counts[c] > C.DOWNSAMPLE_THRESHOLD]
    keptfull = [c for c in full_counts.index if full_counts[c] <= C.DOWNSAMPLE_THRESHOLD]

    parts = []
    # kept-full classes: take target from the held-out split (already unseen)
    for c in keptfull:
        pool = held[held["label_34"] == c]
        n = min(targets[c], len(pool))
        parts.append(pool.sample(n, random_state=C.RANDOM_SEED))
    n_keptfull = sum(len(p) for p in parts)
    print(f"  kept-full classes: {n_keptfull:,} rows from held-out split")

    # capped classes: stream full data, keep UNSEEN rows with per-class prob
    keep_p = {c: min(1.0, targets[c] / max(1, full_counts[c] - C.PER_CLASS_CAP))
              for c in capped}
    capset = set(capped)
    collected = {c: [] for c in capped}
    need = {c: targets[c] for c in capped}
    rng = np.random.default_rng(C.RANDOM_SEED)
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, "*.csv")))
    for i, fpath in enumerate(files, 1):
        if all(len(collected[c]) >= need[c] for c in capped):
            break
        for ch in pd.read_csv(fpath, chunksize=CHUNK):
            ch = ch.rename(columns={C.LABEL_COL: "label_34"})
            ch = ch[ch["label_34"].isin(capset)]
            if ch.empty:
                continue
            h = hash_rows(ch)
            pos = np.searchsorted(sampled_hashes, h)
            pos = np.clip(pos, 0, len(sampled_hashes) - 1)
            unseen = sampled_hashes[pos] != h          # not in training sample
            ch = ch[unseen]
            if ch.empty:
                continue
            probs = ch["label_34"].map(keep_p).to_numpy()
            mask = rng.random(len(ch)) < probs
            ch = ch[mask]
            for c, g in ch.groupby("label_34"):
                if len(collected[c]) < need[c]:
                    collected[c].append(g)
        if i % 40 == 0:
            got = sum(sum(len(g) for g in collected[c]) for c in capped)
            print(f"    streamed {i}/{len(files)} files, capped rows={got:,}")
    for c in capped:
        if collected[c]:
            g = pd.concat(collected[c]).head(need[c])
            parts.append(g)
    test = pd.concat(parts, ignore_index=True)
    print(f"  total realistic test rows: {len(test):,}")
    return test


def main():
    full_counts = load_full_counts()
    reweighted_metrics(full_counts)

    test = build_realistic_test(full_counts)
    model = joblib.load(os.path.join(C.MODELS_DIR, "dtree_34class.joblib"))
    if hasattr(model, "feature_names_in_"):
        del model.feature_names_in_
    le = joblib.load(os.path.join(C.MODELS_DIR, "label_encoder_34.joblib"))
    with open(os.path.join(C.MODELS_DIR, "feature_list.json")) as fh:
        feats = json.load(fh)

    X = build_matrix(test, feats)
    y_true = le.transform(test["label_34"])
    y_pred = model.predict(X)

    print("\n=== (B) empirical realistic-proportion results ===")
    print(f"  rows          : {len(test):,}")
    print(f"  accuracy      : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  macro-F1      : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted-F1   : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  balanced acc. : {balanced_accuracy_score(y_true, y_pred):.4f}")

    rep = classification_report(y_true, y_pred, target_names=le.classes_,
                                zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(os.path.join(C.REPORTS_DIR,
                                            "realistic_per_class.csv"))
    print("  per-class -> artifacts/reports/realistic_per_class.csv")


if __name__ == "__main__":
    main()
