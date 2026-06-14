"""Two-pass streaming sampler for the full CICIoT2023 dataset.

Pass 1: count rows per 34-class label across all 169 CSV parts (label column
        only, so it is cheap on RAM).
Pass 2: stream the parts again and keep each row of a class with probability
        target/count, where target = min(count, cap) for classes above the
        downsample threshold and = count otherwise.

The result is written to a single parquet file with the raw 46 features plus
three label columns (label_34, label_8, label_binary) so downstream work can
switch granularity without re-sampling.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

import config as C

CHUNK = 400_000


def list_parts() -> list[str]:
    parts = sorted(glob.glob(os.path.join(C.DATA_DIR, "*.csv")))
    if not parts:
        raise FileNotFoundError(f"No CSV parts found in {C.DATA_DIR}")
    return parts


def pass1_counts(parts: list[str]) -> pd.Series:
    print(f"[pass 1] counting labels across {len(parts)} parts ...")
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, p in enumerate(parts, 1):
        for ch in pd.read_csv(p, usecols=[C.LABEL_COL], chunksize=CHUNK):
            vc = ch[C.LABEL_COL].value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)
        if i % 20 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)} parts  ({time.time()-t0:.0f}s)")
    s = pd.Series(counts).sort_values(ascending=False)
    unknown = set(s.index) - set(C.CLASS_TO_CATEGORY)
    if unknown:
        raise ValueError(f"Unmapped labels found: {unknown}")
    return s


def compute_targets(counts: pd.Series) -> dict[str, int]:
    targets = {}
    for label, n in counts.items():
        targets[label] = (C.PER_CLASS_CAP
                          if n > C.DOWNSAMPLE_THRESHOLD else int(n))
    return targets


def pass2_sample(parts: list[str], counts: pd.Series,
                 targets: dict[str, int]) -> pd.DataFrame:
    # Per-class keep probability.
    keep_p = {lbl: min(1.0, targets[lbl] / counts[lbl]) for lbl in counts.index}
    rng = np.random.default_rng(C.RANDOM_SEED)
    print("[pass 2] sampling ...")
    kept: list[pd.DataFrame] = []
    t0 = time.time()
    for i, p in enumerate(parts, 1):
        for ch in pd.read_csv(p, chunksize=CHUNK):
            # Vectorised Bernoulli keep mask using each row's class probability.
            probs = ch[C.LABEL_COL].map(keep_p).to_numpy()
            mask = rng.random(len(ch)) < probs
            if mask.any():
                kept.append(ch.loc[mask])
        if i % 20 == 0 or i == len(parts):
            n = sum(len(k) for k in kept)
            print(f"  {i}/{len(parts)} parts  kept={n:,}  ({time.time()-t0:.0f}s)")
    df = pd.concat(kept, ignore_index=True)
    return df


def main() -> None:
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    parts = list_parts()

    counts = pass1_counts(parts)
    targets = compute_targets(counts)

    summary = pd.DataFrame({
        "total": counts,
        "target": pd.Series(targets),
        "category": pd.Series({k: C.to_category(k) for k in counts.index}),
    })
    summary["downsampled"] = summary["total"] > C.DOWNSAMPLE_THRESHOLD
    print("\n=== per-class plan ===")
    print(summary.to_string())
    print(f"\nfull total rows: {counts.sum():,}")
    print(f"planned sample : {summary['target'].sum():,}")

    df = pass2_sample(parts, counts, targets)

    # Clean: replace inf, drop fully-NaN rows on features.
    df[C.RAW_FEATURES] = df[C.RAW_FEATURES].replace([np.inf, -np.inf], np.nan)

    # Derived label columns.
    df = df.rename(columns={C.LABEL_COL: "label_34"})
    df["label_8"] = df["label_34"].map(C.to_category)
    df["label_binary"] = df["label_34"].map(
        lambda x: "Benign" if x == C.BENIGN_LABEL else "Attack")

    # Cast features to float32 to halve the on-disk / in-RAM footprint.
    df[C.RAW_FEATURES] = df[C.RAW_FEATURES].astype("float32")

    df = df.sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)
    df.to_parquet(C.SAMPLED_PARQUET, index=False)

    summary.to_csv(os.path.join(C.REPORTS_DIR, "class_distribution.csv"))
    print(f"\nsaved {len(df):,} rows -> {C.SAMPLED_PARQUET}")
    print("actual sampled distribution:")
    print(df["label_34"].value_counts().to_string())


if __name__ == "__main__":
    main()
