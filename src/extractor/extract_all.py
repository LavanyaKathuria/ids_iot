"""Extract every per-class pcap -> one labeled feature CSV per class, with
GUARANTEED no-duplicate-session rows.

De-duplication by CONTENT FINGERPRINT (size + md5 of head/tail), not filename:
  * two files with identical content (e.g. same session downloaded twice, or
    renamed) -> same fingerprint -> extracted ONCE, included ONCE.
  * a per-fingerprint cache means unchanged files are never re-extracted, and
    only new/changed pcaps (new fingerprint) are processed.
Each class CSV is rebuilt from its current pcaps' caches every run, so it always
reflects exactly the present files with no stale or duplicated rows. Every row
carries 'source' (pcap basename) and 'fp' (fingerprint) columns.
"""
from __future__ import annotations

import glob, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import config as C
from pcap_to_features import extract_features

MAX_PACKETS = 150_000               # per session; rare/small files run full
OUT_DIR = os.path.join(C.ARTIFACTS, "pcap_features")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")


def fingerprint(path: str) -> str:
    sz = os.path.getsize(path)
    h = hashlib.md5(str(sz).encode())
    with open(path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))            # head
        if sz > 4 * 1024 * 1024:
            f.seek(-2 * 1024 * 1024, os.SEEK_END)     # tail
            h.update(f.read(2 * 1024 * 1024))
    return h.hexdigest()[:16]


def label_for(fname: str) -> str | None:
    n = os.path.basename(fname).lower().replace(".pcap", "").replace(".cap", "")
    n = n.replace("-", "").replace("_", "").rstrip("0123456789")
    if "benign" in n:
        return "BenignTraffic"
    cands = {c: c.lower().replace("-", "").replace("_", "") for c in C.CLASS_TO_CATEGORY}
    for c, cn in cands.items():
        if n == cn:
            return c
    hits = [c for c, cn in sorted(cands.items(), key=lambda kv: -len(kv[1]))
            if cn in n or n in cn]
    return hits[0] if hits else None


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    files = sorted(glob.glob(os.path.join(C.ROOT, "pcap", "*.pcap")) +
                   glob.glob(os.path.join(C.ROOT, "pcap", "*.cap")))
    print(f"{len(files)} pcap files present\n", flush=True)

    seen_fp: dict[str, str] = {}             # fp -> first filename (this run)
    class_caches: dict[str, list[str]] = {}
    for f in files:
        lab = label_for(f)
        if lab is None:
            print(f"  ?? unmapped: {os.path.basename(f)}", flush=True); continue
        fp = fingerprint(f)
        cache = os.path.join(CACHE_DIR, f"{fp}.csv")
        if fp in seen_fp:                    # identical content already handled
            print(f"  DUP CONTENT skip: {os.path.basename(f)} == {seen_fp[fp]}", flush=True)
            continue
        seen_fp[fp] = os.path.basename(f)
        class_caches.setdefault(lab, []).append(cache)
        if os.path.exists(cache):
            print(f"  cached: {os.path.basename(f)} ({manifest.get(fp,{}).get('rows','?')} rows)", flush=True)
            continue
        t0 = time.time()
        df = extract_features(f, max_packets=MAX_PACKETS, label=lab, progress_every=0)
        df["source"] = os.path.basename(f); df["fp"] = fp
        df.to_csv(cache, index=False)
        manifest[fp] = {"file": os.path.basename(f), "label": lab, "rows": len(df)}
        json.dump(manifest, open(MANIFEST, "w"), indent=2)
        print(f"  extracted {os.path.basename(f)} [{lab}]: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)

    # Session-level dedup is already guaranteed by fingerprint (each unique
    # session content extracted once). We do NOT drop feature-identical rows:
    # distinct time-windows that happen to share feature values are real
    # observations (common in uniform floods), not duplicated sessions.
    print("\nbuilding per-class CSVs (session-deduped by fingerprint):", flush=True)
    for lab, caches in sorted(class_caches.items()):
        parts = [pd.read_csv(c) for c in dict.fromkeys(caches)]   # unique caches
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(OUT_DIR, f"{lab}.csv"), index=False)
        print(f"  {lab:24s} {len(df):>8,} rows  ({df['source'].nunique()} session(s))", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
