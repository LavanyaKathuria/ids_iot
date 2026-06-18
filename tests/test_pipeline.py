"""End-to-end smoke test: pcap -> 46 features -> two-stage IDS -> labels.

Runs the FULL deployment pipeline on a pcap and prints what came out. With no
argument it uses tests/sample_traffic.pcap (run make_test_pcap.py first); you can
also point it at any real capture to validate accuracy.

    python make_test_pcap.py          # 1) build the synthetic capture
    python test_pipeline.py           # 2) run the pipeline on it
    python test_pipeline.py /path/to/real_attack.pcap --models ../artifacts/models/pathA
"""
from __future__ import annotations

import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "extractor"),
          os.path.join(ROOT, "src", "deploy")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from pcap_to_features import extract_features        # noqa: E402
from predict import IDSModel                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="IDS end-to-end smoke test")
    ap.add_argument("pcap", nargs="?",
                    default=os.path.join(HERE, "sample_traffic.pcap"))
    ap.add_argument("--models", default=None, help="path to models/pathA")
    args = ap.parse_args()

    if not os.path.exists(args.pcap):
        sys.exit(f"no pcap at {args.pcap} — run: python make_test_pcap.py")

    print(f"[1/3] extracting features from {os.path.basename(args.pcap)} ...")
    df = extract_features(args.pcap)
    print(f"      -> {len(df):,} flow rows x {df.shape[1]} features")
    if len(df) == 0:
        sys.exit("no rows extracted — is the pcap empty / unreadable?")

    print("[2/3] loading two-stage IDS ...")
    ids = IDSModel(models_dir=args.models)
    print(f"      -> Stage-2 has {len(ids.le.classes_)} attack classes")

    print("[3/3] scoring ...")
    det = ids.predict_detailed(df)
    anom = det["is_anomaly"].mean()
    print(f"\n  anomaly rate (flagged by Stage-1 gate): {anom:.1%}")
    print("  label breakdown:")
    for lbl, n in det["label"].value_counts().items():
        print(f"    {lbl:24s} {n:6d}  ({n/len(det):5.1%})")

    flood = det["label"].str.contains("Flood", case=False).sum()
    print(f"\n  PASS: pipeline ran end-to-end ({flood} flood rows detected)."
          if len(det) else "\n  WARN: no rows scored.")
    print("  (synthetic traffic tests plumbing, not detection accuracy — "
          "replay a real attack pcap for that.)")


if __name__ == "__main__":
    main()
