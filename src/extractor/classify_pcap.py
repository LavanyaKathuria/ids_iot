"""Classify a pcap (or folder of pcaps) with the flat port-agnostic model.

No anomaly gate: every window gets a direct label (Benign or an attack class).
Use this to test real captures from any IP / any port.

    python src/extractor/classify_pcap.py path/to/attack.pcap
    python src/extractor/classify_pcap.py path/to/folder/      # all *.pcap/*.cap
"""
from __future__ import annotations

import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, pandas as pd
import config as C
from fast_transform import build_matrix
from pcap_to_features import extract_features

OUT = os.path.join(C.MODELS_DIR, "pathA")


def load():
    model = joblib.load(os.path.join(OUT, "flat_dtree_portagnostic.joblib"))
    le = joblib.load(os.path.join(OUT, "label_encoder_flat.joblib"))
    feats = json.load(open(os.path.join(OUT, "feature_list_behavioral.json")))
    if hasattr(model, "feature_names_in_"):
        del model.feature_names_in_
    return model, le, feats


def classify(path, model, le, feats):
    df = extract_features(path)
    if len(df) == 0:
        print(f"  {os.path.basename(path)}: no parseable packets"); return
    pred = le.inverse_transform(model.predict(build_matrix(df, feats)))
    vc = pd.Series(pred).value_counts(normalize=True)
    print(f"\n{os.path.basename(path)}  ({len(df)} windows)")
    for k, v in vc.head(6).items():
        print(f"  {k:26s} {v:6.1%}")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python classify_pcap.py <pcap-or-folder>")
    p = sys.argv[1]
    pcaps = ([p] if os.path.isfile(p)
             else sorted(glob.glob(os.path.join(p, "*.pcap")) +
                         glob.glob(os.path.join(p, "*.cap"))))
    if not pcaps:
        sys.exit(f"no pcaps at {p}")
    model, le, feats = load()
    print(f"flat port-agnostic model: {len(le.classes_)} classes, {len(feats)} features")
    for f in pcaps:
        classify(f, model, le, feats)


if __name__ == "__main__":
    main()
