"""Live IDS runner for the Raspberry Pi (two-stage: benign gate -> attack type).

Two modes:
  LIVE    (default): tcpdump rotates short pcaps; each closed file is extracted,
                     scored, and anomalies appended to the alerts CSV.
  REPLAY  (--replay PATH): process an existing pcap file or a folder of pcaps —
                     use this to TEST the whole pipeline with no live traffic
                     (e.g. on the synthetic capture from tests/make_test_pcap.py).

Examples
  sudo python3 ids_runner.py --iface eth0 --models models/pathA
  python3 ids_runner.py --replay tests/sample_traffic.pcap --models artifacts/models/pathA
  python3 ids_runner.py --replay /captures/        # process every *.pcap in a folder
"""
from __future__ import annotations

import argparse, glob, os, subprocess, sys, time

# --- make the shared modules importable from the repo OR a flattened Pi folder ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "extractor"),
          os.path.join(ROOT, "src", "deploy")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd                                    # noqa: E402
from pcap_to_features import extract_features          # noqa: E402
from predict import IDSModel, FlatIDS                  # noqa: E402


def score_pcap(ids, pcap, alerts_path, max_packets=None):
    """Extract one pcap, score it, append anomalies to alerts CSV. Returns dict."""
    df = extract_features(pcap, max_packets=max_packets)
    if len(df) == 0:
        return {}
    det = ids.predict_detailed(df)
    alerts = det[det["is_anomaly"]].copy()
    if len(alerts):
        alerts["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        alerts["source"] = os.path.basename(pcap)
        alerts[["ts", "label", "source"]].to_csv(
            alerts_path, mode="a", header=not os.path.exists(alerts_path), index=False)
    return alerts["label"].value_counts().to_dict()


def run_replay(ids, path, alerts_path):
    pcaps = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*.pcap")) +
                         glob.glob(os.path.join(path, "*.cap"))))
    if not pcaps:
        print(f"no pcaps found at {path}"); return
    print(f"REPLAY: {len(pcaps)} pcap(s)")
    for f in pcaps:
        counts = score_pcap(ids, f, alerts_path)
        print(f"  {os.path.basename(f):30s} alerts: {counts or 'none'}")
    print(f"\nalerts -> {alerts_path}")


def run_live(ids, iface, cap_dir, rotate, keep, alerts_path):
    os.makedirs(cap_dir, exist_ok=True)
    print(f"LIVE on {iface}: rotating every {rotate}s, writing to {cap_dir}")
    proc = subprocess.Popen(
        ["tcpdump", "-i", iface, "-w", os.path.join(cap_dir, "c_%Y%m%d_%H%M%S.pcap"),
         "-G", str(rotate), "-W", str(keep), "-Z", "root"])
    seen = set()
    try:
        while True:
            files = sorted(glob.glob(os.path.join(cap_dir, "c_*.pcap")))[:-1]  # skip current
            for f in files:
                if f in seen:
                    continue
                seen.add(f)
                try:
                    counts = score_pcap(ids, f, alerts_path)
                    if counts:
                        print(time.strftime("%H:%M:%S"), "alerts:", counts)
                except Exception as e:
                    print("skip", os.path.basename(f), e)
                finally:
                    try: os.remove(f)
                    except OSError: pass
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        proc.terminate()


def main():
    ap = argparse.ArgumentParser(description="Two-stage IDS runner")
    ap.add_argument("--iface", default="eth0", help="capture interface (live mode)")
    ap.add_argument("--models", default=None, help="path to models/pathA (default: repo location)")
    ap.add_argument("--replay", default=None, help="pcap file or folder -> offline test mode")
    ap.add_argument("--alerts", default="alerts.csv")
    ap.add_argument("--rotate", type=int, default=10, help="seconds per pcap (live)")
    ap.add_argument("--keep", type=int, default=6, help="rotated files to keep (live)")
    ap.add_argument("--cap-dir", default="/tmp/ids_cap")
    ap.add_argument("--flat", action="store_true",
                    help="use the flat port-agnostic model (no benign gate, no calibration)")
    args = ap.parse_args()

    ids = FlatIDS(args.models) if args.flat else IDSModel(models_dir=args.models)
    print(f"loaded {'FLAT' if args.flat else 'two-stage'} IDS "
          f"({len(ids.le.classes_)} classes)")
    if args.replay:
        run_replay(ids, args.replay, args.alerts)
    else:
        run_live(ids, args.iface, args.cap_dir, args.rotate, args.keep, args.alerts)


if __name__ == "__main__":
    main()
