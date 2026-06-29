"""Stream each per-attack flow (tests/flows/*.pcap) through the unified pipeline
(ML model + Suricata DPI) and report combined accuracy.

ML runs per flow (in-process). Suricata runs ONCE over all flows in unix-socket
mode (rules loaded once) for speed; alerts attribute back to each flow via the
eve.json `pcap_filename` field. Combined verdict = attack if EITHER engine fires
— mirroring pipeline.classify_flow exactly.
"""
from __future__ import annotations

import collections, glob, json, os, platform, subprocess, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "extractor"),
          os.path.join(ROOT, "src", "deploy")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import config as C  # noqa: E402
from pcap_to_features import extract_features  # noqa: E402
from predict import IDSModel  # noqa: E402
from pipeline import _NOISE, _wsl_path  # noqa: E402

FLOWS = os.path.join(HERE, "flows")
OUT = os.path.join(ROOT, "suri_batch")
LOCAL = os.path.join(ROOT, "src", "deploy", "suricata", "local.rules")
DISTRO = os.environ.get("IDS_WSL_DISTRO", "Ubuntu-24.04")


def suricata_batch(pcaps):
    """Run all pcaps through Suricata once; return {basename: [signatures]}.
    Reuses existing per-flow eve.json if already present (idempotent)."""
    have_all = all(os.path.exists(os.path.join(OUT, os.path.basename(p)[:-5], "eve.json"))
                   for p in pcaps)
    if have_all:
        print("reusing existing Suricata results in", OUT)
    else:
        script = os.path.join(HERE, "suricata_batch.sh")
        on_win = platform.system() == "Windows"
        tr = _wsl_path if on_win else (lambda x: x)
        args = [tr(script), tr(OUT), tr(LOCAL)] + [tr(p) for p in pcaps]
        cmd = (["wsl", "-d", DISTRO, "-u", "root", "-e", "bash"] + args) if on_win \
            else ["bash"] + args
        print("running Suricata (unix-socket, one rule load)...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("  " + r.stdout.strip().replace("\n", "\n  "))
    # each flow's alerts land in OUT/<flowname>/eve.json
    sigs = collections.defaultdict(list)
    for p in pcaps:
        name = os.path.basename(p)[:-5]
        eve = os.path.join(OUT, name, "eve.json")
        if not os.path.exists(eve):
            continue
        with open(eve, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") != "alert":
                    continue
                a = e["alert"]; s = a["signature"]
                if a.get("severity", 3) <= 2 and not any(n in s for n in _NOISE):
                    sigs[name + ".pcap"].append(s)
    return sigs


def main():
    flows = sorted(glob.glob(os.path.join(FLOWS, "*.pcap")))
    if not flows:
        sys.exit("no flows — run: python tests/make_attack_flows.py")
    dpi = suricata_batch(flows)
    ids = IDSModel()

    rows = []
    for f in flows:
        true = os.path.basename(f)[:-5]
        df = extract_features(f)
        if len(df):
            det = ids.predict_detailed(df)
            ml_rate = float(det["is_anomaly"].mean())
            ml_label = (det.loc[det["is_anomaly"], "label"].value_counts().idxmax()
                        if ml_rate > 0 else "BenignTraffic")
        else:
            ml_rate, ml_label = 0.0, "BenignTraffic"
        sigs = dpi.get(os.path.basename(f), [])
        ml_hit = ml_rate >= 0.10                       # >=10% of windows flagged
        dpi_hit = len(sigs) > 0
        detected = ml_hit or dpi_hit
        top_sig = collections.Counter(sigs).most_common(1)[0][0] if sigs else "-"
        rows.append({"true_class": true, "ml_rate": round(ml_rate, 2),
                     "ml_label": ml_label, "dpi_alerts": len(sigs),
                     "detected": detected, "caught_by": ("both" if ml_hit and dpi_hit
                     else "ML" if ml_hit else "DPI" if dpi_hit else "none"),
                     "top_dpi_sig": top_sig[:46]})

    rep = pd.DataFrame(rows).set_index("true_class")
    print("\n" + rep.to_string())

    atk = rep[rep.index != "BenignTraffic"]
    web = atk[atk.index.isin(["SqlInjection", "XSS", "CommandInjection",
                              "BrowserHijacking", "Uploading_Attack", "Backdoor_Malware"])]
    benign = rep[rep.index == "BenignTraffic"]
    print("\n=== combined-pipeline detection ===")
    print(f"  all attacks  : {atk['detected'].sum()}/{len(atk)} detected "
          f"({atk['detected'].mean():.0%})")
    print(f"  WEB attacks  : {web['detected'].sum()}/{len(web)} detected "
          f"({web['detected'].mean():.0%})  <- ML alone could not")
    if len(benign):
        b = benign.iloc[0]
        print(f"  benign       : {'PASS (not flagged)' if not b['detected'] else 'FALSE POSITIVE'}")


if __name__ == "__main__":
    main()
