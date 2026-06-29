"""UNIFIED single-flow IDS pipeline: ML model + Suricata DPI in one place.

Takes ONE flow at a time (a pcap of one flow/session from any source) and runs
BOTH detectors, then combines them:

    Stage 1+2 (ML)  : extract 46 features -> benign gate -> 26-class classifier
                      -> good at floods / scans / Mirai / spoofing (behavioral)
    Suricata (DPI)  : signature + payload inspection
                      -> good at Web/payload attacks the ML is blind to (SQLi/XSS/...)

Combined verdict = attack if EITHER engine fires. The label prefers the DPI
signature when Suricata fires (it names the exact payload attack), else the ML
class.

    from pipeline import classify_flow
    v = classify_flow("flow.pcap")           # -> dict (ml + dpi + combined)
    python pipeline.py flow.pcap [--rules local|full]

Portable: on Linux (the Pi) it calls `suricata` directly; on Windows it shells
into WSL. In production Suricata runs LIVE (rules loaded once); this per-call
invocation is for offline/one-shot scoring.
"""
from __future__ import annotations

import argparse, json, os, platform, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(os.path.dirname(HERE)),            # src/
          os.path.join(os.path.dirname(HERE), "extractor"),
          HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from pcap_to_features import extract_features  # noqa: E402
from predict import IDSModel  # noqa: E402

WSL_DISTRO = os.environ.get("IDS_WSL_DISTRO", "Ubuntu-24.04")
LOCAL_RULES = os.path.join(HERE, "suricata", "local.rules")
# decoder/stream events that are engine noise, not attacks
_NOISE = ("invalid checksum", "STREAM", "Applayer", "unable to match",
          "gzip decompression", "FRAG", "decompression failed")


def _wsl_path(p: str) -> str:
    p = os.path.abspath(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def run_suricata(pcap: str, rules: str = "full") -> list[str]:
    """Run Suricata on one pcap, return the list of real (non-noise) signatures."""
    outdir = tempfile.mkdtemp(prefix="ids_suri_")
    on_win = platform.system() == "Windows"
    rp, op, lr = ((_wsl_path(pcap), _wsl_path(outdir), _wsl_path(LOCAL_RULES))
                  if on_win else (pcap, outdir, LOCAL_RULES))
    rule_args = (["-S", lr] if rules == "local"            # only local.rules (fast)
                 else ["-s", lr])                          # ET ruleset + local.rules
    sur = (["suricata", "-r", rp, "-l", op, "-k", "none",
            "--set", "vars.address-groups.EXTERNAL_NET=any"] + rule_args)
    cmd = (["wsl", "-d", WSL_DISTRO, "-u", "root", "-e"] + sur) if on_win else sur
    subprocess.run(cmd, capture_output=True, text=True)
    eve = os.path.join(outdir, "eve.json")
    sigs = []
    if os.path.exists(eve):
        with open(eve, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") == "alert":
                    a = e["alert"]; s = a["signature"]
                    # keep high/medium severity attack sigs; drop sev-3 info + decoder/stream noise
                    if a.get("severity", 3) <= 2 and not any(n in s for n in _NOISE):
                        sigs.append(s)
    return sigs


def classify_flow(pcap: str, ids: IDSModel | None = None, rules: str = "full") -> dict:
    ids = ids or IDSModel()
    # ---- ML branch ----
    df = extract_features(pcap)
    ml_label, ml_detected = "BenignTraffic", False
    if len(df):
        det = ids.predict_detailed(df)
        anom = det[det["is_anomaly"]]
        if len(anom):
            ml_detected = True
            ml_label = anom["label"].value_counts().idxmax()
    # ---- DPI branch ----
    sigs = run_suricata(pcap, rules=rules)
    dpi_detected = len(sigs) > 0
    # ---- combine ----
    detected = ml_detected or dpi_detected
    if dpi_detected:
        final = "DPI: " + max(set(sigs), key=sigs.count)
    elif ml_detected:
        final = ml_label
    else:
        final = "BenignTraffic"
    return {"flows": len(df), "ml_detected": ml_detected, "ml_label": ml_label,
            "dpi_detected": dpi_detected, "dpi_sigs": sorted(set(sigs))[:5],
            "detected": detected, "final_label": final}


def main():
    ap = argparse.ArgumentParser(description="Unified ML+Suricata single-flow IDS")
    ap.add_argument("pcap")
    ap.add_argument("--rules", choices=["local", "full"], default="full")
    args = ap.parse_args()
    v = classify_flow(args.pcap, rules=args.rules)
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
