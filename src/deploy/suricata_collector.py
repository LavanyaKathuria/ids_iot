"""Bridge Suricata alerts into the same alerts.csv the ML runner + dashboard use.

Suricata writes structured events to eve.json (one JSON object per line). This
tails that file, keeps only `event_type == "alert"`, and appends them to
alerts.csv in the SAME ts,label,source schema as ids_runner.py — so DPI alerts
and ML alerts show up together on dashboard.py with no dashboard changes.

Suricata needs RULES, not training data: install it, run `suricata-update` to
pull the ET Open ruleset, point it at your interface, and it logs to eve.json.
See docs/SURICATA_SETUP.md.

    # live (on the Pi, alongside ids_runner.py):
    python3 suricata_collector.py --eve /var/log/suricata/eve.json --alerts alerts.csv
    # one-shot (re-process an existing eve.json):
    python3 suricata_collector.py --once --eve eve.json --alerts alerts.csv
"""
from __future__ import annotations

import argparse, csv, json, os, time
from datetime import datetime


def parse_ts(s: str) -> str:
    """eve ISO8601 (with tz) -> the runner's '%Y-%m-%d %H:%M:%S' format."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s[:19].replace("T", " ")


def to_row(ev: dict) -> dict:
    a = ev.get("alert", {})
    sig = a.get("signature", "Suricata alert")
    sev = a.get("severity", "")
    src, dst = ev.get("src_ip", "?"), ev.get("dest_ip", "?")
    return {"ts": parse_ts(ev.get("timestamp", "")),
            "label": f"DPI: {sig}" + (f" (sev {sev})" if sev else ""),
            "source": f"{src}->{dst}"}


def process_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None
    return to_row(ev) if ev.get("event_type") == "alert" else None


def emit(rows, alerts_path):
    if not rows:
        return
    new = not os.path.exists(alerts_path)
    with open(alerts_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "label", "source"])
        if new:
            w.writeheader()
        w.writerows(rows)


def run_once(eve, alerts):
    with open(eve, encoding="utf-8", errors="replace") as fh:
        rows = [r for r in (process_line(ln) for ln in fh) if r]
    emit(rows, alerts)
    print(f"wrote {len(rows)} alert(s) -> {alerts}")


def run_follow(eve, alerts, poll=1.0):
    print(f"following {eve} -> {alerts} (Ctrl-C to stop)")
    while not os.path.exists(eve):
        time.sleep(poll)
    fh = open(eve, encoding="utf-8", errors="replace")
    fh.seek(0, os.SEEK_END)                       # only new events
    inode = os.fstat(fh.fileno()).st_ino
    n = 0
    try:
        while True:
            line = fh.readline()
            if line:
                r = process_line(line)
                if r:
                    emit([r], alerts); n += 1
                    print(time.strftime("%H:%M:%S"), "DPI alert:", r["label"][:70])
                continue
            time.sleep(poll)
            try:                                  # handle log rotation
                if os.stat(eve).st_ino != inode:
                    fh.close(); fh = open(eve, encoding="utf-8", errors="replace")
                    inode = os.fstat(fh.fileno()).st_ino
            except FileNotFoundError:
                pass
    except KeyboardInterrupt:
        print(f"\nstopped ({n} alert(s) forwarded)")


def main():
    ap = argparse.ArgumentParser(description="Suricata eve.json -> alerts.csv bridge")
    ap.add_argument("--eve", default="/var/log/suricata/eve.json")
    ap.add_argument("--alerts", default="alerts.csv")
    ap.add_argument("--once", action="store_true", help="process existing file and exit")
    args = ap.parse_args()
    (run_once if args.once else run_follow)(args.eve, args.alerts)


if __name__ == "__main__":
    main()
