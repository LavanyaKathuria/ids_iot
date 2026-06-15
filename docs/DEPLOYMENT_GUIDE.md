# Raspberry Pi Deployment Guide — CICIoT2023 Two-Stage IDS

A step-by-step guide to run the two-stage intrusion detector on a Raspberry Pi 4
and view alerts on a monitor/computer.

```
            ┌─────────────────────────── Raspberry Pi 4 ───────────────────────────┐
 network →  │ tcpdump  →  pcap_to_features.py  →  predict.py (IDSModel)             │
 (mirror)   │ (capture)   (46 features/flow)      Stage1 gate → Stage2 28-class     │ → alerts.csv
            └────────────────────────────────────────────────────────────────────┬─┘
                                                                                  │
                       Computer / monitor:  Streamlit (or Grafana) dashboard ◄────┘
```

---

## Part A — Set up the Raspberry Pi

### A1. How the Pi sees the traffic (pick one)
- **Recommended — switch mirror / SPAN port (passive):** configure a port on a
  managed switch to mirror your IoT VLAN to the Pi's `eth0`. The Pi sees all IoT
  traffic and can't break the network. Put `eth0` in promiscuous mode.
- **Inline gateway / Wi-Fi AP:** make the Pi the router/AP the IoT devices connect
  through (`eth0` = WAN, `wlan0`/`eth1` = IoT). Sees everything but is in the data
  path (a Pi failure drops the network).
- **Single host test:** just monitor the Pi's own interface — fine for a demo,
  but it only sees its own + broadcast traffic.

### A2. OS + dependencies
Use **Raspberry Pi OS 64-bit**. Match the training Python (3.11) if possible.
```bash
sudo apt update && sudo apt install -y python3-pip tcpdump
pip3 install --user scikit-learn pandas numpy joblib dpkt
# only if you will re-calibrate the gate ON the Pi:
pip3 install --user imbalanced-learn       # (calibration needs only sklearn, this is optional)
```
> Keep `scikit-learn` the **same major version** as training (the models were
> saved with scikit-learn 1.8). A very different version may fail to unpickle.

### A3. Files to copy to the Pi (keep this folder layout)
```
ids/
├── config.py                      ← from src/config.py
├── feature_engineering.py         ← from src/feature_engineering.py
├── fast_transform.py              ← from src/fast_transform.py
├── pcap_to_features.py            ← from src/extractor/pcap_to_features.py
├── predict.py                     ← from src/deploy/predict.py
├── ids_runner.py                  ← NEW, template in A5
└── models/pathA/                  ← copy the whole artifacts/models/pathA/ folder
    ├── pathA_dtree_attack.joblib          (Stage-2, 28 classes — default)
    ├── label_encoder_attack.joblib
    ├── anomaly_gate.joblib                (Stage-1 benign gate)
    └── feature_list.json
```
> `predict.py` and `pcap_to_features.py` expect `config.py`, `fast_transform.py`,
> `feature_engineering.py` to be importable. The simplest fix on the Pi: put them
> all in one folder (as above) and run from there. The model path is
> `models/pathA/` — pass it explicitly: `IDSModel(models_dir="models/pathA")`.

You do **NOT** need on the Pi: the dataset, the pcaps, `artifacts/pcap_features/`,
or any `train_*` / `*_compare` / `*_test` scripts — those are for training only.

### A4. ⚠️ On-site benign calibration (DO THIS FIRST — critical)
The default gate was trained on lab benign and **will false-alarm on your
network** (cross-network benign ≈ 14–52%). Calibrate it on *your* network's
normal traffic to reach ~95%:
```bash
# 1) capture a CLEAN benign baseline (network must be attack-free for ~10-30 min)
sudo tcpdump -i eth0 -w benign_baseline.pcap -c 2000000     # ~2M packets

# 2) calibrate (one-time)
python3 - <<'PY'
from pcap_to_features import extract_features
from predict import IDSModel
df = extract_features("benign_baseline.pcap")
ids = IDSModel(models_dir="models/pathA")
ids.calibrate_benign(df)          # re-fits + saves models/pathA/anomaly_gate.joblib
print("gate calibrated on local benign:", len(df), "flows")
PY
```
> The baseline window must be genuinely clean — a baseline containing attacks
> teaches the gate to accept them.

### A5. Live runner (`ids_runner.py` — create this on the Pi)
tcpdump rotates short pcap files; the runner processes each closed file and
appends anomalies to `alerts.csv`.
```python
# ids_runner.py  — capture-rotate-infer loop
import glob, os, time, subprocess, pandas as pd
from pcap_to_features import extract_features
from predict import IDSModel

CAP_DIR, IFACE, ROTATE_SEC = "/tmp/ids_cap", "eth0", 10
os.makedirs(CAP_DIR, exist_ok=True)
ids = IDSModel(models_dir="models/pathA")

# tcpdump: new file every ROTATE_SEC, keep last 6
subprocess.Popen(["tcpdump","-i",IFACE,"-w",f"{CAP_DIR}/c_%Y%m%d_%H%M%S.pcap",
                  "-G",str(ROTATE_SEC),"-W","6","-Z","root"])
seen=set()
while True:
    files=sorted(glob.glob(f"{CAP_DIR}/c_*.pcap"))[:-1]   # skip the file being written
    for f in files:
        if f in seen: continue
        seen.add(f)
        try:
            df = extract_features(f)
            if len(df)==0: continue
            det = ids.predict_detailed(df)
            alerts = det[det.is_anomaly].copy()
            if len(alerts):
                alerts["ts"]=time.strftime("%Y-%m-%d %H:%M:%S")
                alerts[["ts","label"]].to_csv("alerts.csv", mode="a",
                        header=not os.path.exists("alerts.csv"), index=False)
                print(time.strftime("%H:%M:%S"), "alerts:",
                      alerts.label.value_counts().to_dict())
        except Exception as e:
            print("skip", f, e)
        finally:
            try: os.remove(f)
            except: pass
    time.sleep(2)
```
Run it:
```bash
sudo python3 ids_runner.py        # sudo: tcpdump needs raw-socket access
```

### A6. Run on boot (systemd service)
`/etc/systemd/system/ids.service`:
```ini
[Unit]
Description=CICIoT2023 IDS
After=network-online.target
[Service]
WorkingDirectory=/home/pi/ids
ExecStart=/usr/bin/python3 /home/pi/ids/ids_runner.py
Restart=always
User=root
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now ids.service
journalctl -u ids -f          # live logs
```

---

## Part B — Monitoring on a computer / monitor

The runner writes `alerts.csv`. Pick a viewer:

| Option | Effort | Best for |
|---|---|---|
| **Streamlit dashboard** (recommended) | low (pure Python) | a custom IDS view on a monitor |
| **Grafana + InfluxDB** | medium | polished "wall display" / multiple Pis |
| **`tail -f alerts.csv`** | none | quick demo / SSH terminal |

### B1. Streamlit dashboard (recommended)
Install on the computer/monitor (or the Pi): `pip install streamlit pandas`.
Create `dashboard.py`:
```python
# dashboard.py  — run: streamlit run dashboard.py
import pandas as pd, streamlit as st, time
st.set_page_config(page_title="IoT IDS", layout="wide")
st.title("🛡️ CICIoT2023 IoT IDS — live alerts")
ALERTS = "alerts.csv"        # point at the Pi's alerts.csv (shared/NFS/scp-synced)
while True:
    try: df = pd.read_csv(ALERTS, names=["ts","label"], header=0)
    except Exception: df = pd.DataFrame(columns=["ts","label"])
    c1,c2,c3 = st.columns(3)
    c1.metric("Total alerts", len(df))
    c2.metric("Attack types seen", df.label.nunique() if len(df) else 0)
    c3.metric("Last alert", df.ts.iloc[-1] if len(df) else "—")
    st.subheader("Alerts by attack type")
    if len(df): st.bar_chart(df.label.value_counts())
    st.subheader("Recent alerts")
    st.dataframe(df.tail(50).iloc[::-1], use_container_width=True)
    time.sleep(3); st.rerun()
```
- Run on the monitor machine: `streamlit run dashboard.py` → open the shown URL
  (e.g. `http://localhost:8501`) full-screen on the monitor.
- Get `alerts.csv` to the monitor machine via a shared folder, an NFS/Samba mount
  of the Pi, or a periodic `scp pi@<pi-ip>:/home/pi/ids/alerts.csv .`.
- To run the dashboard **on the Pi** and view from any browser on the LAN:
  `streamlit run dashboard.py --server.address 0.0.0.0` → `http://<pi-ip>:8501`.

### B2. Grafana + InfluxDB (polished alternative)
Have the runner write each alert to InfluxDB instead of CSV (a few lines with
`influxdb-client`), then build a Grafana dashboard (counts by attack type, alert
rate over time, last-seen table) and display it full-screen on the monitor.
Heavier to set up but looks professional and scales to multiple Pis.

---

## Part C — Verify it works
```bash
# replay a known attack pcap through the pipeline (on a PC or the Pi)
python3 - <<'PY'
from pcap_to_features import extract_features
from predict import IDSModel
df = extract_features("some_attack.pcap")          # e.g. a DDoS capture
det = IDSModel(models_dir="models/pathA").predict_detailed(df)
print("anomaly rate:", det.is_anomaly.mean().round(3))
print(det.label.value_counts().head())
PY
```
Expect floods/Mirai → ~98% flagged and labelled correctly; benign → mostly
"BenignTraffic".

---

## Notes & honest limits (see PROJECT_REPORT.md for detail)
- **Latency:** batch flows (the runner does, per rotated file) → ~12 µs/flow.
  Single-flow on a Pi is ~15–24 ms; that's fine because the runner works in
  batches.
- **Strong, trustworthy detections:** floods (UDP/SYN/TCP/HTTP), Mirai, ARP
  spoofing, fragmentation — these generalise across networks.
- **Weak — do not rely on alone:** Web attacks (XSS/SQLi/Backdoor/etc.) and brute
  force often slip the gate; flow features can't see them. Pair with payload
  inspection (e.g. Suricata) if you need those.
- **Re-calibrate the gate** periodically (e.g. weekly, A4) as your network's
  normal traffic changes, or false positives will creep up.
- **Tune sensitivity:** the gate's `contamination` (default 0.05) trades benign
  false-alarms vs attack catch-rate — lower it for fewer false alarms.
```
