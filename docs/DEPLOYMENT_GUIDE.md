# Raspberry Pi Deployment Guide — CICIoT2023 IDS (flat model)

Run the **flat, port-agnostic** intrusion detector on a Raspberry Pi 4 and view
alerts on a monitor. This is the current deployable model — a single DecisionTree,
**no anomaly gate, no on-site calibration required**. (Suricata DPI is **not** part
of this pipeline; it's optional/parked — see the note at the end.)

```
            ┌───────────────────────── Raspberry Pi 4 ─────────────────────────┐
 network →  │ tcpdump → pcap_to_features.py → predict.py (FlatIDS)              │
 (mirror)   │ (capture)  (62 behavioural feats)  24-class + Benign DecisionTree │ → alerts.csv
            └──────────────────────────────────────────────────────────────────┴─┐
                                                                                  │
                     Computer / monitor:  Streamlit dashboard ◄───────────────────┘
```

See also [`MODEL_OVERVIEW.md`](MODEL_OVERVIEW.md) (data/features/algorithm) and
[`DATA_COLLECTION.md`](DATA_COLLECTION.md) (capturing pcaps, running them through
the runner).

---

## Part A — Set up the Raspberry Pi

### A1. How the Pi sees the traffic (pick one)
- **Recommended — switch mirror / SPAN port (passive):** mirror your IoT VLAN to
  the Pi's `eth0`; the Pi sees all IoT traffic and can't break the network. Put
  `eth0` in promiscuous mode.
- **Inline gateway / Wi-Fi AP:** the Pi routes the IoT devices. Sees everything but
  a Pi failure drops the network.
- **Single host test:** monitor the Pi's own interface — fine for a demo.

### A2. OS + dependencies
Use **Raspberry Pi OS 64-bit**; match the training Python (3.11) if possible.
```bash
sudo apt update && sudo apt install -y python3-pip tcpdump
pip3 install --user -r requirements.txt
```
> `scikit-learn` is pinned to **1.8.0** — the `.joblib` models were pickled with
> it; a different version may fail to unpickle. Keep the pin.

### A3. Files to copy to the Pi (keep this layout)
```
ids/
├── config.py                      ← from src/config.py
├── feature_engineering.py         ← from src/feature_engineering.py
├── fast_transform.py              ← from src/fast_transform.py
├── pcap_to_features.py            ← from src/extractor/pcap_to_features.py
├── predict.py                     ← from src/deploy/predict.py
├── ids_runner.py                  ← from src/deploy/ids_runner.py
└── models/pathA/
    ├── flat_dtree_portagnostic.joblib    (the deployed model, 24 classes + Benign)
    ├── label_encoder_flat.joblib
    └── feature_list_behavioral.json
```
You do **NOT** need on the Pi: the dataset, the pcaps, `artifacts/pcap_features/`,
or any `train_*` / diagnostic scripts — those are for training only.

> **No calibration step.** Unlike the older two-stage model, the flat model has a
> trained Benign class and runs as-is. (Trade-off: a supervised Benign class doesn't
> generalise across networks as well as a calibrated gate — see "limits" below.)

### A4. Live runner (`ids_runner.py`)
Runs tcpdump to rotate short pcaps, then extracts → scores each closed file and
appends anomalies to `alerts.csv` (`ts,label,source`). **Use `--flat`** for this model:
```bash
# LIVE capture (sudo: tcpdump needs raw sockets)
sudo python3 ids_runner.py --flat --iface eth0 --models models/pathA

# OFFLINE test — replay a pcap or a folder (no live traffic needed)
python3 ids_runner.py --flat --replay some_capture.pcap --models models/pathA
python3 ids_runner.py --flat --replay /captures/        # every *.pcap in a folder
```
Options: `--rotate` (s per pcap, default 10), `--keep`, `--alerts`, `--cap-dir`.
(Drop `--flat` to use the older two-stage model + calibrated gate instead.)

### A5. Run on boot (systemd)
`/etc/systemd/system/ids.service`:
```ini
[Unit]
Description=CICIoT2023 IDS (flat)
After=network-online.target
[Service]
WorkingDirectory=/home/pi/ids
ExecStart=/usr/bin/python3 /home/pi/ids/ids_runner.py --flat --iface eth0 --models models/pathA
Restart=always
User=root
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now ids.service
journalctl -u ids -f
```

---

## Part B — Monitoring on a computer / monitor

The runner writes `alerts.csv`. Viewer options:

| Option | Effort | Best for |
|---|---|---|
| **Streamlit dashboard** (recommended) | low | a custom IDS view on a monitor |
| `tail -f alerts.csv` | none | quick demo / SSH |

### Streamlit dashboard (`src/deploy/dashboard.py`)
`pip install streamlit pandas`, then:
```bash
IDS_ALERTS=/home/pi/ids/alerts.csv streamlit run dashboard.py
#   on the Pi, view from the LAN:  streamlit run dashboard.py --server.address 0.0.0.0
#   → http://<pi-ip>:8501
```
Env: `IDS_ALERTS` (alerts path), `IDS_REFRESH` (seconds, default 3). If the
dashboard runs on another machine, sync `alerts.csv` over (NFS/Samba/`scp`).

---

## Part C — Verify it works
```bash
python3 src/extractor/classify_pcap.py some_attack.pcap      # one-off label check
# or through the runner:
python3 ids_runner.py --flat --replay some_attack.pcap --models models/pathA
```
Expect floods/Mirai → ~99% flagged and labelled correctly; benign → mostly
`BenignTraffic`.

---

## Notes & honest limits (see [`MODEL_OVERVIEW.md`](MODEL_OVERVIEW.md) / PROJECT_REPORT.md)
- **Strong, validated:** SYN/UDP/TCP/PSH-ACK/RST-FIN/ICMP floods (99–100% held-out,
  on **any port**), Mirai, ARP spoofing.
- **Moderate:** `DDoS-Fragmentation` (~60% cross-capture — detected as an attack,
  sometimes mislabelled).
- **Weak — don't rely on alone:** Web attacks (XSS/SQLi/Backdoor/etc.), brute force,
  DNS spoofing — flow features can't see payloads. A signature/DPI engine (Suricata)
  is the right tool for these.
- **Benign on a new network:** the supervised Benign class may over-flag unfamiliar
  devices. If that's a problem, switch to the two-stage model with an on-site
  calibrated gate (drop `--flat`).

### Optional / parked: Suricata DPI
A Suricata layer (for the payload/web attacks above) is built but **not part of this
pipeline**. To add it later, see [`SURICATA_SETUP.md`](SURICATA_SETUP.md) — it writes
to the same `alerts.csv` so the dashboard shows both.
