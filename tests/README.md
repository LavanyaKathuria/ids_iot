# Testing the IDS — with or without live traffic

How to validate the two-stage IDS end-to-end when you have **no IoT network
traffic** to capture (e.g. before the Pi is wired to a mirror port).

## 1. No live traffic — synthetic capture (self-contained)

Generate a small pcap (benign mix + UDP/SYN/ICMP flood bursts) and run the full
deployment pipeline on it. Uses `dpkt` only (already a project dependency) — no
network, no extra installs.

```bash
python make_test_pcap.py      # -> tests/sample_traffic.pcap  (~1000 packets, ~100 KB)
python test_pipeline.py       # pcap -> 46 features -> gate -> 28-class classifier
```

Expected: ~100 flow rows, the gate flags ~90%+ as anomalous, and the flood
bursts come back labelled `UDP_Flood`, `SYN_Flood`, `DDoS-ICMP_Flood`.

> ⚠️ This is a **plumbing / smoke test** — it proves the chain runs and produces
> sane labels. The packets are crafted, not real device traffic, so it does
> **not** measure detection accuracy (and the lab-calibrated gate may mislabel
> the synthetic "benign" block). For accuracy, use a real pcap (section 2).

Scale it up if you want more rows: `python make_test_pcap.py --scale 5`.

## 2. Real traffic — replay an attack pcap

Point the same pipeline at any real capture (e.g. a CICIoT2023 attack pcap):

```bash
python test_pipeline.py /path/to/real_attack.pcap --models ../artifacts/models/pathA
```

Floods / Mirai should come back ~98% flagged and correctly labelled. To see it
go through the **runner** exactly as it will on the Pi (writing `alerts.csv`):

```bash
python ../src/deploy/ids_runner.py --replay /path/to/real_attack.pcap --alerts alerts.csv
```

## 3. View the alerts on the dashboard

After either step produced an `alerts.csv`, launch the monitor view:

```bash
cd ../src/deploy
IDS_ALERTS=../../tests/alerts.csv streamlit run dashboard.py
# open http://localhost:8501
```

## Files

| File | Purpose |
|---|---|
| `make_test_pcap.py` | crafts `sample_traffic.pcap` with dpkt (no live traffic) |
| `test_pipeline.py`  | one-shot end-to-end test: pcap → features → IDS → labels |
| `sample_traffic.pcap` | generated capture (git-ignored; rebuild with the script) |
