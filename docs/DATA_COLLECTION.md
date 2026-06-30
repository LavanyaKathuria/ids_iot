# Data Collection & Running pcaps Through the IDS

How the own-network attack captures (in `captures/train_manifest.csv`) were
generated, the capture/listening commands, and how a captured pcap is fed into
`ids_runner.py`.

## Setup — two machines on the same network

- **Target + sensor:** the machine running the IDS (example IP `192.168.0.100`).
- **Attacker:** a second laptop on the same LAN running `hping3` / `nmap`
  (Linux, or WSL with **mirrored networking** so it shares the host's real IP —
  default WSL2 NAT throttles floods and can't do layer-2 attacks).

---

## Step 1 — Listen / capture on the target

Capture only the attacker→target traffic into a pcap.

**Linux / Raspberry Pi (tcpdump):**
```bash
sudo tcpdump -i eth0 "dst host 192.168.0.100" -w captures/<name>.pcap
#   stop with Ctrl+C.  add  -c 200000  to stop after N packets.
```

**Windows (Wireshark's dumpcap — capture by interface NAME, not number):**
```powershell
& "C:\Program Files\Wireshark\dumpcap.exe" -i Wi-Fi -f "dst host 192.168.0.100" `
  -w "C:\...\captures\<name>.pcap"
```
> The `dst host` filter records only inbound attack traffic (excludes the target's
> own replies → cleaner windows). Npcap/dumpcap captures at the adapter, so a
> firewall silently dropping the packets does **not** stop the capture.

---

## Step 2 — Generate the attack (on the attacker)

Start the capture first, run the attack ~20–30 s, then `Ctrl+C` both. These are
the exact commands behind the manifest captures (target = `192.168.0.100`):

| class (manifest label) | command |
|---|---|
| `SYN_Flood` | `sudo hping3 --flood -S -p 53 192.168.0.100` (any port — model is port-agnostic) |
| `UDP_Flood` | `sudo hping3 --flood --udp -p 53 192.168.0.100` (vary `-d 100/300/500` for size) |
| `TCP_Flood` | `sudo hping3 --flood -A -p 80 192.168.0.100` |
| `DDoS-PSHACK_Flood` | `sudo hping3 --flood -PA -p 80 192.168.0.100` |
| `DDoS-RSTFINFlood` | `sudo hping3 --flood -RF -p 80 192.168.0.100` |
| `DDoS-ICMP_Flood` | `sudo hping3 --flood --icmp 192.168.0.100` |
| `DDoS-Fragmentation` | `sudo hping3 --flood -A -f -d 400 -p 80 192.168.0.100` (also `--udp -f`, `--icmp -f`) |
| `Recon-Scanning` | `sudo nmap -sS -Pn -T4 -p- 192.168.0.100` |

Notes:
- **UDP needs size variety** — capture small *and* large payloads (`-d 100`, `-d 500`)
  or held-out large UDP floods get missed.
- nmap scans are slow when the target silently drops — add `-T4 --max-retries 0`.
- For a held-out test capture, repeat with slightly different params (different
  port/size) and a `2`-suffixed filename.

---

## Step 3 — Add a capture to training (optional)

To make a capture part of the model, add a row to `captures/train_manifest.csv`:
```csv
file,label
my_udp_flood.pcap,UDP_Flood
```
then retrain (it auto-balances per class and prints a regression check):
```bash
python src/extractor/train_flat_portagnostic.py
```

---

## Step 4 — Run a captured pcap through the IDS

**Quick label check (any pcap):**
```bash
python src/extractor/classify_pcap.py captures/my_capture.pcap
```

**Through the live runner — offline replay** (no live traffic needed). The pcap is
extracted → scored → anomalies appended to `alerts.csv` (`ts,label,source`):
```bash
python src/deploy/ids_runner.py --flat --replay captures/my_capture.pcap \
    --models artifacts/models/pathA
# or a whole folder of pcaps:
python src/deploy/ids_runner.py --flat --replay captures/ --models artifacts/models/pathA
```

**Live capture** (the runner rotates short pcaps with tcpdump and scores each):
```bash
sudo python3 src/deploy/ids_runner.py --flat --iface eth0 --models artifacts/models/pathA
```
`--flat` selects the deployed flat model. Drop it to use the older two-stage model
(needs the calibrated benign gate).

---

## Step 5 — View alerts on the dashboard

The runner appends each alert to `alerts.csv`; the Streamlit dashboard shows them
live (counts by attack type, timeline, recent-alerts table; auto-refreshes).
```bash
pip install streamlit pandas
# point it at the runner's alerts file and launch:
IDS_ALERTS=alerts.csv streamlit run src/deploy/dashboard.py
#   open http://localhost:8501
# to view from another machine on the LAN (e.g. dashboard on the Pi):
IDS_ALERTS=alerts.csv streamlit run src/deploy/dashboard.py --server.address 0.0.0.0
#   → http://<host-ip>:8501
```
Env vars: `IDS_ALERTS` (alerts CSV path, default `./alerts.csv`), `IDS_REFRESH`
(seconds between refreshes, default 3).

---

## Flow summary
```
attacker (hping3/nmap)  →  [LAN]  →  target NIC
                                       │  tcpdump / dumpcap  (dst host filter)
                                       ▼
                                  captures/*.pcap
                          ┌────────────┴─────────────┐
              classify_pcap.py             ids_runner.py --flat --replay
              (one-off label)              (alerts.csv → dashboard)
```
