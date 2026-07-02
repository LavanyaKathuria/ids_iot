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

## ⚠️ Capture placement — this makes or breaks the result

**The single biggest cause of bad results is capturing where the sensor can't see
the attack.** The model classifies each 10-packet window by what's *in* it — if the
window is full of background instead of attack, everything misclassifies (floods
come out as scans, scans as nothing, etc.).

### Correct: capture ON the victim (or on a mirror port feeding the monitor)
The recommended command **run on the victim itself** — it receives 100% of the
attack (the packets are addressed to it):
```bash
sudo ip link set eth0 promisc on                     # promiscuous mode
sudo tcpdump -i eth0 "dst host 192.168.0.100" -w captures/<name>.pcap
```
This is what produces clean, correctly-classified captures.

### The problem with a separate monitor on a switch
If the capture runs on a **different machine** (e.g. a separate Ubuntu monitor)
plugged normally into a **switch**, that machine **does not receive the
attacker→victim unicast traffic** — the switch only forwards those packets to the
*victim's* port. The monitor then sees only **broadcast + background**, so the
windows are background, not attack → **misclassification**. This is exactly why an
office capture failed while an on-victim capture of the *same attack* classified
correctly.

To use a separate monitor, it must be **fed** the traffic — one of:
- a switch **SPAN / mirror port** mirroring the victim's / IoT VLAN to the monitor,
- **inline** placement (monitor/Pi in the traffic path), or
- a shared segment (old hub). *(Wi-Fi monitor mode gives 802.11-framed, WPA2-encrypted
  frames — unusable for this IP model.)*
In all cases the capture interface must be in **promiscuous mode**.

### Secondary factor: a victim with no firewall replies
A firewalled Windows victim silently drops the attack → **no replies** → clean,
one-directional capture. A **Raspberry Pi (no firewall by default) replies**
(RST/SYN-ACK/ICMP echo) → the capture is bidirectional, which can muddy the class.
The `dst host <victim>` filter mitigates this by keeping only inbound attack packets.

### Verify the capture point actually sees the attack (30-second check)
On the machine you intend to capture on, **during an attack**:
```bash
sudo tcpdump -i eth0 "dst host 192.168.0.100" -c 20
```
- **packets stream in** → the point sees the attack → captures will classify. ✅
- **nothing / only broadcast** → blind to the attack (no mirror port) → fix placement.

And check a suspect capture is attack-heavy, not background:
```bash
python src/extractor/diagnose_flood.py captures/<name>.pcap
# expect real Rate / syn_count etc. — if it's all near-zero background, the sensor
# wasn't seeing the attack.
```
Decisive test: capture the *same* attack **on the victim** with the `dst host`
filter and classify it — if that's correct but your monitor capture isn't, the
monitor placement (not the model) is the problem.

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
