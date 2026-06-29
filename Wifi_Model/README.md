# WiFi (802.11) Layer — work area

Separate sensor + detection model for **802.11 management/control-frame attacks**
(deauth/disassoc floods, evil-twin / rogue AP, KRACK, Kr00k, PMKID, (re)assoc
floods, beacon/probe abuse) — invisible to the IP-layer pipeline in the parent
project.

## Key decisions (carried from the IP-layer work)
- **Capture:** monitor-mode (radiotap/802.11) on a capable adapter — NOT the
  Ethernet/IP extractor. This is a brand-new capture path + feature set.
- **Dataset:** **AWID3** (2021, WPA2/802.1X, ships PCAPs). Extract 802.11 features
  from its PCAPs with our own pipeline for train/serve consistency.
  - Caveat: AWID3 is enterprise 802.1X; deployment is WPA2-PSK. Deauth/disassoc/
    evil-twin/rogue-AP detection transfers; KRACK/Kr00k/PMKID may not — validate
    with a small self-captured set (aireplay-ng deauth, hostapd evil-twin).
- **Integration contract:** append alerts to the shared sink as
  `ts,label,source` (same schema as `../src/deploy/ids_runner.py` and
  `../src/deploy/suricata_collector.py`) so `../src/deploy/dashboard.py` shows
  them. Architecture = parallel medium-specific sensors (no router).
- **Target:** Raspberry Pi 4 (4 GB); keep lightweight (IP stack already ~1.4 GB).
- Detection may be ML, heuristic/threshold, or hybrid — decide per attack.
- Prevention note: enabling 802.11w / PMF on the AP removes classic deauth.

## Data
Place downloaded AWID3 PCAPs/CSVs under `Wifi_Model/data/` (git-ignored).
