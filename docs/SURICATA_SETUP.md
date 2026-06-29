# Phase 1 — Suricata DPI alongside the ML model

Adds a signature/DPI engine next to the flow-ML model so the **payload attacks
the ML can't see** (SQLi, XSS, command injection, malware droppers, C2) are
covered. Both engines watch the **same traffic** and write to the **same
`alerts.csv`**, so they share one dashboard.

```
mirror tap ─┬─► Suricata (rules) ─► eve.json ─► suricata_collector.py ─┐
            └─► ids_runner.py (ML) ────────────────────────────────────┤─► alerts.csv ─► dashboard.py
```

## Does it need data?
**No training data.** Unlike the ML model, Suricata is rule-based. It needs:
1. **Rules** — the free, maintained **ET Open** ruleset, fetched by `suricata-update`
   (tens of thousands of rules: SQLi/XSS/CmdInj, malware, CVEs). Plus optional
   local rules (`src/deploy/suricata/local.rules`).
2. **Config** — your network range (`HOME_NET`) and capture interface.

That's exactly why Phase 1 solves what adding ML data couldn't: it covers the
payload classes with **zero data collection or labeling**.

## Install (Raspberry Pi OS)
```bash
sudo apt update && sudo apt install -y suricata
sudo suricata-update                       # pull ET Open ruleset
sudo suricata-update list-sources          # (optional) more free feeds
```

## Configure  (`/etc/suricata/suricata.yaml`)
```yaml
vars:
  address-groups:
    HOME_NET: "[192.168.1.0/24]"           # <-- your IoT subnet
af-packet:
  - interface: eth0                        # <-- your mirror/capture interface
# eve.json (alerts) is enabled by default under outputs: -> eve-log:
```
Add the local rules:
```bash
sudo cp src/deploy/suricata/local.rules /var/lib/suricata/rules/
# in suricata.yaml under `rule-files:` add a line:   - local.rules
sudo suricata -T -c /etc/suricata/suricata.yaml       # test config + rules
```

## Run
```bash
# IDS (passive, reads a copy of the traffic):
sudo suricata -c /etc/suricata/suricata.yaml -i eth0
#   ...or enable the service:  sudo systemctl enable --now suricata

# bridge its alerts onto your dashboard (run beside ids_runner.py):
python3 src/deploy/suricata_collector.py --eve /var/log/suricata/eve.json --alerts alerts.csv
```
Now `dashboard.py` shows both ML alerts and `DPI: ...` alerts together.

## Verify it works
```bash
# trigger a local rule from another host on the network:
curl "http://<an-iot-device>/?id=1%20union%20select%20pass%20from%20users"
# watch it appear:
tail -f /var/log/suricata/fast.log          # human-readable
# and on the dashboard as:  DPI: LOCAL SQLi UNION SELECT in URI
```

## Encrypted traffic — Suricata is NOT idle
If a flow is TLS-encrypted, payload-content rules can't match the body, but
Suricata still logs and alerts on:
- **TLS metadata**: JA3/JA3S fingerprint, SNI, certificate issuer/validity
  (catches malware C2 and bad domains while still encrypted),
- **DNS / QUIC / DHCP** (usually cleartext),
- **anomalies / policy** (TLS on odd ports, self-signed certs).
Meanwhile the ML model keeps detecting volumetric/behavioral attacks regardless
of encryption. Only *encrypted payload exploits* remain out of reach (that would
need TLS interception — generally impractical for IoT).

## Passive (IDS) vs inline (IPS)
Start **passive** (above) — Suricata only alerts. Once rules are tuned and false
positives are low, you can run it **inline** (`-q 0` with NFQUEUE) so it can
*drop* malicious packets. Keep auto-blocking gated until you trust it.

## What each engine owns
| Engine | Detects | Needs |
|---|---|---|
| ML flow model | floods, fragmentation, Mirai, scans, ARP/DNS spoofing | trained model (have it) |
| Suricata DPI | SQLi, XSS, CmdInj, malware, C2, CVEs, TLS-metadata | rules (free), no training data |

## Phase 1 validation (tested 2026-06-22) — two config gotchas
Validated on Suricata 7.0.3 (Ubuntu 24.04 / WSL). Results that matter for setup:

- **Synthetic web attacks** (`tests/make_web_attack_pcap.py`): SQLi/XSS/CmdInjection
  → all fired immediately. Suricata covers the ML's weak Web classes. ✅
- **Real CICIDS2017 Tuesday (FTP/SSH Patator), 11 GB** took 3 tries; the fixes are
  the lesson:
  1. **`-k none` for offline pcaps.** Dataset captures use NIC checksum offload, so
     every packet has a "bad" checksum. With validation ON, 99.6% of alerts were
     `SURICATA TCPv4 invalid checksum` noise *and* stream reassembly was skipped →
     0 real detections. (Live traffic from a real NIC does NOT need this.)
  2. **`HOME_NET` must match the topology.** CICIDS2017's attacker `172.16.0.1` is a
     private IP inside the default `HOME_NET`, so ET's `EXTERNAL_NET->HOME_NET` brute
     rules never matched. Narrowing `HOME_NET` to the victim subnet
     (`--set vars.address-groups.HOME_NET="[192.168.10.0/24]"`) made the attacker
     external → **2,372 `ET SCAN Potential FTP Brute-Force` alerts** fired.
- **FTP (plaintext) brute force → detected (2,372). SSH (encrypted) → ~30 vague
  "scan" hits, no real detection.** Encrypted brute force is behavioral, not a
  payload signature → leave it to the ML/rate layer (or host tools like fail2ban).

Takeaway: on a live deployment set `HOME_NET` to your real IoT subnet, skip
`-k none`, and tune/trim the ruleset (next section's note) to fit the Pi.

## Trimming the ruleset for a low-memory Pi (coverage kept)
The full ET Open (~50k rules) is the main RAM cost. You can trim it **without
losing any attack coverage** because the categories that detect your attacks are
separate from the noisy/irrelevant ones. `src/deploy/suricata/disable.conf` drops
whole noise categories by their `ET <CATEGORY>` prefix:

```bash
sudo cp src/deploy/suricata/disable.conf /etc/suricata/disable.conf
sudo suricata-update          # re-applies, now with the noise disabled
```

Verified result (tested 2026-06-22): **50,770 → 34,532 enabled rules (-32%)**.
- **Kept (still enabled):** WEB_SERVER, WEB_CLIENT, WEB_SPECIFIC_APPS, GPL SQL,
  SHELLCODE, EXPLOIT, EXPLOIT_KIT, MALWARE, MOBILE_MALWARE, SCAN, DOS, SCADA, JA3,
  ATTACK_RESPONSE — i.e. all SQLi/XSS/CmdInj/malware/C2/scan/brute coverage.
- **Dropped to ~0:** INFO, HUNTING, DYN_DNS, PHISHING, ADWARE_PUP, USER_AGENTS,
  TOR, GAMES, P2P, FILE_SHARING, GPL NETBIOS — none of which detect your classes,
  and which produced most of the false positives.

Need it even smaller for a 4 GB Pi? The bulk is MALWARE (~18k). Trim it last and
carefully (that's where you start losing coverage), and lower the `*-memcap`
settings in suricata.yaml (`stream`, `flow`, `stream.reassembly`).
