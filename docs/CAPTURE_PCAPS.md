# Capturing pcaps from the office network

How to record traffic into `.pcap` files for this project. Two kinds:

- **IP traffic** (for the main model) — uses the normal NIC.
- **WiFi 802.11 traffic** (for the WiFi layer) — needs a monitor-mode adapter.

Capture only; this guide does **not** run detection.

---

## A. IP traffic (Ethernet / Wi-Fi as normal client)

No special hardware — the normal network interface sees the IP traffic in its path.

```bash
# list interfaces
ip link

# capture to a file (Ctrl+C to stop)
sudo tcpdump -i eth0 -w office_ip_$(date +%F).pcap

# capture a fixed amount instead of Ctrl+C:
sudo tcpdump -i eth0 -c 200000 -w office_ip.pcap        # stop after 200k packets
sudo timeout 1800 tcpdump -i eth0 -w office_ip.pcap     # stop after 30 min
```

Useful options:
- `-i any` — capture on all interfaces.
- `-s 0` — full packet (default on modern tcpdump; add it if payloads look cut off).
- Filter to one host/port: `sudo tcpdump -i eth0 host 192.168.1.50 -w one_device.pcap`

**To see ALL devices' traffic (not just this machine's):** plug the Pi/laptop into a
**mirror / SPAN port** on the switch, or capture on the device that routes the IoT
subnet. A normal switch port only shows broadcast + this machine's own traffic.

Windows alternative: install **Wireshark** (includes Npcap) → pick the interface →
red record button → **File ▸ Save As ▸ .pcap**. Or `dumpcap -i 1 -w office.pcap`.

---

## B. WiFi 802.11 traffic (management/control frames)

Needs a **monitor-mode USB adapter** (you already have one — your existing
`Wifi_Model/office_pcap/` captures came from it). The Pi's onboard WiFi and normal
client mode **cannot** capture other devices' 802.11 frames.

```bash
# 1. put the adapter into monitor mode  -> creates e.g. wlan1mon
sudo airmon-ng start wlan1

# 2a. capture everything the radio hears on its current channel
sudo tcpdump -i wlan1mon -w office_wifi_$(date +%F).pcap

# 2b. or use airodump-ng (also shows APs/clients live; writes .cap)
sudo airodump-ng wlan1mon -w office_wifi

# pin to your AP's channel for clean capture (a radio hears only one channel):
sudo iw dev wlan1mon set channel 6

# 3. stop monitor mode when done
sudo airmon-ng stop wlan1mon
```

Notes:
- A sniffer only hears the **channel it's tuned to**. Find your AP's channel
  (`airodump-ng wlan1mon` shows it), then pin it with `iw ... set channel`.
- These capture as **datalink 105 (raw 802.11)** — same format as your existing
  `office_pcap/` files.
- Encrypted WiFi: you'll still see all **management/control** frames (beacons,
  deauth, assoc) in the clear; only the data payloads are encrypted.

---

## Where to put the files

```
external_pcaps/         # IP captures for analysis
Wifi_Model/office_pcap/ # WiFi captures (already git-ignored)
```
Both `*.pcap` and `*.cap` are git-ignored, so captures never get pushed.

## Quick reference

| Want | Command |
|---|---|
| IP, one machine | `sudo tcpdump -i eth0 -w out.pcap` |
| IP, all devices | mirror/SPAN port → `tcpdump -i eth0 -w out.pcap` |
| WiFi 802.11 | `sudo airmon-ng start wlan1` → `tcpdump -i wlan1mon -w out.pcap` |
| Stop after 30 min | prefix with `sudo timeout 1800 ...` |
