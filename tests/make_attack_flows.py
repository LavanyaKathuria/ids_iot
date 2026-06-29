"""Build 'a few flows from each attack' for the unified-pipeline test.

Slices the first N packets of each real CICIoT2023 per-class capture in pcap/
into a small pcap under tests/flows/, named by its TRUE (merged) label. These
small per-attack pcaps are what test_flows.py streams through pipeline.py one at
a time, replicating live arrivals.
"""
from __future__ import annotations

import os, sys
import dpkt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config as C  # noqa: E402

PCAP_DIR = os.path.join(C.ROOT, "pcap")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flows")
N = 20000  # packets per slice (enough to contain attack traffic + ML windows)

# source capture -> true label (merged scheme). Mix of ML-strong, behavioral, Web, benign.
SOURCES = {
    # Web / payload (ML-weak, Suricata-strong)
    "SqlInjection.pcap": "SqlInjection",
    "XSS.pcap": "XSS",
    "commandinjection.pcap": "CommandInjection",
    "browserhijacking.pcap": "BrowserHijacking",
    "Uploading_Attack.pcap": "Uploading_Attack",
    "Backdoor_Malware.pcap": "Backdoor_Malware",
    # ML-strong (behavioral/volumetric)
    "DDoS-SYN_Flood.pcap": "SYN_Flood",
    "DDoS-ICMP_Flood11.pcap": "DDoS-ICMP_Flood",
    "DDoS-TCP_Flood17.pcap": "TCP_Flood",
    "Mirai-udpplain1.pcap": "Mirai-udpplain",
    # behavioral / other
    "DictionaryBruteForce.pcap": "DictionaryBruteForce",
    "Recon-PortScan.pcap": "Recon-Scanning",
    "MITM-ArpSpoofing.pcap": "MITM-ArpSpoofing",
    "DNS_Spoofing.pcap": "DNS_Spoofing",
    "VulnerabilityScan.pcap": "VulnerabilityScan",
    # benign (false-positive check)
    "BenignTraffic2.pcap": "BenignTraffic",
}


def slice_pcap(src, dst, n):
    with open(src, "rb") as f:
        try:
            r = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0); r = dpkt.pcapng.Reader(f)
        with open(dst, "wb") as o:
            w = dpkt.pcap.Writer(o)
            c = 0
            for ts, buf in r:
                w.writepkt(buf, ts=ts); c += 1
                if c >= n:
                    break
    return c


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, label in SOURCES.items():
        src = os.path.join(PCAP_DIR, fname)
        if not os.path.exists(src):
            print(f"  SKIP (missing): {fname}"); continue
        dst = os.path.join(OUT_DIR, f"{label}.pcap")
        c = slice_pcap(src, dst, N)
        print(f"  {label:22s} <- {fname:32s} {c:6d} pkts")
    print(f"\nflows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
