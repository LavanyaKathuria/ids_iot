"""Shared configuration for the CICIoT2023 IDS pipeline.

Centralises everything that must stay consistent between training and the
Raspberry Pi inference step: the raw feature schema, the 34 -> 8 category
mapping, and the binary mapping.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw schema (46 features + label) as produced by the CICIoT2023 extractor.
# ---------------------------------------------------------------------------
RAW_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count", "HTTP",
    "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
LABEL_COL = "label"

# ---------------------------------------------------------------------------
# 34-class -> 8-category mapping (official CICIoT2023 grouping).
# ---------------------------------------------------------------------------
CLASS_TO_CATEGORY = {
    # DDoS (12)
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    # DoS (4)
    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    # Mirai (3)
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon (5)
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon", "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    # Spoofing (2)
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    # Web (6)
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web",
    "Uploading_Attack": "Web", "SqlInjection": "Web", "CommandInjection": "Web",
    # Brute Force (1)
    "DictionaryBruteForce": "BruteForce",
    # Benign (1)
    "BenignTraffic": "Benign",
}

CATEGORIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web",
              "BruteForce"]
BENIGN_LABEL = "BenignTraffic"


def to_category(label: str) -> str:
    return CLASS_TO_CATEGORY[label]


def to_binary(label: str) -> str:
    return "Benign" if label == BENIGN_LABEL else "Attack"


# ---------------------------------------------------------------------------
# Sampling configuration.
# ---------------------------------------------------------------------------
# Rule: a 34-class label is downsampled to PER_CLASS_CAP *only if* its total
# count exceeds DOWNSAMPLE_THRESHOLD. Everything at or below the threshold is
# kept in full. This curbs the giant DDoS/DoS flood classes while preserving
# all rows for medium and rare classes.
DOWNSAMPLE_THRESHOLD = 1_000_000
PER_CLASS_CAP = 100_000
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "CICIoT2023")
ARTIFACTS = os.path.join(ROOT, "artifacts")
SAMPLED_PARQUET = os.path.join(ARTIFACTS, "sampled.parquet")
MODELS_DIR = os.path.join(ARTIFACTS, "models")
REPORTS_DIR = os.path.join(ARTIFACTS, "reports")
