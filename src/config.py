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


# Merged scheme: DoS and DDoS subtypes are flow-indistinguishable (the DoS vs
# DDoS split did NOT generalise across capture sessions — see the cross-capture
# test), so they are collapsed into one "Flood" category.
def to_category_merged(label: str) -> str:
    cat = CLASS_TO_CATEGORY[label]
    return "Flood" if cat in ("DDoS", "DoS") else cat


# Attack categories for the Stage-2 classifier (attacks only; benign handled by
# the anomaly gate).
ATTACK_CATEGORIES_MERGED = ["Flood", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]


# Fine-grained scheme: keep ALL 34-class labels, but fold ONLY the DoS+DDoS
# subtypes into a single 'Flood' label (they are flow-indistinguishable and do
# not generalise across captures as separate classes). Everything else keeps its
# individual class. -> 18 attack classes + Benign.
def to_flood_merged(label: str) -> str:
    cat = CLASS_TO_CATEGORY[label]
    return "Flood" if cat in ("DDoS", "DoS") else label


# Protocol-wise flood merge (max granularity): each DoS flood folds into its
# matching DDoS flood by protocol; everything else stays as the original 34.
# DoS-X_Flood + DDoS-X_Flood -> X_Flood  (X in UDP/SYN/TCP/HTTP). -> 30 classes.
PROTO_FLOOD_MERGE = {
    "DoS-UDP_Flood": "UDP_Flood",  "DDoS-UDP_Flood": "UDP_Flood",
    "DoS-SYN_Flood": "SYN_Flood",  "DDoS-SYN_Flood": "SYN_Flood",
    "DoS-TCP_Flood": "TCP_Flood",  "DDoS-TCP_Flood": "TCP_Flood",
    "DoS-HTTP_Flood": "HTTP_Flood", "DDoS-HTTP_Flood": "HTTP_Flood",
    # SynonymousIP is a SYN flood with spoofed source IPs -> not separable from SYN_Flood.
    "DDoS-SynonymousIP_Flood": "SYN_Flood",
}


def to_proto_flood_merged(label: str) -> str:
    return PROTO_FLOOD_MERGE.get(label, label)


# Reconnaissance scan merge: PingSweep + PortScan + OSScan are all active scans
# that trigger the SAME operator response (block the scanner) and are not
# reliably separable by flow features. HostDiscovery and VulnerabilityScan are
# kept distinct (different intent/footprint).
RECON_SCAN_MERGE = {
    "Recon-PingSweep": "Recon-Scanning",
    "Recon-PortScan":  "Recon-Scanning",
    "Recon-OSScan":    "Recon-Scanning",
}


def to_final_merged(label: str) -> str:
    """Deployable Stage-2 scheme: protocol-wise flood merge (incl. SynonymousIP
    -> SYN_Flood) PLUS the recon-scan merge. -> 26 attack classes."""
    return RECON_SCAN_MERGE.get(label, to_proto_flood_merged(label))


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
