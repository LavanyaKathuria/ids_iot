"""Domain-informed feature engineering for CICIoT2023.

Every engineered feature is a deterministic, per-flow algebraic function of the
raw 46 features. There is no cross-row / windowed state, so the *exact* same
transform runs at training time and on the Raspberry Pi after CICFlowMeter
emits a flow record -> no train/serve skew, no label leakage.

Rationale by group:
  * Flag dynamics  -> SYN floods, RST/FIN floods, scan handshakes have very
    characteristic flag *ratios* that raw counts hide.
  * Rate / throughput asymmetry -> volumetric DDoS vs benign chatter.
  * Packet-size dispersion -> fragmentation & flood attacks have degenerate
    size distributions (near-zero variance or fixed sizes).
  * Protocol breadth -> benign hosts speak several L7 protocols; floods speak
    one transport protocol with no L7.
  * Header / inter-arrival consistency -> crafted packets violate the normal
    header-to-payload and rate-vs-IAT relationships.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6

# Engineered columns, in a fixed order so the matrix is stable across runs.
ENGINEERED_FEATURES = [
    # --- flag dynamics ---
    "flag_number_sum", "flag_count_sum", "syn_ack_ratio", "rst_ratio",
    "fin_ratio", "urg_ratio", "syn_no_ack", "rst_fin_combo",
    # --- rate / throughput ---
    "log_rate", "rate_dst_ratio", "rate_per_byte", "bytes_per_duration",
    "rate_iat_consistency",
    # --- size dispersion ---
    "size_range", "size_cv", "min_max_ratio", "header_to_size",
    "avg_pkt_size", "size_per_count",
    # --- protocol breadth ---
    "l7_proto_count", "transport_count", "tcp_no_l7", "udp_no_l7",
    # --- stat-feature consistency ---
    "variance_to_avg", "radius_to_magnitude", "duration_ratio", "log_iat",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with engineered columns added and cleaned.

    `df` must contain the raw 46 CICIoT2023 feature columns.
    """
    f = df.copy()

    # --- flag dynamics ---------------------------------------------------
    f["flag_number_sum"] = (f["fin_flag_number"] + f["syn_flag_number"]
                            + f["rst_flag_number"] + f["psh_flag_number"]
                            + f["ack_flag_number"] + f["ece_flag_number"]
                            + f["cwr_flag_number"])
    f["flag_count_sum"] = (f["ack_count"] + f["syn_count"] + f["fin_count"]
                           + f["urg_count"] + f["rst_count"])
    fcs = f["flag_count_sum"] + EPS
    f["syn_ack_ratio"] = f["syn_count"] / (f["ack_count"] + EPS)
    f["rst_ratio"] = f["rst_count"] / fcs
    f["fin_ratio"] = f["fin_count"] / fcs
    f["urg_ratio"] = f["urg_count"] / fcs
    # SYN set but ACK absent -> half-open / SYN-flood signature.
    f["syn_no_ack"] = f["syn_flag_number"] * (1.0 - f["ack_flag_number"])
    f["rst_fin_combo"] = f["rst_flag_number"] * f["fin_flag_number"]

    # --- rate / throughput ----------------------------------------------
    f["log_rate"] = np.log1p(f["Rate"].clip(lower=0))
    f["rate_dst_ratio"] = f["Srate"] / (f["Drate"] + EPS)
    f["rate_per_byte"] = f["Rate"] / (f["Tot size"] + EPS)
    f["bytes_per_duration"] = f["Tot size"] / (f["flow_duration"] + EPS)
    # If Rate and IAT are mutually consistent the product is ~O(1); crafted
    # floods break this relationship.
    f["rate_iat_consistency"] = f["Rate"] * f["IAT"]

    # --- size dispersion -------------------------------------------------
    f["size_range"] = f["Max"] - f["Min"]
    f["size_cv"] = f["Std"] / (f["AVG"] + EPS)
    f["min_max_ratio"] = f["Min"] / (f["Max"] + EPS)
    f["header_to_size"] = f["Header_Length"] / (f["Tot size"] + EPS)
    f["avg_pkt_size"] = f["Tot sum"] / (f["Number"] + EPS)
    f["size_per_count"] = f["Tot size"] / (f["Number"] + EPS)

    # --- protocol breadth ------------------------------------------------
    f["l7_proto_count"] = (f["HTTP"] + f["HTTPS"] + f["DNS"] + f["Telnet"]
                           + f["SMTP"] + f["SSH"] + f["IRC"])
    f["transport_count"] = f["TCP"] + f["UDP"] + f["ICMP"] + f["ARP"]
    f["tcp_no_l7"] = f["TCP"] * (f["l7_proto_count"] < EPS).astype("float32")
    f["udp_no_l7"] = f["UDP"] * (f["l7_proto_count"] < EPS).astype("float32")

    # --- stat-feature consistency ---------------------------------------
    f["variance_to_avg"] = f["Variance"] / (f["AVG"] + EPS)
    f["radius_to_magnitude"] = f["Radius"] / (f["Magnitue"] + EPS)
    f["duration_ratio"] = f["flow_duration"] / (f["Duration"] + EPS)
    f["log_iat"] = np.log1p(f["IAT"].clip(lower=0))

    # Clean: kill inf produced by divisions, then impute. 0 is a safe,
    # tree-friendly sentinel and keeps RF/DT baselines happy too.
    f[ENGINEERED_FEATURES] = (f[ENGINEERED_FEATURES]
                              .replace([np.inf, -np.inf], np.nan)
                              .fillna(0.0)
                              .astype("float32"))
    return f
