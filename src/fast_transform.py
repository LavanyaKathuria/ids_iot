"""Vectorised numpy feature builder for low-latency inference.

`feature_engineering.engineer_features` is pandas-based and convenient for
training, but its per-operation overhead dominates single-/small-batch latency
on a Raspberry Pi. This module reproduces the EXACT same features with plain
numpy and selects the model's feature list directly into a float32 matrix.

`build_matrix` is verified against the pandas path in tests/check_parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

EPS = 1e-6


def signed_log(a):
    """Range-compressing transform used before the (distance-based) anomaly gate.
    sign-preserving log1p of |x|; tree models do not use this."""
    return np.sign(a) * np.log1p(np.abs(a))


def build_matrix(raw_df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """Return a float32 (n, len(feature_list)) matrix from raw 46-feature rows."""
    # Pull raw columns once as a float64 array for stable intermediate math.
    r = {name: raw_df[name].to_numpy(dtype="float64") for name in C.RAW_FEATURES}
    f: dict[str, np.ndarray] = dict(r)  # start with raw features

    # --- flag dynamics ---
    f["flag_number_sum"] = (r["fin_flag_number"] + r["syn_flag_number"]
                            + r["rst_flag_number"] + r["psh_flag_number"]
                            + r["ack_flag_number"] + r["ece_flag_number"]
                            + r["cwr_flag_number"])
    f["flag_count_sum"] = (r["ack_count"] + r["syn_count"] + r["fin_count"]
                           + r["urg_count"] + r["rst_count"])
    fcs = f["flag_count_sum"] + EPS
    f["syn_ack_ratio"] = r["syn_count"] / (r["ack_count"] + EPS)
    f["rst_ratio"] = r["rst_count"] / fcs
    f["fin_ratio"] = r["fin_count"] / fcs
    f["urg_ratio"] = r["urg_count"] / fcs
    f["syn_no_ack"] = r["syn_flag_number"] * (1.0 - r["ack_flag_number"])
    f["rst_fin_combo"] = r["rst_flag_number"] * r["fin_flag_number"]

    # --- rate / throughput ---
    f["log_rate"] = np.log1p(np.clip(r["Rate"], 0, None))
    f["rate_dst_ratio"] = r["Srate"] / (r["Drate"] + EPS)
    f["rate_per_byte"] = r["Rate"] / (r["Tot size"] + EPS)
    f["bytes_per_duration"] = r["Tot size"] / (r["flow_duration"] + EPS)
    f["rate_iat_consistency"] = r["Rate"] * r["IAT"]

    # --- size dispersion ---
    f["size_range"] = r["Max"] - r["Min"]
    f["size_cv"] = r["Std"] / (r["AVG"] + EPS)
    f["min_max_ratio"] = r["Min"] / (r["Max"] + EPS)
    f["header_to_size"] = r["Header_Length"] / (r["Tot size"] + EPS)
    f["avg_pkt_size"] = r["Tot sum"] / (r["Number"] + EPS)
    f["size_per_count"] = r["Tot size"] / (r["Number"] + EPS)

    # --- protocol breadth ---
    l7 = (r["HTTP"] + r["HTTPS"] + r["DNS"] + r["Telnet"] + r["SMTP"]
          + r["SSH"] + r["IRC"])
    f["l7_proto_count"] = l7
    f["transport_count"] = r["TCP"] + r["UDP"] + r["ICMP"] + r["ARP"]
    no_l7 = (l7 < EPS).astype("float64")
    f["tcp_no_l7"] = r["TCP"] * no_l7
    f["udp_no_l7"] = r["UDP"] * no_l7

    # --- stat consistency ---
    f["variance_to_avg"] = r["Variance"] / (r["AVG"] + EPS)
    f["radius_to_magnitude"] = r["Radius"] / (r["Magnitue"] + EPS)
    f["duration_ratio"] = r["flow_duration"] / (r["Duration"] + EPS)
    f["log_iat"] = np.log1p(np.clip(r["IAT"], 0, None))

    cols = [f[name] for name in feature_list]
    X = np.column_stack(cols).astype("float32")
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
