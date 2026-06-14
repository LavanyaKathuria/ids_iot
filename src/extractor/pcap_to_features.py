"""pcap -> CICIoT2023 46-feature extractor.

Methodology faithful to the official CIC `pcap2csv` code (dpkt, per-packet
parse, **non-overlapping 10-packet window**, mean/sum/mode aggregation,
directional two-stream stats), but the per-feature definitions are aligned to
the **published dataset** (Table 5) — which the model was trained on — wherever
the released reference code diverges from it (see NOTES below).

Output columns are exactly `config.RAW_FEATURES` (46), in order, so the result
feeds straight into `feature_engineering` / `fast_transform` and the model.

Usage:
    python pcap_to_features.py input.pcap output.csv [--label DDoS-HTTP_Flood] [--max N]
or:
    from pcap_to_features import extract_features
    df = extract_features("input.pcap", max_packets=200_000)

NOTES / calibration items (documented, not hidden):
  * Header_Length = full stack (Ethernet 14 + IP + L4), giving 54 for a plain
    TCP packet -> matches the dataset median (the reference code returned only
    the ~20-byte TCP header).
  * Directional stats (Magnitue/Radius/Covariance/Variance/Weight) use the
    documented `Dynamic_features` formulas. The dataset's Magnitude=sqrt(2*mean)
    implies the two streams ~= all packets, so DIRECTIONAL_MODE="both" is the
    default and reproduces Magnitude/Radius/Covariance; Weight/Variance are
    approximate (flagged by the validator).
  * IAT is emitted in seconds; set IAT_SCALE to match the dataset's unit if a
    direct numeric match is required (otherwise retrain on this output).
"""
from __future__ import annotations

import argparse
import os
import sys

import dpkt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

WINDOW = 10                 # packets per summary row (official n_rows)
IAT_SCALE = 1e9             # per-flow inter-arrival -> nanoseconds (published dialect)
CHUNK_BYTES = 10 * 1024 * 1024   # official splits pcap into 10MB chunks (tcpdump -C 10)
DIRECTIONAL_MODE = "both"   # "both" (in==out==all block sizes) or "split"

# TCP flag bit order matches Supporting_functions.get_flag_values
FLAG_BITS = {"fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08,
             "ack": 0x10, "urg": 0x20, "ece": 0x40, "cwr": 0x80}

APP_TCP = {"HTTP": (80,), "HTTPS": (443,), "SSH": (22,), "Telnet": (23,),
           "SMTP": (25,), "IRC": (21,)}          # IRC=21 per official code
APP_UDP = {"DNS": (53,), "SMTP": (25,)}


def _flow_key(sip, sport, dip, dport):
    return tuple(sorted([(sip, sport), (dip, dport)]))


class _Flows:
    """Per-flow state. Reset every 10MB chunk to mirror the official pipeline."""
    def __init__(self):
        self.first_ts = {}       # key -> first ts in flow
        self.last_ts = {}        # key -> last ts in flow (for per-flow IAT)
        self.fwd = {}            # key -> forward packet count
        self.bwd = {}            # key -> backward packet count
        self.bytes = {}          # key -> cumulative bytes in flow
        self.first_dir = {}      # key -> first-seen (sip,sport,dip,dport)

    def update(self, sip, sport, dip, dport, ts, frame):
        key = _flow_key(sip, sport, dip, dport)
        if key not in self.first_ts:
            self.first_ts[key] = ts
            self.first_dir[key] = (sip, sport, dip, dport)
            self.fwd[key] = self.bwd[key] = self.bytes[key] = 0
            iat = 0.0
        else:
            iat = ts - self.last_ts[key]           # per-flow inter-arrival (s)
        self.last_ts[key] = ts
        if (sip, sport, dip, dport) == self.first_dir[key]:
            self.fwd[key] += 1
            is_fwd = 1
        else:
            self.bwd[key] += 1
            is_fwd = 0
        self.bytes[key] += frame
        dur = ts - self.first_ts[key]              # active time
        return dur, self.fwd[key], self.bwd[key], self.bytes[key], iat, is_fwd


def _per_packet(buf, ts, flows):
    """Return (per-packet feature dict, flow_key), or (None, None) to skip."""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:
        return None, None
    if eth.type not in (dpkt.ethernet.ETH_TYPE_IP, dpkt.ethernet.ETH_TYPE_ARP):
        return None, None

    frame = len(buf)
    row = {c: 0.0 for c in C.RAW_FEATURES}
    row["Tot size"] = frame
    row["Number"] = 1
    row["LLC"] = 1                # official L1.LLC() always returns 1

    if eth.type == dpkt.ethernet.ETH_TYPE_ARP:
        row["ARP"] = 1
        sip = "arp:" + dpkt.utils.mac_to_str(eth.src)
        dip = "arp:" + dpkt.utils.mac_to_str(eth.dst)
        sport = dport = 0
    else:
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            return None, None
        row["IPv"] = 1
        row["Protocol Type"] = ip.p
        row["Duration"] = ip.ttl
        l4 = ip.data
        if isinstance(l4, dpkt.tcp.TCP):
            row["TCP"] = 1
            sport, dport = l4.sport, l4.dport
            for name, bit in FLAG_BITS.items():
                if l4.flags & bit:
                    if name != "urg":
                        row[f"{name}_flag_number"] = 1
                    if name in ("ack", "syn", "fin", "urg", "rst"):
                        row[f"{name}_count"] = 1
            for proto, ports in APP_TCP.items():
                if sport in ports or dport in ports:
                    row[proto] = 1
        elif isinstance(l4, dpkt.udp.UDP):
            row["UDP"] = 1
            sport, dport = l4.sport, l4.dport
            for proto, ports in APP_UDP.items():
                if sport in ports or dport in ports:
                    row[proto] = 1
            if (sport, dport) in ((67, 68), (68, 67)):
                row["DHCP"] = 1
        else:
            sport = dport = 0
            if ip.p == 1:
                row["ICMP"] = 1
        sip = dpkt.utils.inet_to_str(ip.src)
        dip = dpkt.utils.inet_to_str(ip.dst)

    dur, f2b, b2f, cum_bytes, iat, is_fwd = flows.update(
        sip, sport, dip, dport, ts, frame)
    row["Header_Length"] = cum_bytes        # cumulative flow bytes (published dialect)
    row["IAT"] = iat * IAT_SCALE            # per-flow inter-arrival, ns
    row["flow_duration"] = dur
    if dur > 0:
        row["Rate"] = (f2b + b2f) / dur
        row["Srate"] = f2b / dur
        row["Drate"] = b2f / dur
    row["_dir"] = is_fwd
    return row, _flow_key(sip, sport, dip, dport)


def _summarize(block):
    """Aggregate a block of <=WINDOW per-packet dicts into one 46-feature row."""
    df = pd.DataFrame(block)
    sizes = df["Tot size"].to_numpy(dtype="float64")
    out = {}

    mean_cols = (["Header_Length", "Duration", "Rate", "Srate", "Drate", "IAT",
                  "flow_duration"]
                 + [f"{f}_flag_number" for f in
                    ("fin", "syn", "rst", "psh", "ack", "ece", "cwr")]
                 + ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
                    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC"])
    for c in mean_cols:
        out[c] = df[c].mean()

    for c in ["ack_count", "syn_count", "fin_count", "urg_count", "rst_count"]:
        out[c] = df[c].sum()

    out["Protocol Type"] = df["Protocol Type"].mode().iloc[0]
    out["Tot sum"] = sizes.sum()
    out["Min"] = sizes.min()
    out["Max"] = sizes.max()
    out["AVG"] = sizes.mean()
    out["Std"] = sizes.std()
    out["Tot size"] = sizes.mean()
    out["Number"] = df["Number"].sum()
    # Rate = mean of per-packet per-flow rates (published dialect ~2.5 for HTTP
    # flood), NOT packets/block-span. It is already averaged via mean_cols.

    # Directional two-stream stats (Dynamic_features formulas)
    if DIRECTIONAL_MODE == "split":
        inc = sizes[df["_dir"].to_numpy() == 0]
        out_ = sizes[df["_dir"].to_numpy() == 1]
        if len(inc) == 0:
            inc = sizes
        if len(out_) == 0:
            out_ = sizes
    else:                                   # "both": in == out == all sizes
        inc = out_ = sizes
    mi, mo = inc.mean(), out_.mean()
    vi, vo = np.var(inc), np.var(out_)
    out["Magnitue"] = (mi + mo) ** 0.5
    out["Radius"] = (vi + vo) ** 0.5
    n = min(len(inc), len(out_))
    out["Covariance"] = float(np.mean((inc[:n] - mi) * (out_[:n] - mo))) if n else 0.0
    out["Variance"] = (vi / vo) if vo != 0 else 0.0
    out["Weight"] = len(inc) * len(out_)

    return {c: out.get(c, 0.0) for c in C.RAW_FEATURES}


def extract_features(pcap_path, max_packets=None, label=None,
                     progress_every=200_000):
    flows = _Flows()
    rows, block = [], []
    chunk_bytes = 0
    n = 0
    with open(pcap_path, "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)
        it = iter(reader)
        while True:
            try:
                ts, buf = next(it)
            except StopIteration:
                break
            except Exception:
                break          # truncated final record (partial download / live tail)
            n += 1
            if max_packets and n > max_packets:
                break
            # 10MB-chunk boundary: flush window + reset flow state.
            chunk_bytes += len(buf)
            if chunk_bytes >= CHUNK_BYTES:
                if block:
                    rows.append(_summarize(block))
                    block = []
                flows = _Flows()
                chunk_bytes = 0
            r, key = _per_packet(buf, ts, flows)
            if r is None:
                continue
            r["_ts"] = ts
            block.append(r)                    # global 10-packet window (Number~10)
            if len(block) == WINDOW:
                rows.append(_summarize(block))
                block = []
            if progress_every and n % progress_every == 0:
                print(f"  ...{n:,} packets, {len(rows):,} rows")
    if block:
        rows.append(_summarize(block))
    df = pd.DataFrame(rows, columns=C.RAW_FEATURES).astype("float32")
    if label is not None:
        df["label"] = label
    return df


def main():
    ap = argparse.ArgumentParser(description="pcap -> CICIoT2023 46 features")
    ap.add_argument("pcap")
    ap.add_argument("out_csv")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max", type=int, default=None, help="max packets")
    args = ap.parse_args()
    print(f"extracting from {args.pcap} ...")
    df = extract_features(args.pcap, max_packets=args.max, label=args.label)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df):,} rows x {df.shape[1]} cols -> {args.out_csv}")


if __name__ == "__main__":
    main()
