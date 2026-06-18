"""Generate a synthetic pcap so the IDS can be tested with NO live traffic.

Writes a small capture containing four contiguous blocks — a varied benign mix,
then UDP-flood, SYN-flood and ICMP-flood bursts. Because the extractor uses a
non-overlapping 10-packet window, contiguous same-type packets become
homogeneous feature rows, so the pipeline produces clearly benign rows and clear
flood rows.

This is a PLUMBING / smoke test of the full chain (pcap -> 46 features -> gate ->
classifier), not an accuracy benchmark — the packets are crafted, not real
device traffic. For accuracy, replay a real CICIoT2023 attack pcap instead.

    python make_test_pcap.py                 # -> tests/sample_traffic.pcap
    python make_test_pcap.py --out x.pcap --scale 2
"""
from __future__ import annotations

import argparse, os, random, socket

import dpkt

random.seed(42)
SMAC, DMAC = b"\x02\x00\x00\x00\x00\x01", b"\x02\x00\x00\x00\x00\x02"


def _eth(ip):
    return bytes(dpkt.ethernet.Ethernet(
        src=SMAC, dst=DMAC, type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def _ip(p, src, dst, l4, ttl=64):
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst),
                    p=p, ttl=ttl, data=l4)
    ip.len = len(ip)
    return _eth(ip)


def _tcp(src, dst, sport, dport, flags, payload=b""):
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags,
                       seq=random.randint(0, 2**32 - 1), data=payload)
    return _ip(dpkt.ip.IP_PROTO_TCP, src, dst, seg)


def _udp(src, dst, sport, dport, payload=b""):
    seg = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    seg.ulen = len(seg)
    return _ip(dpkt.ip.IP_PROTO_UDP, src, dst, seg)


def _icmp(src, dst, payload=b"abcdefgh"):
    seg = dpkt.icmp.ICMP(type=8, data=dpkt.icmp.ICMP.Echo(
        id=random.randint(0, 65535), seq=1, data=payload))
    return _ip(dpkt.ip.IP_PROTO_ICMP, src, dst, seg)


def build(scale: int):
    """Return a list of (ts, raw_bytes), timestamps increasing."""
    pkts, ts = [], 1_700_000_000.0
    hosts = [f"192.168.1.{i}" for i in range(10, 20)]
    servers = ["192.168.1.1", "8.8.8.8", "93.184.216.34"]

    # --- benign mix: DNS, short HTTP/HTTPS exchanges, varied sizes/timing ---
    for _ in range(50 * scale):
        h = random.choice(hosts)
        kind = random.random()
        if kind < 0.35:                                   # DNS query/response
            pkts.append((ts, _udp(h, "8.8.8.8", random.randint(1024, 65000), 53,
                                   os.urandom(random.randint(20, 60)))))
            ts += random.uniform(0.01, 0.05)
            pkts.append((ts, _udp("8.8.8.8", h, 53, 40000,
                                   os.urandom(random.randint(60, 200)))))
        elif kind < 0.7:                                  # HTTPS-ish
            sp, dp = random.randint(1024, 65000), 443
            pkts.append((ts, _tcp(h, servers[2], sp, dp, dpkt.tcp.TH_SYN)))
            ts += random.uniform(0.02, 0.08)
            pkts.append((ts, _tcp(servers[2], h, dp, sp,
                                  dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK)))
            ts += random.uniform(0.02, 0.08)
            pkts.append((ts, _tcp(h, servers[2], sp, dp,
                                  dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                                  os.urandom(random.randint(200, 1400)))))
        else:                                             # HTTP GET
            sp = random.randint(1024, 65000)
            pkts.append((ts, _tcp(h, servers[2], sp, 80,
                                  dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
                                  b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")))
        ts += random.uniform(0.05, 0.25)

    # --- UDP flood: many small UDP to one victim:port, microsecond spacing ---
    victim = "192.168.1.50"
    for _ in range(300 * scale):
        pkts.append((ts, _udp(random.choice(hosts), victim,
                              random.randint(1024, 65000), 9999, b"\x00" * 32)))
        ts += 0.0001

    # --- SYN flood: many TCP SYN to victim:80 from spoofed-ish sources ---
    for _ in range(300 * scale):
        src = f"10.0.{random.randint(0,255)}.{random.randint(1,254)}"
        pkts.append((ts, _tcp(src, victim, random.randint(1024, 65000), 80,
                              dpkt.tcp.TH_SYN)))
        ts += 0.0001

    # --- ICMP flood: echo requests ---
    for _ in range(300 * scale):
        pkts.append((ts, _icmp(random.choice(hosts), victim, b"\x00" * 56)))
        ts += 0.0001

    return pkts


def main():
    ap = argparse.ArgumentParser(description="synthetic pcap for IDS testing")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "sample_traffic.pcap"))
    ap.add_argument("--scale", type=int, default=1, help="multiply packet counts")
    args = ap.parse_args()

    pkts = build(args.scale)
    with open(args.out, "wb") as fh:
        w = dpkt.pcap.Writer(fh)
        for ts, buf in pkts:
            w.writepkt(buf, ts=ts)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {len(pkts):,} packets -> {args.out} ({size_kb:.0f} KB)")
    print("blocks: benign mix, then UDP / SYN / ICMP flood bursts")


if __name__ == "__main__":
    main()
