"""Craft a tiny pcap with real HTTP payload attacks so Suricata fires in seconds
(no 11 GB file needed). Each attack is a full TCP handshake + an HTTP request
whose URI carries the exploit, so Suricata marks the flow established and parses
http.uri -> the local.rules (SQLi/XSS/CmdInjection) match.

    python make_web_attack_pcap.py        # -> tests/web_attack_test.pcap
Then (in WSL, after reboot):
    suricata -r /mnt/c/.../tests/web_attack_test.pcap -l out -s /mnt/c/.../src/deploy/suricata/local.rules
"""
from __future__ import annotations

import os, random
import dpkt

random.seed(1)
SMAC, DMAC = b"\x02\x00\x00\x00\x00\x01", b"\x02\x00\x00\x00\x00\x02"
CLIENT, SERVER = "192.168.1.55", "192.168.1.10"     # SERVER in HOME_NET
import socket


def _frame(src, dst, seg):
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst),
                    p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=seg)
    ip.len = len(ip)
    return bytes(dpkt.ethernet.Ethernet(src=SMAC, dst=DMAC,
                type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def _tcp(sport, dport, flags, seq, ack, payload=b""):
    return dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, seq=seq, ack=ack,
                        data=payload)


def http_attack(pkts, ts, sport, uri):
    """Emit a handshake + GET <uri> + 200 OK for one attack flow."""
    c0, s0 = random.randint(0, 2**31), random.randint(0, 2**31)
    req = (f"GET {uri} HTTP/1.1\r\nHost: {SERVER}\r\n"
           f"User-Agent: curl/8.0\r\nAccept: */*\r\n\r\n").encode()
    resp = (b"HTTP/1.1 200 OK\r\nServer: lighttpd\r\nContent-Length: 2\r\n\r\nok")
    S, SA, A, PA = (dpkt.tcp.TH_SYN, dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK,
                    dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK)
    seq = [
        (CLIENT, SERVER, _tcp(sport, 80, S,  c0,     0)),          # SYN
        (SERVER, CLIENT, _tcp(80, sport, SA, s0,     c0 + 1)),     # SYN-ACK
        (CLIENT, SERVER, _tcp(sport, 80, A,  c0 + 1, s0 + 1)),     # ACK
        (CLIENT, SERVER, _tcp(sport, 80, PA, c0 + 1, s0 + 1, req)),       # request
        (SERVER, CLIENT, _tcp(80, sport, A,  s0 + 1, c0 + 1 + len(req))), # ack data
        (SERVER, CLIENT, _tcp(80, sport, PA, s0 + 1, c0 + 1 + len(req), resp)),  # response
        (CLIENT, SERVER, _tcp(sport, 80, A,  c0 + 1 + len(req), s0 + 1 + len(resp))),
    ]
    for src, dst, segres in seq:
        pkts.append((ts, _frame(src, dst, segres)))
        ts += 0.001
    return ts


def main():
    out = os.path.join(os.path.dirname(__file__), "web_attack_test.pcap")
    pkts, ts = [], 1_700_000_000.0
    attacks = [
        (40001, "/index.php?id=1%20union%20select%20user,pass%20from%20users"),  # SQLi
        (40002, "/search?q=<script>alert(1)</script>"),                          # XSS
        (40003, "/cgi-bin/cmd?x=;cat%20/etc/passwd"),                            # CmdInjection
        (40004, "/api?id=1%27%20or%201=1--"),                                    # SQLi tautology
    ]
    for sport, uri in attacks:
        ts = http_attack(pkts, ts, sport, uri) + 0.05
    with open(out, "wb") as fh:
        w = dpkt.pcap.Writer(fh)
        for t, buf in pkts:
            w.writepkt(buf, ts=t)
    print(f"wrote {len(pkts)} packets ({len(attacks)} attack flows) -> {out}")
    print("attacks: SQLi(union/select), XSS(<script>), CmdInjection(;cat), SQLi(' or 1=1)")


if __name__ == "__main__":
    main()
