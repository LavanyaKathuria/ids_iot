from scapy.utils import PcapReader
from scapy.layers.inet import IP
from scapy.layers.inet import TCP, UDP, IP
from scapy.layers.l2 import Ether
import pandas as pd
import numpy as np

MAX_ROWS=100

PCAP_FILE = "D:\intienrship\capture-26.cap"
WINDOW_SIZE = 10

rows = []

window = []

with PcapReader(PCAP_FILE) as pcap:

    fin_flag_number = 0
    syn_flag_number = 0
    rst_flag_number = 0
    psh_flag_number = 0
    ack_flag_number = 0
    ece_flag_number = 0
    cwr_flag_number = 0

    ack_count = 0
    syn_count = 0
    fin_count = 0
    urg_count = 0
    rst_count = 0

    HTTP = 0
    HTTPS = 0
    DNS = 0
    Telnet = 0
    SMTP = 0
    SSH = 0
    IRC = 0

    DHCP = 0
    ARP = 0
    ICMP = 0

    for pkt in pcap:

        try:
            eth = Ether(bytes(pkt))

            if eth.type == 0x0806:
             ARP = 1
             continue


            if eth.type != 0x0800:
                continue

            ip = IP(bytes(eth.payload))

            # TCP
            if ip.proto == 6 and isinstance(ip.payload, TCP):

                tcp = ip.payload
                flags = int(tcp.flags)

                if flags & 0x01:
                    fin_flag_number += 1
                    fin_count += 1

                if flags & 0x02:
                    syn_flag_number += 1
                    syn_count += 1

                if flags & 0x04:
                    rst_flag_number += 1
                    rst_count += 1

                if flags & 0x08:
                    psh_flag_number += 1

                if flags & 0x10:
                    ack_flag_number += 1
                    ack_count += 1

                if flags & 0x20:
                    urg_count += 1

                if flags & 0x40:
                    ece_flag_number += 1

                if flags & 0x80:
                    cwr_flag_number += 1

                sport = tcp.sport
                dport = tcp.dport

                if sport in [80,8080] or dport in [80,8080]:
                    HTTP = 1

                if sport == 443 or dport == 443:
                    HTTPS = 1

                if sport == 22 or dport == 22:
                    SSH = 1

                if sport == 23 or dport == 23:
                    Telnet = 1

                if sport == 25 or dport == 25:
                    SMTP = 1

                if sport in [6667,6668,6669] or dport in [6667,6668,6669]:
                    IRC = 1


            # UDP
            if ip.proto == 17 and isinstance(ip.payload, UDP):

                udp = ip.payload

                sport = udp.sport
                dport = udp.dport

                if sport == 53 or dport == 53:
                    DNS = 1

                if sport in [67,68] or dport in [67,68]:
                    DHCP = 1

            packet_info = {
                "time": float(pkt.time),
                "size": ip.len,
                "proto": ip.proto,
                "header_len": ip.ihl * 4,
                "src": ip.src,
                "dst": ip.dst
            }

            window.append(packet_info)

        except Exception:
            continue

        if len(window) == WINDOW_SIZE:

            sizes = [p["size"] for p in window]
            times = [p["time"] for p in window]
            headers = [p["header_len"] for p in window]
            protocols = [p["proto"] for p in window]
           

            duration = max(times[-1] - times[0], 0.000001)

            iats = np.diff(times)

            row = {
                "flow_duration": duration,
                "Header_Length": np.mean(headers),
                "Protocol Type": protocols[0],
                "Duration": duration,
                "Rate": len(window) / duration,

                "Srate": len(window) / duration,
                "Drate": 0,

                "fin_flag_number": fin_flag_number,
                "syn_flag_number": syn_flag_number,
                "rst_flag_number": rst_flag_number,
                "psh_flag_number": psh_flag_number,
                "ack_flag_number": ack_flag_number,
                "ece_flag_number": ece_flag_number,
                "cwr_flag_number": cwr_flag_number,

                "ack_count": ack_count,
                "syn_count": syn_count,
                "fin_count": fin_count,
                "urg_count": urg_count,
                "rst_count": rst_count,

                "HTTP": HTTP,
                "HTTPS": HTTPS,
                "DNS": DNS,
                "Telnet": Telnet,
                "SMTP": SMTP,
                "SSH": SSH,
                "IRC": IRC,

                "TCP": int(6 in protocols),
                "UDP": int(17 in protocols),

                "DHCP": DHCP,
                "ARP": ARP,
                "ICMP": int(1 in protocols),
                "IPv": 4,
                "LLC": 0,

                "Tot sum": np.sum(sizes),
                "Min": np.min(sizes),
                "Max": np.max(sizes),
                "AVG": np.mean(sizes),
                "Std": np.std(sizes),

                "Tot size": np.mean(sizes),

                "IAT": np.mean(iats) if len(iats) else 0,

                "Number": len(window),

                "Magnitue": np.linalg.norm(sizes),
                "Radius": np.sqrt(np.var(sizes)),
                "Covariance": float(np.var(sizes)),
                "Variance": np.var(sizes),

                "Weight": np.sum(sizes) / len(window),

                "label": "DDoS_HTTP_Flood"
            }

            rows.append(row)

            if len(rows) >= MAX_ROWS:
                 break

            window = []

            fin_flag_number = 0
            syn_flag_number = 0
            rst_flag_number = 0
            psh_flag_number = 0
            ack_flag_number = 0
            ece_flag_number = 0
            cwr_flag_number = 0

            ack_count = 0
            syn_count = 0
            fin_count = 0
            urg_count = 0
            rst_count = 0

            HTTP = 0
            HTTPS = 0
            DNS = 0
            Telnet = 0
            SMTP = 0
            SSH = 0
            IRC = 0

            DHCP = 0
            ARP = 0
            ICMP = 0

df = pd.DataFrame(rows)

df.to_csv("D:\SAG\sampling.csv", index=False)

print("done successfully written")
print("Rows generated:", len(df))