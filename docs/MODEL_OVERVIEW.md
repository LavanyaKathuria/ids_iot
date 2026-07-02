# Model Overview — IP-layer IDS (current model)

A plain-language summary of the **deployed** model: what data trained it, what
features it uses, the algorithm, and the techniques that make it work.

> One line: a **single DecisionTree** that reads **flow statistics** (not packet
> payloads, not ports) and labels each traffic window as **Benign or one of 24
> attack classes**.

---

## 1. Data used

**Base dataset — CICIoT2023** (~1.29 million feature windows in
`artifacts/pcap_features/`). The original 34 attack classes are merged down to
**24** for deployment (`config.to_deploy_merged`):
- DoS + DDoS floods of the same protocol → one flood class (e.g. `UDP_Flood`).
- The 3 recon scans (PingSweep/PortScan/OSScan) → `Recon-Scanning`.
- The 3 fragmentation types (ACK/UDP/ICMP) → `DDoS-Fragmentation`
  (they're flow-indistinguishable — fragments carry no L4 header).

**Own-network captures** (`captures/`, listed in `captures/train_manifest.csv`)
— real floods/scans generated against our own host and added to training,
**balanced** (capped at 6,000 windows per class so no class dominates). These
teach the model our network's real traffic profile for the fragile classes.

| class group | source |
|---|---|
| All 24 attack types + Benign | CICIoT2023 |
| SYN/UDP/TCP/PSH-ACK/RST-FIN/ICMP floods, fragmentation, recon | + own captures (balanced) |

---

## 2. Features — 62 behavioural

Each row = a **10-packet window**, summarised into 62 numbers. Built by
`pcap_to_features.py` (46 raw) → `fast_transform.build_matrix` (adds engineered),
then the **11 port-identity features are dropped** (see technique #1):

| group | examples |
|---|---|
| Flow timing / rate | `flow_duration`, `Rate`, `Srate`, `Drate`, `IAT` |
| TCP flags | `syn/ack/rst/psh/fin/ece/cwr` flag-numbers, `syn/ack/rst/fin/urg` counts |
| Transport protocol | `TCP`, `UDP`, `ICMP`, `ARP`, `Protocol Type` |
| Packet sizes | `Min`, `Max`, `AVG`, `Std`, `Tot size/sum`, `Header_Length`, `Number` |
| Directional stats | `Magnitude`, `Radius`, `Covariance`, `Variance`, `Weight` |
| Engineered ratios | flag ratios, rate-per-byte, size dispersion, `log_rate`, `log_iat`, … |

**Dropped (11):** `HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, DHCP` (these are just
"is the port 80/53/22/…") plus `l7_proto_count, tcp_no_l7, udp_no_l7` derived from
them. See `config.PORT_FEATURES`.

---

## 3. Algorithm

**scikit-learn `DecisionTreeClassifier`**, single stage (no anomaly gate):
```python
DecisionTreeClassifier(max_depth=35, min_samples_leaf=5,
                       class_weight="balanced", random_state=42)
```
- **Why a tree, not a neural net:** an MLP was marginally better in-distribution
  but *less* robust on real captures (mislabelled clean SYN floods) and ~3.6× slower.
  The tree is robust, interpretable, and runs in microseconds on a Pi.
- **Why flat (no two-stage gate):** the two-stage version needs on-site benign
  calibration; without it, a single flat classifier with a Benign class is used.
- Size: ~2.2 MB. Output: `flat_dtree_portagnostic.joblib` + `label_encoder_flat.joblib`
  + `feature_list_behavioral.json`.

---

## 4. Key techniques

1. **Port-agnostic features** — dropping the port flags stops the model taking a
   spurious "port → attack" shortcut (e.g. SYN-flood-to-:80 → HTTP_Flood). It now
   judges by behaviour, so floods are detected on **any port**.
2. **Principled class merges** — flow-indistinguishable classes that trigger the
   same response are merged (floods, recon scans, fragmentation).
3. **Balanced own-capture augmentation** — real network captures added per class,
   capped equally (unbalanced augmentation made one class a catch-all).
4. **Held-out validation** — every added class is tested on a *separate* capture
   it was never trained on, and any CICIoT class dropping >0.02 F1 blocks the change.

---

## 5. Performance

- **CICIoT held-out:** accuracy **0.939**, macro-F1 0.724.
- **Cross-capture held-out** (train one capture, test a separate one):
  **SYN / UDP / TCP / PSH-ACK / RST-FIN / ICMP floods → 99–100%**;
  `DDoS-Fragmentation` ~60% (0.99 in-distribution).

## 6. Honest limits (confirmed by cross-laptop testing)

**Classifies correctly (real captures, on-victim / mirror-port capture):**
SYN, UDP, TCP, PSH-ACK, RST-FIN, ICMP floods (any port), Recon-Scanning, HTTP_Flood,
Mirai-udpplain (reads as UDP_Flood, its near-twin).

**Known limitations:**
- **Web attacks (SQLi / XSS / CmdInj / BrowserHijacking / Backdoor / Uploading)** —
  the flow model is blind to these; the signal is in the packet payload. Needs a
  signature/DPI engine (Suricata), not this model. *(by design)*
- **DDoS-SlowLoris** → misclassified as **DDoS-PSHACK_Flood**. Both are PSH-ACK
  traffic; SlowLoris's defining "low and slow, hold connections open" behaviour is a
  long-timescale property that a 10-packet window can't see. Needs connection-state /
  multi-window features to fix.
- **DDoS-Fragmentation** → top class correct but only ~60–80%; bleeds ~10–20% into
  Recon/UDP. Fragments carry no L4 header, so the 3 fragmentation types can't be
  separated (already merged into one class).
- **Mirai-greeth / greip** → validated on CICIoT (F1 0.999) but not faithfully
  reproducible at home (hping3 can't craft real GRE floods), so not field-tested.
- **VulnerabilityScan** → misclassified as **Recon-Scanning** (~77%). A version/vuln
  scan *is* a scan — the two are flow-indistinguishable (both are many probes to many
  ports) and trigger the same response. Within-recon-family confusion.
- **DNS_Spoofing / DictionaryBruteForce / Recon-HostDiscovery** — weak/moderate
  (CICIoT F1 0.45–0.82); not yet validated on real traffic.

**Deployment / capture limits (not the model):**
- **Capture placement is critical.** The sensor must actually see the attack traffic —
  capture **on the victim**, or on a switch **SPAN/mirror port** feeding the monitor,
  with the interface in **promiscuous mode**. A passive monitor plugged into a switched
  network *without* a mirror sees only background/broadcast → everything misclassifies.
  (This was the cause of an earlier "bad results" run.)
- The attack must **dominate the capture window** — capture *during* the active attack;
  a flood swamps the 10-packet windows and classifies cleanly even unfiltered.
- **ARP / DNS spoofing** can't be generated from WSL2 (no layer-2) — needs a native
  Linux attacker; this is a test-setup limit, not a model limit.
