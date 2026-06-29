#!/bin/bash
# Run Suricata in unix-socket mode: load rules ONCE, process many flow pcaps.
# Usage: suricata_batch.sh OUTDIR LOCALRULES PCAP1 [PCAP2 ...]
# eve.json events carry "pcap_filename" so alerts attribute back to each flow.
set -u
OUT="$1"; LOCAL="$2"; shift 2
SOCK=/tmp/ids_suri.socket
mkdir -p "$OUT"; rm -f "$OUT"/eve.json "$SOCK"

suricata --unix-socket="$SOCK" -l "$OUT" -k none \
  --set vars.address-groups.EXTERNAL_NET=any -s "$LOCAL" -D >/dev/null 2>&1

for i in $(seq 1 300); do
  suricatasc -c uptime "$SOCK" >/dev/null 2>&1 && break
  sleep 1
done
echo "daemon up after ${i}s; submitting $# pcap(s)"

for p in "$@"; do
  name=$(basename "$p" .pcap)
  mkdir -p "$OUT/$name"
  suricatasc -c "pcap-file $p $OUT/$name" "$SOCK" >/dev/null 2>&1
done

for i in $(seq 1 900); do
  n=$(suricatasc -c pcap-file-number "$SOCK" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  cur=$(suricatasc -c pcap-current "$SOCK" 2>/dev/null)
  if [ "${n:-0}" = "0" ] && echo "$cur" | grep -q None; then break; fi
  sleep 2
done
sleep 3
suricatasc -c shutdown "$SOCK" >/dev/null 2>&1
echo "BATCH_DONE alerts=$(grep -c '\"event_type\":\"alert\"' "$OUT"/eve.json 2>/dev/null)"
