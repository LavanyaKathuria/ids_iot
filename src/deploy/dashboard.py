"""Streamlit dashboard for the two-stage IDS — live alert view for a monitor.

Reads the alerts CSV that ids_runner.py appends to (columns: ts,label,source)
and refreshes on an interval. Run it on the monitor/computer, or on the Pi and
open it from any browser on the LAN.

    # point it at the runner's alerts file (env var or --) and launch:
    streamlit run dashboard.py
    # serve to the whole LAN (view from another machine's browser):
    streamlit run dashboard.py --server.address 0.0.0.0

Set the alerts path with the IDS_ALERTS env var (default: ./alerts.csv).
"""
from __future__ import annotations

import os, time

import pandas as pd
import streamlit as st

ALERTS = os.environ.get("IDS_ALERTS", "alerts.csv")
REFRESH_SEC = int(os.environ.get("IDS_REFRESH", "3"))

st.set_page_config(page_title="IoT IDS — live", layout="wide",
                   page_icon="🛡️")


def load() -> pd.DataFrame:
    if not os.path.exists(ALERTS):
        return pd.DataFrame(columns=["ts", "label", "source"])
    try:
        df = pd.read_csv(ALERTS)
    except Exception:
        return pd.DataFrame(columns=["ts", "label", "source"])
    for c in ("ts", "label", "source"):
        if c not in df.columns:
            df[c] = ""
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


st.title("🛡️ CICIoT2023 IoT IDS — live alerts")
st.caption(f"source: `{ALERTS}` · auto-refresh {REFRESH_SEC}s · "
           f"{time.strftime('%H:%M:%S')}")

df = load()

if df.empty:
    st.info("No alerts yet. Start the runner: "
            "`sudo python3 ids_runner.py --iface eth0` "
            "(or replay a pcap with `--replay`).")
else:
    recent = df[df["ts"] > pd.Timestamp.now() - pd.Timedelta(minutes=1)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts", f"{len(df):,}")
    c2.metric("Attack types", df["label"].nunique())
    c3.metric("Last minute", f"{len(recent):,}")
    c4.metric("Last alert",
              df["ts"].max().strftime("%H:%M:%S") if df["ts"].notna().any() else "—")

    left, right = st.columns(2)
    with left:
        st.subheader("Alerts by attack type")
        st.bar_chart(df["label"].value_counts())
    with right:
        st.subheader("Alerts over time")
        timeline = (df.dropna(subset=["ts"])
                      .set_index("ts").resample("10s").size().rename("alerts"))
        if len(timeline):
            st.line_chart(timeline)

    st.subheader("Most recent alerts")
    st.dataframe(df.sort_values("ts", ascending=False).head(100),
                 use_container_width=True, hide_index=True)

time.sleep(REFRESH_SEC)
st.rerun()
