"""GRPO-Guard monitoring panel (interview demo).

Streamlit app that reads an append-only event log (the SAME canonical
JSON the guard writes) and renders: event-stream overview, guard decision
distribution with reason codes, lineage tracing (event -> parents/children),
and run health (sync / canary / validation).  Zero GPU: point it at any
evidence dir, e.g.

    uv run streamlit run examples/monitor/panel.py -- --event-dir artifacts/v0.1.0/loop/events/events

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="GRPO-Guard Monitor", layout="wide")

EVENT_TYPES = {
    "generation": "GenerationEvent",
    "reward_finished": "RewardEvent",
    "validation_decision": "ValidationDecisionEvent",
    "update_input": "UpdateInputEvent",
    "update_committed": "UpdateCommittedEvent",
    "sync": "SyncEvent",
    "canary_passed": "CanaryEvent",
}


@st.cache_data(show_spinner=False)
def load_events(event_dir: str) -> list[dict]:
    out = []
    for p in sorted(Path(event_dir).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", default="artifacts/v0.1.0/loop/events/events")
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ns = parser.parse_args(args)

    events = load_events(ns.event_dir)
    st.title("GRPO-Guard — append-only event monitor")
    st.caption(f"{len(events)} events from `{ns.event_dir}`")

    df = pd.DataFrame(events)

    # ---- overview row -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(events))
    types = df["event_type"].value_counts().to_dict()
    c2.metric("Generation", types.get("generation", 0))
    c3.metric("Validation", types.get("validation_decision", 0))
    c4.metric("Sync / Canary", types.get("sync", 0) + types.get("canary_passed", 0))

    # ---- event-type histogram --------------------------------------------
    st.subheader("Event stream")
    st.bar_chart(df["event_type"].value_counts())

    # ---- decision panel ---------------------------------------------------
    st.subheader("Guard decisions (allow / quarantine / reject)")
    dec = df[df["event_type"] == "validation_decision"]
    if len(dec):
        decisions = dec.apply(
            lambda r: (r.get("decision_payload") or {}).get("decision", "unknown"), axis=1
        ).value_counts()
        st.bar_chart(decisions)
        codes = []
        for r in dec["decision_payload"]:
            for c in (r or {}).get("reason_codes", []):
                codes.append(c)
        st.dataframe(pd.Series(codes).value_counts().rename("count"), use_container_width=True)
    else:
        st.info("no validation decisions in this event dir")

    # ---- lineage trace -----------------------------------------------------
    st.subheader("Lineage trace")
    ev_id = st.text_input("event_id (e.g. gen-0-loop-1787364111-0004)", "")
    if ev_id:
        by_id = {e["event_id"]: e for e in events}
        by_ref = {}
        for e in events:
            for ref in e.get("input_events", []) or []:
                by_ref.setdefault(ref.get("event_id"), []).append(e["event_id"])
        chain = []
        seen = set()

        def walk(eid: str, depth: int = 0):
            if eid in seen or depth > 4:
                return
            seen.add(eid)
            e = by_id.get(eid)
            if not e:
                return
            chain.append((depth, e["event_type"], eid))
            for ref in e.get("input_events", []) or []:
                walk(ref.get("event_id", ""), depth + 1)
            for child in by_ref.get(eid, []):
                walk(child, depth + 1)

        walk(ev_id)
        st.dataframe(
            pd.DataFrame(chain, columns=["depth", "type", "event_id"]),
            use_container_width=True,
            hide_index=True,
        )

    # ---- run health ---------------------------------------------------------
    st.subheader("Run health")
    synced = df[df["event_type"] == "sync"]
    if len(synced):
        st.metric("Observed weight-sync events", len(synced))
    canary = df[df["event_type"] == "canary_passed"]
    if len(canary):
        st.write("Canary passes:", len(canary))

    st.dataframe(df[["event_type", "event_id", "component_id", "created_at_utc"]].tail(50),
                 use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
