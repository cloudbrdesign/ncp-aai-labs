"""Module 11 capstone: the finished support desk, end to end.

    python m11/capstone.py run                          # four conversations through the v10 desk -> report.md
    python m11/capstone.py layers --request-id <id>     # every layer one request went through, with evidence
    python m11/capstone.py map                          # the ten exam domains -> where you built each one

Nothing new is built here. The v10 desk (m10/oversight_desk.py) already wraps v9 (safety), which wraps v8
(monitoring), v6 (evaluation), v5 (NIM, NeMo Guardrails) and the M4 graph (plan, tools, RAG, critique) with
M3's memory inside. This script sends it four conversations and reads back what each layer did, from records
the desk already writes: the M9 audit record, M8's request log, M10's cards, approvals and ledger, joined by
the request ID (m10/decision_record.py).

  1 manual     "My D300 dock shows E42. What does it mean?"       RAG over the manuals, cited, critiqued
  2 order      "Where is my order A1002?"                         the order tool, inside Tom's identity scope
  3 injection  a direct prompt-injection attempt (M9's attack j1) stopped by the input rails (layer L2 and up)
  4 refund     "I want my money back for order A1002."            an approval card; "Ana" approves; paid once

State goes to m11/state/ (M10_STATE_DIR), so Modules 9 and 10 keep their own results.
Free mode on the Mac: Ollama (llama3.2:3b + embeddinggemma), m02/ticket_api.py on 8765 and
m09/heuristics_server.py on 1337, as in Module 10. --layer L1 runs without the heuristics server
(the injection then reaches the model: the rules layer alone does not catch it).
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = pathlib.Path(os.environ.get("M11_STATE_DIR", HERE / "state"))
os.environ.setdefault("M10_STATE_DIR", str(STATE))      # before m10's bootstrap reads it
if os.environ.get("M11_FAKE_LLM") == "1":
    os.environ["M10_FAKE_LLM"] = "1"                    # offline self-test: M10's scripted model server
sys.path.insert(0, str(LABS / "m10"))

import bootstrap  # noqa: E402,F401  (m10: environment, state folder, the v9 desk and everything under it)
import approvals  # noqa: E402  (m10)
import decision_record  # noqa: E402  (m10)
import oversight_desk as ov  # noqa: E402  (m10)

ATTACK = json.loads((LABS / "m09" / "data" / "attacks.jsonl").read_text().splitlines()[0])   # j1, direct
CONVERSATIONS = [
    ("manual", "Tom B.", "My D300 dock shows E42. What does it mean?"),
    ("order", "Tom B.", "Where is my order A1002?"),
    ("injection", "Tom B.", ATTACK["text"]),
    ("refund", "Tom B.", "I want my money back for order A1002."),
]
REPORT = STATE / "capstone" / "report.md"

# The ten exam domains (NVIDIA's NCP-AAI study guide) and where the course project implements each.
DOMAINS = [
    ("Agent architecture and design", "M1", "m01/react_by_hand.py, m01/support_tools/",
     "a ReAct loop by hand, then a hand-off to a second agent over A2A"),
    ("Agent development", "M2", "m02/desk_tools/, m02/breaker_demo.py, m02/stream_client.py",
     "real tools, retries and a circuit breaker, streaming, image input"),
    ("Cognition, planning and memory", "M3", "m03/planner.py, m03/critic.py, m03/desk_memory/, m03/long_term.py",
     "a planner, a reflection step, short- and long-term memory on a checkpointed graph"),
    ("Knowledge integration and data handling", "M4", "m04/ingest.py, m04/retrieve.py, m04/router.py, m04/orders_sql.py",
     "RAG over the manuals: cleaning, chunking, a vector store, hybrid search, a SQL tool"),
    ("NVIDIA platform implementation", "M5", "m05/nim_client.py, m05/guardrails/, m05/desk_app.py",
     "one OpenAI-compatible base URL for NIM, NeMo Guardrails, NeMo Agent Toolkit"),
    ("Evaluation and tuning", "M6", "m06/testset.py, m06/evaluators.py, m06/compare_configs.py",
     "a test set, retrieval and answer metrics, LLM-as-judge, configs compared with McNemar"),
    ("Deployment and scaling", "M7", "m07/k8s/, m07/balancer.py, m07/failover_drill.py",
     "containers on EKS, autoscaling, a load test, failover, CI/CD"),
    ("Run, monitor and maintain", "M8", "m08/observability/, m08/fault_drill.py, m08/bench_release.py, m08/flywheel.py",
     "metrics, dashboards, traces, alerts, a release benchmark, the data flywheel"),
    ("Safety, ethics and compliance", "M9", "m09/rails/, m09/audit_log.py, m09/redteam.py, m09/escalate.py",
     "layered rails, identity scope, an audit log, red teaming, escalation"),
    ("Human-AI interaction and oversight", "M10", "m10/oversight_desk.py, m10/approvals.py, m10/decision_record.py",
     "approve/edit/reject on an interrupt, a stop switch, feedback measured, a record per answer"),
]


def names(res: dict) -> str:
    """The rails that ran, in order; the one that stopped the message is marked."""
    out = [r["name"] + (" (stopped it)" if r.get("stop") else "") for r in res.get("rails") or []]
    return ", ".join(out) or "-"


def layers(rid: str) -> list[tuple[str, str, str]]:
    """(module, layer, what it did) for one request, from the records the desk wrote."""
    rec = decision_record.build(rid)
    t = rec["audit"] or {}
    rows = []
    ir = t.get("input_rails") or {}
    rows.append(("M9", "input rails", f"{ir.get('status', '-')}: {names(ir)}"))
    rows.append(("M9", "identity scope", f"caller {t.get('caller', '-')}, scope {t.get('scope', '-')}"))
    if t.get("route"):
        rows.append(("M3/M4", "plan and route", " -> ".join(rec["route"]) or "-"))
    for tc in t.get("tool_calls") or []:
        rows.append(("M2/M4", f"tool {tc.get('tool')}", json.dumps(tc.get("params") or tc.get("args") or {})[:120]))
    if rec["retrieved"]:
        rows.append(("M4", "retrieval", f"retrieved {rec['retrieved']}; cited {rec['cited']}"))
    calls = [c["workload"] for c in rec["model_calls"]]
    if calls:
        rows.append(("M5/M8", "model calls", f"{t.get('model', '-')}: {', '.join(calls)} (M8 request log)"))
    if rec["critique"]:
        rows.append(("M3", "critique", rec["critique"][:120].replace("\n", " ")))
    orr = t.get("output_rails") or {}
    if orr:
        rows.append(("M9", "output rails", f"{orr.get('status', '-')}: {names(orr)}"))
    if t.get("blocked_by"):
        rows.append(("M9", "stopped by", f"{t['blocked_by']}: the customer gets a refusal instead"))
    for c in rec["cards"]:
        rows.append(("M10", "approval card", f"{c['card_id']} {c['status']}"))
    for a in rec["approvals"]:
        ap = a.get("approval") or {}
        if not ap.get("partial"):
            rows.append(("M10", "reviewer", f"{' + '.join(ap.get('reviewers') or [])}: {ap.get('decision')}, "
                         f"proposed {ap.get('proposed_args', {}).get('amount')} final {(ap.get('final_args') or {}).get('amount')}"))
    for row in rec["ledger"]:
        rows.append(("M10", "ledger", f"refund {row.get('refund_id', '')} EUR {row.get('amount')}"))
    rows.append(("M8/M10", "record", f"request {rid}, trace {t.get('trace_id', '-')}, {t.get('config_version', '-')}, "
                 f"complete {rec['complete']}"))
    return rows


def run(layer: str, say=print) -> dict:
    od = ov.OversightDesk(layer=layer)
    out = {"layer": layer, "conversations": []}
    try:
        for name, customer, text in CONVERSATIONS:
            t0 = time.time()
            r = od.chat(text, customer=customer, thread_id=f"capstone-{name}-{int(t0)}")
            if r.get("pending"):
                say(f"[{name}] card {r['pending']} waiting; reviewer Ana approves")
                approvals.decide(od, r["thread_id"], "approve", "Ana")
                r = {**r, **od.thread(r["thread_id"])}
            rid = r["request_id"]
            item = {"name": name, "customer": customer, "input": text, "action": r.get("action"),
                    "status": r.get("status"), "reply": r.get("reply"), "request_id": rid,
                    "seconds": round(time.time() - t0, 1), "layers": layers(rid)}
            out["conversations"].append(item)
            say(f"[{name}] {item['action']} / {item['status']} in {item['seconds']} s, request {rid}")
            say(f"         reply: {(item['reply'] or '').strip()[:160]}")
            for mod, lay, what in item["layers"]:
                say(f"         {mod:7} {lay:16} {what}")
    finally:
        od.close()
    write_report(out)
    say(f"[report] {REPORT}")
    return out


def write_report(out: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Capstone run ({time.strftime('%Y-%m-%d %H:%M')}, layer {out['layer']})", ""]
    for c in out["conversations"]:
        lines += [f"## {c['name']}: {c['action']} ({c['seconds']} s)", "", f"> {c['input']}", "",
                  f"Reply: {(c['reply'] or '').strip()}", "", "| Module | Layer | What it did |", "| --- | --- | --- |"]
        lines += [f"| {m} | {l} | {w.replace('|', '/')} |" for m, l, w in c["layers"]]
        lines.append("")
    REPORT.write_text("\n".join(lines))
    (REPORT.parent / "report.json").write_text(json.dumps(out, indent=1, default=str))


def show_map(say=print) -> None:
    for i, (dom, mod, files, what) in enumerate(DOMAINS, 1):
        say(f"{i:2}. {dom:42} {mod:4} {what}")
        say(f"    {'':42} {'':4} {files}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--layer", default="L4", choices=["L1", "L2", "L3", "L4"])
    l = sub.add_parser("layers")
    l.add_argument("--request-id", required=True)
    sub.add_parser("map")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.layer)
    elif a.cmd == "layers":
        for mod, lay, what in layers(a.request_id):
            print(f"{mod:7} {lay:16} {what}")
    else:
        show_map()


if __name__ == "__main__":
    main()
