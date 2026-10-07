"""A "why this answer" record for every reply, joined by one request ID (lesson 10.3).

    python m10/decision_record.py show --request-id 3f2a9c1e0b7d4e21                   # operator view
    python m10/decision_record.py show --request-id 3f2a9c1e0b7d4e21 --view customer
    python m10/decision_record.py coverage                       # share of turns with a complete record
    python m10/decision_record.py reasoning --request-id ...     # optional: qwen3:4b with thinking on

build() joins what the desk already writes, nothing new is logged:
  audit      Module 9's audit record of the turn (route, rails and their decisions, tool calls with
             parameters, the reply, config version) and the approval records (approvals.py) that carry the
             same request ID: reviewer(s), decision, proposed vs final arguments, wait seconds
  model      M8's request log (state/logstore/requests.jsonl): one line per model call with the workload
             (plan, draft, critique, sql, refund), the prompt and the reply; the critique's verdict is there
  evidence   retrieved chunk IDs (the manual_search tool call) vs the IDs the reply cites
  refund     the card (state/approvals.db) and the ledger row (state/refunds.db)
  feedback   both channels (state/feedback/feedback.jsonl) for this request or trace ID

view(customer) is the "Why this answer?" panel: the sources it cites, the steps in plain words, whether a
person approved. No prompts, no rail internals, no staff names. view(operator) is everything.

coverage(): every turn in the audit log must have its IDs, a reply, the config version, M8's model-call lines
when the desk called the model, and, when a refund card was decided, an approval record with a reviewer.

reasoning: a reasoning model's thinking is a narrative the model writes, not a trace of how it computed
the answer; the record above is what actually happened (the queries, sources, checks and people).
`reasoning` runs the turn's question once through qwen3:4b with thinking on, to show the two side by side.
"""
import argparse
import json
import re
import sys

import bootstrap  # noqa: F401  (first: state folder, fake server, the v9 desk)
from bootstrap import STATE, audit_log, gd

LOGSTORE = STATE / "logstore" / "requests.jsonl"
CITATION = gd.desk_graph.CITATION
PASSAGES = re.compile(r"^\d+ passages: (.*)$")
PLAIN = {"order_status": "looked up the order in the orders database",
         "return_check": "checked the order against the 30-day return rule",
         "manual_search": "searched the product manuals", "open_ticket": "opened a support ticket",
         "answer": "used the returns policy", "free_sql": "queried the orders database (your orders only)"}


def _load(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def retrieved_ids(turn: dict) -> list[str]:
    ids = []
    for t in turn.get("tool_calls") or []:
        m = PASSAGES.match(str(t.get("result") or ""))
        if t.get("tool") == "manual_search" and m:
            ids += [x.strip() for x in m.group(1).split(",") if x.strip()]
    return ids


def route_of(turn: dict) -> list[str]:
    """The desk's route (v10 keeps it in the audit record); for older records, the tool calls say the same."""
    if turn.get("route"):
        return turn["route"]
    return [f"{t.get('tool')} {' '.join(str(v) for v in (t.get('params') or {}).values())}".strip()
            for t in turn.get("tool_calls") or []]


def model_calls(request_id: str) -> list[dict]:
    out = []
    for r in _load(LOGSTORE):
        if r.get("request_id") != request_id:
            continue
        msgs = r["request"]["messages"]
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        out.append({"workload": r["workload_id"], "model": r["request"]["model"], "latency_s": r.get("latency_s"),
                    "prompt": user[-300:], "reply": r["response"]["choices"][0]["message"]["content"][:400],
                    "tokens": r["response"].get("usage", {}).get("total_tokens")})
    return out


def build(request_id: str) -> dict:
    import feedback_loop
    import oversight_desk as ov
    import refunds
    recs = audit_log.query(request_id=request_id)
    turn = next((r for r in recs if r.get("action") != "approval"), None)
    approvals = [r for r in recs if r.get("action") == "approval"]
    calls = model_calls(request_id)
    with ov.cards_db() as con:
        cards = [dict(r) for r in con.execute("SELECT * FROM cards WHERE request_id = ?", (request_id,))]
    trace = (turn or {}).get("trace_id")
    fb = [r for r in feedback_loop.records() if r.get("request_id") == request_id or (trace and r.get("trace_id") == trace)]
    critique = next((c["reply"] for c in calls if c["workload"] == "critique"), None)
    rec = {"request_id": request_id, "trace_id": trace, "audit": turn, "approvals": approvals, "model_calls": calls,
           "route": route_of(turn or {}), "retrieved": retrieved_ids(turn or {}),
           "cited": list(dict.fromkeys(CITATION.findall((turn or {}).get("reply") or ""))),
           "critique": critique, "cards": cards, "ledger": refunds.rows(request_id=request_id), "feedback": fb}
    rec["complete"], rec["missing"] = complete(rec)
    return rec


def complete(rec: dict) -> tuple[bool, list[str]]:
    t = rec["audit"]
    if not t:
        return False, ["audit record"]
    missing = [k for k in ("request_id", "trace_id", "config_version", "reply") if not t.get(k)]
    if (t.get("model_calls") or {}).get("desk", 0) and not rec["model_calls"]:
        missing.append("M8 request-log lines for the desk's model calls")
    for c in rec["cards"]:
        if c["status"] in ("decided", "expired"):
            final = [a for a in rec["approvals"] if (a.get("approval") or {}).get("card_id") == c["card_id"]
                     and not a["approval"].get("partial")]
            if not final or not final[-1]["approval"].get("reviewers"):
                missing.append(f"approval record with a reviewer for card {c['card_id']}")
    return not missing, missing


def view(rec: dict, who: str = "operator") -> dict:
    if who == "operator":
        return rec
    t = rec["audit"] or {}
    steps = []
    for tc in t.get("tool_calls") or []:
        steps.append(PLAIN.get(tc.get("tool"), tc.get("tool")))
    if (t.get("input_rails") or {}).get("rails") or (t.get("output_rails") or {}).get("rails"):
        steps.append("ran the safety checks on your message and the reply"
                     + (" (one of them stopped the reply)" if t.get("blocked_by") else ""))
    if rec["critique"]:
        steps.append("checked the draft against the sources before sending it")
    sections = {}
    for c in rec["cited"]:
        sections[c] = c.rsplit("-", 1)[0].replace("-", " ", 1)
    people = []
    for a in rec["approvals"]:
        ap = a.get("approval") or {}
        if ap.get("partial"):
            continue
        what = {"approve": "approved", "edit": "approved a changed amount for", "reject": "did not approve"}[ap["decision"]]
        n = len(ap.get("reviewers") or [])
        people.append(f"{'two colleagues' if n == 2 else 'a colleague'} {what} the refund"
                      + (f" ({ap['final_args']['amount']:.2f})" if ap.get("final_args") else ""))
    final = next((a.get("reply") for a in reversed(rec["approvals"]) if a.get("reply")), None)
    return {"request_id": rec["request_id"], "answer": final or t.get("reply"),
            "why_this_answer": {"sources": [{"id": c, "from": s} for c, s in sections.items()],
                                "steps": list(dict.fromkeys(steps)), "people": people,
                                "automated": "This reply was written by an AI assistant from the sources above."}}


def coverage(say=print) -> dict:
    turns = [r for r in audit_log.records() if r.get("action") != "approval"]
    bad = []
    for t in turns:
        rec = build(t["request_id"])
        if not rec["complete"]:
            bad.append((t["request_id"], rec["missing"]))
    import oversight_desk as ov
    with ov.cards_db() as con:
        cards = [dict(r) for r in con.execute("SELECT * FROM cards")]
    decided = [c for c in cards if c["status"] in ("decided", "expired")]
    share = (len(turns) - len(bad)) / len(turns) if turns else None
    say(f"[coverage] {len(turns) - len(bad)}/{len(turns)} turns with a complete record"
        + (f" ({share:.0%})" if share is not None else "") + f"; refund cards: {len(cards)} "
        f"({len(decided)} decided, every one with its reviewer: {'yes' if not any('approval' in ' '.join(m) for _, m in bad) else 'NO'}; "
        f"{sum(c['status'] == 'pending' for c in cards)} pending)")
    for rid, m in bad[:10]:
        say(f"[coverage] incomplete {rid}: missing {', '.join(m)}")
    return {"turns": len(turns), "complete": len(turns) - len(bad), "share": share, "incomplete": bad,
            "cards": len(cards), "decided": len(decided)}


def show(rec: dict, who: str, say=print) -> None:
    t = rec["audit"] or {}
    if who == "customer":
        v = view(rec, "customer")
        say(gd.desk_graph.wrap(f"Answer: {v['answer']}"))
        say("\nWhy this answer?")
        for s in v["why_this_answer"]["sources"]:
            say(f"  source  {s['id']}  ({s['from']})")
        for s in v["why_this_answer"]["steps"]:
            say(f"  step    {s}")
        for p in v["why_this_answer"]["people"]:
            say(f"  person  {p}")
        say(f"  note    {v['why_this_answer']['automated']}")
        return
    say(f"request {rec['request_id']}  trace {rec['trace_id']}  {t.get('time')}  desk {t.get('desk_version')} "
        f"({t.get('config_version')})  model {t.get('model')}  layer {t.get('layer')}")
    say(f"caller    {t.get('caller')} ({t.get('scope')}), session/thread {t.get('session_id')}")
    say(f"input     {t.get('input')}")
    say(f"route     {rec['route'] or '-'}")
    for kind in ("input_rails", "output_rails"):
        v = t.get(kind) or {}
        names = [f"{r['name']} ({'STOP' if r['stop'] else 'pass'})" for r in v.get("rails", [])]
        say(f"{kind[:-6]:<6}    {v.get('status')}: {', '.join(names) or 'none'}")
    for tc in t.get("tool_calls") or []:
        say(f"tool      {tc.get('tool')} {json.dumps(tc.get('params'), ensure_ascii=False)} -> {str(tc.get('result'))[:100]}")
    say(f"evidence  retrieved {rec['retrieved'] or '-'}; cited {rec['cited'] or '-'}"
        + (f"; cited but not retrieved: {sorted(set(rec['cited']) - set(rec['retrieved']))}"
           if set(rec["cited"]) - set(rec["retrieved"]) else ""))
    for c in rec["model_calls"]:
        say(f"model     {c['workload']:<9} {c['model']} {c['latency_s']} s, {c['tokens']} tokens -> {c['reply'][:110]!r}")
    for c in rec["cards"]:
        say(f"card      {c['card_id']} order {c['order_id']} proposed {c['amount']:.2f}, approvers {c['required']}, {c['status']}")
    for a in rec["approvals"]:
        ap = a.get("approval") or {}
        say(f"approval  {ap.get('decision')} by {' + '.join(ap.get('reviewers') or [])} after {ap.get('wait_s')} s"
            + (" (first of two)" if ap.get("partial") else f" -> {ap.get('outcome')}")
            + f"; proposed {json.dumps(ap.get('proposed_args', {}).get('amount'))}, final "
            + f"{json.dumps((ap.get('final_args') or {}).get('amount'))}" + (f"; message {ap['message']!r}" if ap.get("message") else ""))
    for r in rec["ledger"]:
        say(f"ledger    {r['refund_id']} {r['order_id']} {r['amount']:.2f} {r['decision']} by {r['approved_by']} "
            f"key {r['idempotency_key']}")
    for f in rec["feedback"]:
        say(f"feedback  {f['channel']}: {f['value']}" + (f" ({f['reason']})" if f.get("reason") else "")
            + (f", correction {f['correction'][:60]!r}" if f.get("correction") else ""))
    say(f"reply     {str(t.get('reply'))[:300]}")
    final = next((a.get("reply") for a in reversed(rec["approvals"]) if a.get("reply")), None)
    if final:
        say(f"after review {final[:300]}")
    say(f"[record] {'complete' if rec['complete'] else 'INCOMPLETE: missing ' + ', '.join(rec['missing'])}")


def reasoning(rec: dict, say=print) -> dict:
    """One call to qwen3:4b with thinking on (setup/llm.py switches reasoning on for qwen3)."""
    t = rec["audit"] or {}
    question = t.get("desk_input") or t.get("input")
    llm = gd.llm_calls.llm.get_llm(model="qwen3:4b", temperature=0)
    msg = llm.invoke([("system", "Answer the customer of an electronics shop in two sentences."), ("user", question)])
    think = (msg.additional_kwargs or {}).get("reasoning_content") or ""
    say(f"question  {question}")
    say(f"thinking  {think[:800] or '(no reasoning block returned)'}")
    say(f"answer    {msg.content[:300]}")
    say("[note] the thinking is text the model wrote; the decision record lists what the desk actually did")
    return {"thinking": think, "answer": msg.content}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--request-id", required=True)
    s.add_argument("--view", choices=["operator", "customer"], default="operator")
    s.add_argument("--json", action="store_true")
    sub.add_parser("coverage")
    r = sub.add_parser("reasoning")
    r.add_argument("--request-id", required=True)
    a = ap.parse_args()
    try:
        if a.cmd == "coverage":
            res = coverage()
            sys.exit(0 if res["share"] in (None, 1.0) else 1)
        rec = build(a.request_id)
        if not rec["audit"]:
            sys.exit(f"[record] no audit record for request {a.request_id}")
        if a.cmd == "reasoning":
            reasoning(rec)
        elif a.json:
            print(json.dumps(view(rec, a.view), ensure_ascii=False, indent=1, default=str))
        else:
            show(rec, a.view)
    finally:
        gd.close()


if __name__ == "__main__":
    main()
