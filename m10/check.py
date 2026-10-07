"""Check that the M10 lab works (free mode: everything local, Ollama with llama3.2:3b and embeddinggemma).

    python m10/check.py                    # Ollama with llama3.2:3b and embeddinggemma
    M10_FAKE_LLM=1 python m10/check.py     # offline self-test (CI): scripted models

check.py works in its own state folder, m10/state/check/ (emptied at the start), and starts M2's ticket API
on a free port. The desk runs at layer L1 (rules only) here, so the checks don't depend on what a rail model
decides; the drills and the lessons use L4.

1.  Ollama is up and has llama3.2:3b and embeddinggemma (offline: the scripted server).
2.  Versions and imports: langchain 1.4.2 (HumanInTheLoopMiddleware, create_agent), langgraph (interrupt,
    Command, GraphOutput, invoke(version=...)), langgraph-checkpoint-sqlite (SqliteSaver), FastAPI, NAT's
    interactive prompt models; each version printed.
3.  Data: 3 prices; 12 feedback events that M6's schema accepts; 12 burst requests with every case.
4.  A refund request leaves a pending card: one interrupt with action_requests + review_configs, read back
    from get_state(...).tasks; nothing paid before the review (10.1).
5.  The card survives a restart (a new desk object on the same SQLite file), and a new message in that
    thread gets the holding reply.
6.  Approve executes once: one ledger row with the approved amount (or, if the 3B proposed an amount that
    can't be paid, no row and a ticket); the reply says what happened.
7.  Edit: an amount above what can be refunded is refused before anything resumes; half the amount is paid.
8.  Reject: refused without a message; with one, nothing is paid and the customer reads the message.
9.  Ineligible requests (outside the window, not delivered, someone else's order) never reach the queue.
10. Dual approval: above the threshold one approval keeps the card pending, the same person twice is
    refused, a second person releases it; the ledger names both.
11. Every decision has an M9 audit record (action approval: reviewer, wait seconds, proposed vs final
    arguments, the turn's request ID) and a reviewer_decision record in the feedback store.
12. Feedback from the page: a valid click is stored in M6's shape with its request ID; a reason outside
    M6's list and an unknown request ID are refused (HTTP 422).
13. Decision records: a manual answer (route, rails, retrieved vs cited chunks, M8's model calls) and a
    refund (card, reviewer, ledger row) are complete; the customer view has no prompts or staff names.
14. The page and its endpoints answer: /, /health, /v1/approvals, /v1/threads, /v1/records, /v1/stop,
    and a decision on a thread with nothing pending is HTTP 409.
15. hitl_middleware_demo.py: the middleware's interrupt payload, and approve / edit / reject / no-interrupt
    outcomes (offline: all asserted; with the 3B: whether it called the tool is reported).
16. drill.py --drill all: the card survives a killed server, a double resume pays once, nothing is paid
    while stopped and stale cards become tickets, the burst sorts refused-by-rule from queued; report.md.
17. feedback_loop.py: ingest, report, promote (M6's logic, needs_review), review, eval before, fix (FAQ
    passages in the index, v10.1), eval after, compare (flips and McNemar).
18. Coverage: every turn in the audit log has a complete decision record (100%).
"""
import json
import os
# Same pins as Module 9: macOS libomp, and the desk on the local 3B even when NVIDIA_API_KEY is set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LLM_PROVIDER", "ollama")
import pathlib
import shutil
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
CHECK_STATE = HERE / "state" / "check"
FAKE = os.environ.get("M10_FAKE_LLM") == "1"
LOG = CHECK_STATE / "check.log"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)
    with LOG.open("a") as f:
        f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail else "") + "\n")


def info(text):
    print(f"       [INFO] {text}", flush=True)
    with LOG.open("a") as f:
        f.write(f"       [INFO] {text}\n")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(args: list[str], timeout: float = 3600) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable] + args, cwd=LABS, capture_output=True, text=True, timeout=timeout, env=os.environ)
    with (CHECK_STATE / "scripts.log").open("a") as f:
        f.write(f"$ python {' '.join(args)}\n{p.stdout}\n{p.stderr[-3000:]}\n")
    return p


# ---- 1-3 ----------------------------------------------------------------------------------------

def ollama_ready() -> bool:
    if FAKE:
        info("offline self-test: M10's scripted server stands in for Ollama (M10_FAKE_LLM=1)")
        return True
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else "http://" + host
    try:
        names = {m["name"] for m in httpx.get(host.rstrip("/") + "/api/tags", timeout=5).json()["models"]}
    except Exception as e:
        check("Ollama answers", False, f"{type(e).__name__}: {e}. Start Ollama (ollama serve).")
        return False
    need = {"llama3.2:3b", "embeddinggemma:latest"}
    have = {n if ":" in n else n + ":latest" for n in names}
    check("Ollama has llama3.2:3b and embeddinggemma", need <= have, f"missing {sorted(need - have)}: ollama pull <name>")
    return need <= have


def versions() -> None:
    import inspect
    from importlib.metadata import version
    import langchain  # noqa: F401
    import langgraph  # noqa: F401
    from langchain.agents import create_agent  # noqa: F401
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    from langgraph.pregel import Pregel
    from langgraph.types import Command, GraphOutput, interrupt  # noqa: F401
    from nat.data_models.interactive import HumanPromptBinary, HumanPromptText  # noqa: F401
    vers = {p: version(p) for p in ("langchain", "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite",
                                    "langchain-ollama", "fastapi", "nemoguardrails", "nvidia-nat")}
    hitl = list(inspect.signature(HumanInTheLoopMiddleware.__init__).parameters)
    v2 = "version" in inspect.signature(Pregel.invoke).parameters
    check("versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()) + f"; HumanInTheLoopMiddleware{tuple(hitl[1:])}; "
          f"invoke(version=) {'yes' if v2 else 'NO'}",
          vers["langchain"] == "1.4.2" and v2 and "interrupt_on" in hitl, "pip install -r m10/requirements.txt")


def data_checks(refunds, feedback_loop) -> None:
    prices = refunds.PRICES["unit_price"]
    events = feedback_loop.load(feedback_loop.SCRIPT)
    ok_events = True
    for e in events:
        try:
            feedback_loop.m06_feedback.make_record(e["question"], {}, e["value"], e["reason"] or None, e["comment"],
                                                   e["correction"])
        except ValueError:
            ok_events = False
    burst = feedback_loop.load(refunds.HERE / "data" / "refund_burst.jsonl")
    whys = " ".join(b["why"] for b in burst)
    cases = all(w in whys for w in ("inside the 30-day window", "outside the window", "not delivered", "someone else's",
                                    "dual-approval"))
    check(f"data: prices {prices}; {len(events)} feedback events ({sum(e['value'] == 'down' for e in events)} down, "
          f"{sum(bool(e['correction']) for e in events)} with a correction); {len(burst)} burst requests "
          f"({sum(b['expect'] == 'queued' for b in burst)} expected in the queue)",
          set(prices) == {"H200", "D300", "M270"} and len(events) == 12 and ok_events and len(burst) == 12 and cases)


# ---- 4-14: the desk in this process -------------------------------------------------------------

def desk_checks(ov, approvals, refunds, audit_log, feedback_loop, decision_record) -> dict:
    od = ov.OversightDesk(layer="L1")
    info(f"model: {ov.gd.llm_calls.describe()}; desk {ov.VERSION}, {od.desk.describe()}")
    ask = "I want my money back for order A1002."

    out = od.chat(ask, customer="Tom B.", thread_id="check-approve")
    p = od.pending("check-approve") or {}
    ar, rc = (p.get("action_requests") or [{}])[0], (p.get("review_configs") or [{}])[0]
    check(f"pending card: status {out['status']}, interrupt payload keys {sorted(p)}; action {ar.get('name')} "
          f"{json.dumps(ar.get('args'))}; allowed {rc.get('allowed_decisions')}; ledger rows {len(refunds.rows())}",
          out["status"] == "pending" and ar.get("name") == "issue_refund" and rc.get("allowed_decisions") == ["approve", "edit", "reject"]
          and {"action_requests", "review_configs", "card"} <= set(p) and not refunds.rows(), json.dumps(out, default=str)[:400])
    if not FAKE and out.get("card"):
        info(f"the 3B proposed {out['card']['proposal']['amount']} for a {out['card']['order']['max_amount']} order "
             f"({out['card']['proposal']['source']}); checks: {out['card']['checks'] or 'none'}")

    od2 = ov.OversightDesk(layer="L1")              # a new process would do the same: a new connection, same file
    p2 = od2.pending("check-approve")
    hold = od2.chat("Any news?", customer="Tom B.", thread_id="check-approve")
    od2.conn.close()
    check(f"restart: a new desk object finds card {(p2 or {}).get('card', {}).get('card_id')}; a new message gets "
          f"{hold['action']!r}: {hold['reply'][:70]!r}",
          p2 and p2["card"]["card_id"] == p["card"]["card_id"] and hold["action"] == "holding" and hold["pending"])

    card = p["card"]
    r = approvals.decide(od, "check-approve", "approve", "Ana")
    rows = refunds.rows(request_id=card["request_id"])
    valid = not card["checks"]
    check(f"approve: {r['status']} after {r['wait_s']} s; ledger rows {[(x['amount'], x['approved_by']) for x in rows]}; "
          f"reply {r['reply'][:80]!r}",
          (valid and r["status"] == "refunded" and len(rows) == 1 and rows[0]["amount"] == card["proposal"]["amount"]
           and rows[0]["refund_id"] in r["reply"])
          or (not valid and r["status"] == "not_executed" and not rows and r.get("ticket_id")))

    e = od.chat("Please refund order A1004, both headsets.", customer="Marco D.", thread_id="check-edit")
    mx = e["card"]["order"]["max_amount"] if e.get("card") else 0
    try:
        approvals.decide(od, "check-edit", "edit", "Ana", amount=mx + 100)
        too_much = "accepted (WRONG)"
    except approvals.InvalidDecision as err:
        too_much = str(err)
    still = od.pending("check-edit") is not None
    half = round(mx / 2, 2)
    r = approvals.decide(od, "check-edit", "edit", "Ana", amount=half) if still else {"status": "-", "reply": ""}
    rows = refunds.rows(request_id=e["request_id"])
    check(f"edit: {mx + 100} refused ({too_much[:60]}), card still pending {still}; {half} -> {r['status']}, ledger "
          f"{[x['amount'] for x in rows]}",
          e.get("card") and still and "more than" in too_much and r["status"] == "refunded" and [x["amount"] for x in rows] == [half])

    refunds.LEDGER = CHECK_STATE / "reject.refunds.db"      # A1002 was refunded in full above: a fresh ledger
    j = od.chat(ask, customer="Tom B.", thread_id="check-reject")
    try:
        approvals.decide(od, "check-reject", "reject", "Ana")
        nomsg = "accepted (WRONG)"
    except approvals.InvalidDecision as err:
        nomsg = str(err)
    msg = "Please send the dock back first; we refund it when it arrives."
    r = approvals.decide(od, "check-reject", "reject", "Ana", message=msg)
    check(f"reject: without a message refused ({nomsg[:45]}); with one -> {r['status']}, ledger rows "
          f"{len(refunds.rows(request_id=j['request_id']))}, the reply quotes it {msg in r['reply']}",
          j.get("card") and "message" in nomsg and r["status"] == "rejected" and not refunds.rows(request_id=j["request_id"])
          and msg in r["reply"])

    inel = [("Aisha R.", "Refund order A1006 please."), ("Tom B.", "Refund my monitor order A1005."),
            ("Tom B.", "Give me my money back for order A1004.")]
    outs = [od.chat(t, customer=c) for c, t in inel]
    with ov.cards_db() as con:
        queued = [x[0] for x in con.execute("SELECT request_id FROM cards")]
    check("ineligible never queued: " + "; ".join(f"{o['status']} ({(o.get('reply') or '').rsplit(': ', 1)[-1][:45]})" for o in outs),
          all(o["status"] == "refused_by_rule" and not o.get("card") and o["request_id"] not in queued for o in outs))

    refunds.LEDGER = CHECK_STATE / "dual.refunds.db"        # A1004 was half refunded above: a fresh ledger
    d = od.chat("Please refund order A1004, both headsets.", customer="Marco D.", thread_id="check-dual")
    amount = d["card"]["proposal"]["amount"] if d.get("card") else 0
    first = approvals.decide(od, "check-dual", "approve", "Ana") if d.get("card") else {"status": "-"}
    same, second = "-", {"status": "-"}
    if refunds.needs_two(amount):
        try:
            approvals.decide(od, "check-dual", "approve", "Ana")
            same = "accepted (WRONG)"
        except approvals.InvalidDecision as err:
            same = str(err)
        second = approvals.decide(od, "check-dual", "approve", "Ben")
    rows = refunds.rows()
    if refunds.needs_two(amount):
        ok = (first["status"].startswith("waiting") and "second, different person" in same and second["status"] == "refunded"
              and rows and rows[0]["approved_by"] == "Ana + Ben")
    else:
        info(f"the model proposed {amount}, not above the {refunds.DUAL_APPROVAL_ABOVE} threshold: one approver was enough")
        ok = first["status"] == "refunded"
    check(f"dual approval for {amount}: Ana -> {first['status']}; Ana again -> refused ({same[:40]}); Ben -> "
          f"{second['status']}; ledger approved_by {[x['approved_by'] for x in rows]}", d.get("card") and ok)
    refunds.LEDGER = ov.STATE / "refunds.db"

    recs = [x for x in audit_log.query(request_id=card["request_id"]) if x.get("action") == "approval"]
    a = (recs[-1] if recs else {}).get("approval") or {}
    fb = [x for x in feedback_loop.records("reviewer") if x["request_id"] == card["request_id"]]
    check(f"approval audit record: reviewer {a.get('reviewer')}, decision {a.get('decision')}, wait {a.get('wait_s')} s, "
          f"proposed {a.get('proposed_args', {}).get('amount')} final {(a.get('final_args') or {}).get('amount')}; "
          f"reviewer_decision feedback records {len(fb)}",
          a.get("reviewer") == "Ana" and isinstance(a.get("wait_s"), (int, float)) and a.get("proposed_args")
          and recs[-1]["trace_id"] == card["trace_id"] and len(fb) == 1 and fb[0]["channel"] == "reviewer")

    from fastapi.testclient import TestClient
    m = od.chat("My D300 dock shows E42. What does it mean?", customer="Tom B.", thread_id="check-manual")
    with TestClient(ov.make_app(od)) as client:
        good = client.post("/v1/feedback", json={"request_id": m["request_id"], "value": "down", "reason": "wrong_fact",
                                                 "comment": "check", "correction": "Use the 130 W adapter."})
        bad = client.post("/v1/feedback", json={"request_id": m["request_id"], "value": "down", "reason": "rude"})
        unknown = client.post("/v1/feedback", json={"request_id": "nope", "value": "up"})
        g = good.json()
        check(f"page feedback: valid -> {good.status_code} {g.get('key')} {g.get('value')} {g.get('reason')} "
              f"(request {g.get('request_id')}, retrieved {g.get('retrieved_ids')}); bad reason -> {bad.status_code}; "
              f"unknown request -> {unknown.status_code}",
              good.status_code == 201 and g.get("key") == "user_rating" and g.get("request_id") == m["request_id"]
              and g.get("channel") == "user" and bad.status_code == 422 and unknown.status_code == 422)

        rec = decision_record.build(m["request_id"])
        ref = decision_record.build(card["request_id"])
        cust = decision_record.view(ref, "customer")
        text = json.dumps(cust)
        check(f"decision records: manual answer route {rec['route']}, retrieved {rec['retrieved']}, cited {rec['cited']}, "
              f"model calls {[c['workload'] for c in rec['model_calls']]}, complete {rec['complete']}; refund: card "
              f"{[c['status'] for c in ref['cards']]}, approvals {len(ref['approvals'])}, ledger {len(ref['ledger'])}, "
              f"complete {ref['complete']}; customer view {list(cust['why_this_answer'])}",
              rec["complete"] and rec["route"] and rec["model_calls"] and ref["complete"] and ref["approvals"]
              and "Ana" not in text and "prompt" not in text and (rec["retrieved"] or not FAKE), json.dumps(rec["missing"] + ref["missing"]))

        codes = {"/": client.get("/").status_code, "/health": client.get("/health").status_code,
                 "/v1/approvals": client.get("/v1/approvals").status_code,
                 "/v1/threads": client.get("/v1/threads/check-approve").status_code,
                 "/v1/records": client.get(f"/v1/records/{m['request_id']}?view=customer").status_code,
                 "/v1/stop": client.get("/v1/stop").status_code}
        page = client.get("/").text
        refunds.LEDGER = CHECK_STATE / "page.refunds.db"      # A1002 is refunded in the main ledger: a fresh one
        c = client.post("/v1/chat", json={"messages": [{"role": "user", "content": ask}]},
                        headers={"x-customer": "Tom B.", "x-thread-id": "check-http"})
        nothing = client.post("/v1/approvals/check-approve/decision", json={"decision": "approve", "reviewer": "Ana"})
        invalid = client.post("/v1/approvals/check-http/decision", json={"decision": "approve", "reviewer": ""})
        queue = client.get("/v1/approvals").json()["cards"]
        done = client.post("/v1/approvals/check-http/decision", json={"decision": "reject", "reviewer": "Ana",
                                                                     "message": "Checked by the lab check."})
    check(f"page: {codes}; chat -> x-pending {c.headers.get('x-pending')!r}, queue {[q['thread_id'] for q in queue]}; "
          f"decision with nothing pending {nothing.status_code}, without a reviewer {invalid.status_code}, reject {done.status_code}",
          all(v == 200 for v in codes.values()) and "Reviewer queue" in page and c.headers.get("x-pending")
          and "check-http" in [q["thread_id"] for q in queue] and nothing.status_code == 409 and invalid.status_code == 422
          and done.status_code == 200)
    refunds.LEDGER = ov.STATE / "refunds.db"
    od.close()                                     # frees the Milvus Lite file for the scripts
    return {"card": card}


# ---- 15-18: the scripts -------------------------------------------------------------------------

def script_checks(feedback_loop, decision_record) -> None:
    p = run([str(HERE / "hitl_middleware_demo.py")])
    demo = {r["name"]: r for r in feedback_loop.load(CHECK_STATE / "demo" / "demo.jsonl")}
    intr = "action_requests" in p.stdout and "review_configs" in p.stdout
    summary = {k: (v["interrupted"], v.get("refunds")) for k, v in demo.items()}
    if FAKE:
        ok = (summary.get("approve") == (True, [189.0]) and summary.get("edit") == (True, [94.5])
              and summary.get("reject") == (True, []) and summary.get("status", (None,))[0] is False
              and summary.get("when") == (False, [189.0]) and intr)
    else:
        ok = p.returncode == 0 and len(demo) == 5 and (intr or not any(v["interrupted"] for v in demo.values()))
        info(f"the 3B called a tool in {sum(bool(v['tool_calls']) for v in demo.values())} of 5 runs")
    check(f"hitl_middleware_demo: exit {p.returncode}; (interrupted, refunds) per run {summary}", ok, p.stderr[-600:])

    p = run([str(HERE / "drill.py"), "--drill", "all", "--layer", "L1", "--pace", "0"])
    out = CHECK_STATE / "oversight"
    get = lambda n: json.loads((out / f"{n}.json").read_text()) if (out / f"{n}.json").exists() else {}  # noqa: E731
    rs, dr, st, bu = get("restart"), get("double-resume"), get("stop"), get("burst")
    report = (out / "report.md").read_text() if (out / "report.md").exists() else ""
    burst_ok = bu.get("queued", 0) + bu.get("refused_by_rule", 0) + bu.get("other", 0) == 12 and bu.get("dual", 0) >= (1 if FAKE else 0)
    if FAKE:
        burst_ok = burst_ok and bu.get("matches_expectation") == 12
    else:
        info(f"burst: {bu.get('matches_expectation')}/12 as expected (the 3B's proposals decide the amounts)")
    check(f"drills: restart pending {rs.get('pending_before_kill')}->{rs.get('pending_after_restart')}, approve "
          f"{rs.get('status')}, rows {rs.get('ledger_rows')}; double resume {dr.get('refund_statuses')} rows "
          f"{dr.get('ledger_rows')}; stop: refunds while stopped {st.get('refunds_while_stopped')}, expired "
          f"{len(st.get('expired', []))}; burst queued {bu.get('queued')} refused {bu.get('refused_by_rule')}; report.md "
          f"{'written' if report else 'MISSING'}",
          p.returncode == 0 and rs.get("pending_after_restart") and rs.get("ledger_rows") == 1 and dr.get("ledger_rows") == 1
          and st.get("refunds_while_stopped") == 0 and st.get("expired") and burst_ok
          and "override rate" in report and "refunds executed while stopped | 0" in report, p.stderr[-800:])

    steps = [["ingest"], ["report"], ["promote"], ["review", "--accept-all", "--reviewer", "Ana"],
             ["eval", "--label", "before", "--reps", "1", "--limit", "3"], ["fix", "--reviewer", "Ana"],
             ["eval", "--label", "after", "--reps", "1", "--limit", "3"], ["compare"]]
    codes = [run([str(HERE / "feedback_loop.py")] + s).returncode for s in steps]
    cands = feedback_loop.load(feedback_loop.CANDIDATES)
    reviewed = feedback_loop.load(feedback_loop.REVIEWED)
    ver = json.loads(feedback_loop.bootstrap.VERSION_FILE.read_text()) if feedback_loop.bootstrap.VERSION_FILE.exists() else {}
    comp = (feedback_loop.EVAL / "compare.md").read_text() if (feedback_loop.EVAL / "compare.md").exists() else ""
    fields = {"id", "split", "category", "question", "answer", "keywords", "product", "ref_sections", "source", "needs_review"}
    check(f"feedback loop: exit codes {codes}; promoted {len(cands)} candidates ({[c['id'] for c in cands]}), reviewed "
          f"{len(reviewed)}; version {ver.get('version')} with {len(ver.get('faq_items', []))} FAQ passages; compare.md "
          f"{'has the table' if '| reviewed |' in comp else 'MISSING'}",
          codes == [0] * len(steps) and cands and all(fields <= set(c) and c["needs_review"] and c["source"] == "feedback" for c in cands)
          and reviewed and not any(r["needs_review"] for r in reviewed) and ver.get("version") == "v10.1"
          and "| reviewed |" in comp and "McNemar" in comp)
    run([str(HERE / "feedback_loop.py"), "fix", "--revert"])

    cov = decision_record.coverage(say=lambda *a: None)
    check(f"coverage: {cov['complete']}/{cov['turns']} turns with a complete decision record; {cov['cards']} refund cards, "
          f"{cov['decided']} decided", cov["turns"] > 30 and cov["share"] == 1.0, json.dumps(cov["incomplete"][:5]))


def main():
    shutil.rmtree(CHECK_STATE, ignore_errors=True)
    CHECK_STATE.mkdir(parents=True)
    LOG.write_text(f"m10 check {time.strftime('%Y-%m-%dT%H:%M:%S')}{' (fake model)' if FAKE else ''}\n")
    os.environ["M10_STATE_DIR"] = str(CHECK_STATE)
    port = free_port()
    os.environ["TICKET_API_URL"] = f"http://localhost:{port}"
    t0 = time.time()
    if not ollama_ready():
        return finish()
    versions()
    tickets = subprocess.Popen([sys.executable, str(LABS / "m02" / "ticket_api.py"), "--port", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        sys.path.insert(0, str(HERE))
        import bootstrap  # noqa: F401  (state folder, fake server, the v9 desk)
        import approvals
        import decision_record
        import feedback_loop
        import oversight_desk as ov
        import refunds
        data_checks(refunds, feedback_loop)
        desk_checks(ov, approvals, refunds, bootstrap.audit_log, feedback_loop, decision_record)
        script_checks(feedback_loop, decision_record)
    finally:
        tickets.terminate()
    info(f"total time {time.time() - t0:.0f} s; logs in m10/state/check/ (check.log, scripts.log)")
    return finish()


def finish():
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
