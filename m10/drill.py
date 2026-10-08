"""The oversight drill (lesson 10.4): decisions, a restart, a double resume, the stop switch, a burst.

    python m10/drill.py --drill all              # every drill, then state/oversight/report.md
    python m10/drill.py --drill decisions        # approve, edit (full -> half), reject with a message
    python m10/drill.py --drill restart          # kill the server with a card pending, restart, resume
    python m10/drill.py --drill double-resume    # the same approval twice: one ledger row
    python m10/drill.py --drill stop             # stop switch on: 0 refunds, a ticket, pending cards expire
    python m10/drill.py --drill burst            # 12 requests: refused by rule vs queued, dual approval
    python m10/drill.py --drill burst --pace 5   # the scripted reviewer takes 5 s per decision

The reviewer is scripted (names Ana and Ben, a fixed pace per decision), so the waiting times measure the
queue, not a person's attention. Each drill has its own ledger (state/oversight/<drill>/refunds.db) and
its own threads, so drills don't refund each other's orders; the audit log and the feedback store are the
desk's own (state/audit, state/feedback), so decision_record.py and feedback_loop.py report see the drills.

report.md: per drill what happened, then the oversight numbers: decisions by type, override rate (edited +
rejected / reviewed by a person, NIST GenAI Profile MS-4.2-004 "monitor human overrides"), waiting time p50 /
p95, double payments (must be 0), refunds executed while stopped (must be 0).
"""
import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
import uuid

import bootstrap
from bootstrap import LABS, STATE, gd

import approvals
import oversight_desk as ov
import refunds

OUT = STATE / "oversight"
DRILLS = ["restart", "decisions", "double-resume", "stop", "burst"]
BURST = bootstrap.HERE / "data" / "refund_burst.jsonl"
ASK = "I want my money back for order A1002."


def say(msg: str) -> None:
    print(msg, flush=True)


def fresh(name: str) -> None:
    """This drill's own ledger, and the stop switch off."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    refunds.LEDGER = d / "refunds.db"
    refunds.LEDGER.unlink(missing_ok=True)
    refunds.set_stop(False)


_made: list[str] = []          # the threads the current drill created (the report counts their decisions)


def tid(name: str) -> str:
    t = f"{name}-{uuid.uuid4().hex[:6]}"
    _made.append(t)
    return t


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_tickets() -> subprocess.Popen | None:
    """M2's ticket API takes the tickets (stop, expiry). Start one on a free port if none answers."""
    import httpx
    try:
        httpx.get(bootstrap.escalate.ticket_url() + "/health", timeout=2).raise_for_status()
        return None
    except httpx.HTTPError:
        port = free_port()
        os.environ["TICKET_API_URL"] = f"http://localhost:{port}"
        p = subprocess.Popen([sys.executable, str(LABS / "m02" / "ticket_api.py"), "--port", str(port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                httpx.get(f"http://localhost:{port}/health", timeout=1)
                say(f"[drill] M2 ticket API started on port {port}")
                return p
            except httpx.HTTPError:
                time.sleep(0.2)
        return p


def card_line(out: dict) -> str:
    c = out.get("card")
    if not c:
        return f"no card ({out['status']})"
    return (f"card {c['card_id']}: {refunds.money(c['proposal']['amount'])} for {c['order']['order_id']}, "
            f"proposed by the {c['proposal']['source']}, approvers {c['required_approvals']}")


# ---- the drills ---------------------------------------------------------------------------------

def decisions(od, pace: float) -> dict:
    fresh("decisions")
    rows = []
    plan = [("reject", "Tom B.", ASK, {"message": "Please send the dock back first; the refund follows when it arrives."}),
            ("approve", "Tom B.", "Please refund order A1002, the dock does not fit my desk.", {}),
            ("edit", "Marco D.", "Please refund order A1004, both headsets.", {"amount": "half"})]
    for decision, customer, text, kw in plan:
        out = od.chat(text, customer=customer, thread_id=tid("decisions"))
        say(f"[decisions] {customer}: {text!r} -> {card_line(out)}")
        if not out.get("card"):
            rows.append({"decision": decision, "status": out["status"], "card": None, "reply": out["reply"]})
            continue
        time.sleep(pace)
        if kw.get("amount") == "half":
            kw = {"amount": round(out["card"]["order"]["max_amount"] / 2, 2)}
        res = approvals.decide(od, out["thread_id"], decision, "Ana", **kw)
        led = refunds.rows(request_id=out["request_id"])
        say(f"[decisions]   {decision} by Ana -> {res['status']}; ledger rows {len(led)}"
            + (f" ({led[0]['amount']:.2f})" if led else "") + f"; reply: {res['reply'][:110]!r}")
        rows.append({"decision": decision, "status": res["status"], "card": out["card"]["card_id"],
                     "proposed": out["card"]["proposal"]["amount"], "final": (res.get("final_args") or {}).get("amount"),
                     "ledger": [r["amount"] for r in led], "wait_s": res["wait_s"], "reply": res["reply"],
                     "request_id": out["request_id"]})
    return {"drill": "decisions", "rows": rows}


def restart(layer: str, timeout: float = 600) -> dict:
    """The server is killed (SIGKILL) while a card is pending; a new server finds it in the checkpoint."""
    import httpx
    fresh("restart")
    gd.close()                                    # this process must not hold the Milvus Lite file
    port, log = free_port(), open(OUT / "restart" / "server.log", "w")
    env = {**os.environ, "M10_LEDGER": str(refunds.LEDGER)}
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, str(bootstrap.HERE / "oversight_desk.py"), "--serve", "--port", str(port), "--layer", layer]

    def start():
        p = subprocess.Popen(cmd, cwd=LABS, env=env, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                raise SystemExit(f"[restart] the server exited (see {OUT / 'restart' / 'server.log'})")
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    return p, time.time() - t0
            except httpx.HTTPError:
                time.sleep(0.5)
        p.kill()
        raise SystemExit("[restart] the server did not start in time")

    p, up1 = start()
    say(f"[restart] server up on port {port} in {up1:.0f} s")
    r = httpx.post(url + "/v1/chat", json={"messages": [{"role": "user", "content": ASK}]},
                   headers={"x-customer": "Tom B.", "x-thread-id": tid("restart")}, timeout=600)
    thread, card = r.headers.get("x-thread-id"), r.headers.get("x-pending")
    before = [c["card_id"] for c in httpx.get(url + "/v1/approvals", timeout=30).json()["cards"]]
    say(f"[restart] chat -> thread {thread}, pending card {card or 'NONE'}; queue {before}")
    p.kill()                                      # no shutdown, no cleanup: as if the machine lost power
    p.wait()
    say(f"[restart] server killed (exit {p.returncode}) with the card pending")
    p, up2 = start()
    after = httpx.get(url + "/v1/approvals", timeout=30).json()["cards"]
    found = next((c for c in after if c["card_id"] == card), None)
    say(f"[restart] new server up in {up2:.0f} s; card {card} {'still pending, waiting ' + str(found['wait_s']) + ' s' if found else 'NOT FOUND'}")
    res = httpx.post(f"{url}/v1/approvals/{thread}/decision", json={"decision": "approve", "reviewer": "Ana"}, timeout=120)
    th = httpx.get(f"{url}/v1/threads/{thread}", timeout=30).json()
    p.terminate()
    p.wait()
    led = refunds.rows()
    say(f"[restart] approve after the restart -> HTTP {res.status_code} {res.json().get('status')}; ledger rows {len(led)}; "
        f"reply: {str(th.get('reply'))[:100]!r}")
    return {"drill": "restart", "thread": thread, "card": card, "pending_before_kill": card in before,
            "pending_after_restart": bool(found), "decision_http": res.status_code, "status": res.json().get("status"),
            "ledger_rows": len(led), "server_start_s": [round(up1, 1), round(up2, 1)], "reply": th.get("reply")}


def double_resume(od) -> dict:
    """Two resumes of the same card get through at once (the API's pending check is skipped, as if two
    clicks raced past it): the ledger's idempotency key still allows one row. A third, normal decision
    is refused because nothing is pending any more."""
    fresh("double-resume")
    out = od.chat(ASK, customer="Tom B.", thread_id=tid("double"))
    say(f"[double-resume] {card_line(out)}")
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(approvals.decide, od, out["thread_id"], "approve", who, guard=False) for who in ("Ana", "Ben")]
        res = [f.result() for f in futs]
    try:
        approvals.decide(od, out["thread_id"], "approve", "Ana")
        third = "accepted (WRONG)"
    except approvals.NothingPending as e:
        third = f"refused: {e}"
    led = refunds.rows(request_id=out["request_id"])
    statuses = [(r.get("refund") or {}).get("status") for r in res]
    say(f"[double-resume] two concurrent resumes -> refund statuses {statuses}; ledger rows {len(led)}; "
        f"a third decision: {third}")
    return {"drill": "double-resume", "resumes": 2, "refund_statuses": statuses, "ledger_rows": len(led),
            "third": third, "keys": [r["idempotency_key"] for r in led]}


def stop(od) -> dict:
    fresh("stop")
    a = od.chat(ASK, customer="Tom B.", thread_id=tid("stop"))
    b = od.chat("Please refund order A1004, both headsets.", customer="Marco D.", thread_id=tid("stop"))
    say(f"[stop] two cards before the stop: {card_line(a)}; {card_line(b)}")
    refunds.set_stop(True, "drill: payment provider incident", "Ana")
    n0, t0 = len(refunds.rows()), time.time()
    say("[stop] STOP switch on (state/stop.json)")
    out = {"drill": "stop"}
    if a.get("card"):
        r = approvals.decide(od, a["thread_id"], "approve", "Ana")
        out["approved_while_stopped"] = {"status": r["status"], "ticket": r.get("ticket_id"), "reply": r["reply"]}
        say(f"[stop] approve while stopped -> {r['status']}, ticket {r.get('ticket_id')}; reply: {r['reply'][:120]!r}")
    c = od.chat("Can you refund order A1002 now?", customer="Tom B.", thread_id=tid("stop"))
    out["new_request"] = {"status": c["status"], "card": bool(c.get("card")), "ticket": c.get("ticket_id"), "reply": c["reply"]}
    say(f"[stop] new request while stopped -> {c['status']}, card {'yes' if c.get('card') else 'no'}, ticket {c.get('ticket_id')}")
    exp = approvals.expire(od, 0)
    out["expired"] = [{"card": e["card_id"], "ticket": e["ticket_id"], "reply": e["reply"]} for e in exp]
    say(f"[stop] expire --older-than 0 -> {len(exp)} card(s) rejected with a ticket: {[e['ticket_id'] for e in exp]}")
    out["refunds_while_stopped"] = len(refunds.rows()) - n0
    out["stopped_for_s"] = round(time.time() - t0, 1)
    refunds.set_stop(False)
    say(f"[stop] refunds executed while stopped: {out['refunds_while_stopped']}; STOP switch off")
    return out


def burst(od, pace: float) -> dict:
    """12 requests. Rules refuse the ineligible ones; the rest wait in the queue, most critical first."""
    fresh("burst")
    reqs = [json.loads(x) for x in BURST.read_text().splitlines() if x.strip()]
    threads, rows = {}, []
    t0 = time.time()
    for b in reqs:
        out = od.chat(b["text"], customer=b["customer"], thread_id=tid(f"burst-{b['id']}"))
        got = "queued" if out.get("card") else ("refused" if out["status"] == "refused_by_rule" else f"desk {out['action']}")
        threads[out["thread_id"]] = b["id"]
        rows.append({"id": b["id"], "customer": b["customer"], "expect": b["expect"], "got": got,
                     "amount": out["card"]["proposal"]["amount"] if out.get("card") else None,
                     "why": (out.get("card") or {}).get("policy") or out["reply"].rsplit(": ", 1)[-1][:90]})
        say(f"[burst] {b['id']} {b['customer']:<9} {got:<8} " + (f"{refunds.money(rows[-1]['amount'])}" if rows[-1]["amount"] else rows[-1]["why"]))
    intake_s = time.time() - t0
    queue = approvals.pending(od)
    queue = [c for c in queue if c["thread_id"] in threads]
    say("[burst] queue, most critical first: " + ", ".join(f"{threads[c['thread_id']]} ({refunds.money(c['args']['amount'])}, "
                                                           f"{c['required_approvals']} approver{'s' if c['required_approvals'] > 1 else ''})" for c in queue))
    script = {"b02": [("approve", "Ana", {}), ("approve", "Ben", {})],
              "b11": [("reject", "Ana", {"message": "Order A1004 is already covered by the full refund you asked for."})],
              "b12": [("approve", "Ben", {})]}
    decided, t1 = [], time.time()
    for c in queue:
        bid = threads[c["thread_id"]]
        for decision, who, kw in script.get(bid, [("approve", "Ana", {})]):
            time.sleep(pace)
            try:
                r = approvals.decide(od, c["thread_id"], decision, who, **kw)
            except (approvals.NothingPending, approvals.InvalidDecision) as err:
                # e.g. the 3B proposed less than the dual threshold, so the first approval already decided it
                say(f"[burst] {bid} {decision} by {who}: not applied ({err})")
                continue
            say(f"[burst] {bid} {decision} by {who} after {approvals.wait_text(r['wait_s'])} -> {r['status']}"
                + (f": {r['refund'].get('why')}" if (r.get("refund") or {}).get("why") else ""))
            decided.append({"id": bid, "decision": decision, "reviewer": who, "status": r["status"], "wait_s": r["wait_s"],
                            "partial": r["status"].startswith("waiting")})
    review_s = time.time() - t1
    final = [d for d in decided if not d["partial"]]
    waits = sorted(d["wait_s"] for d in final)
    return {"drill": "burst", "requests": len(reqs), "rows": rows, "decisions": decided,
            "queued": sum(r["got"] == "queued" for r in rows), "refused_by_rule": sum(r["got"] == "refused" for r in rows),
            "other": sum(r["got"] not in ("queued", "refused") for r in rows),
            "matches_expectation": sum(r["got"] == r["expect"] for r in rows), "dual": sum(d["partial"] for d in decided),
            "wait_p50": pct(waits, 0.5), "wait_p95": pct(waits, 0.95), "intake_s": round(intake_s, 1),
            "decisions_per_min": round(len(decided) / (review_s / 60), 1) if review_s > 0 else None, "pace_s": pace,
            "ledger": [{k: r[k] for k in ("order_id", "amount", "approved_by")} for r in refunds.rows()]}


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))]


# ---- the report ---------------------------------------------------------------------------------

def report(layer: str) -> str:
    res = {d: json.loads((OUT / f"{d}.json").read_text()) for d in DRILLS if (OUT / f"{d}.json").exists()}
    from bootstrap import audit_log
    mine = {t for r in res.values() for t in r.get("threads", [])}
    recs = [r for r in audit_log.query(action="approval") if r["session_id"] in mine]
    final = [r["approval"] for r in recs if not r["approval"].get("partial")]
    human = [a for a in final if not a["reviewer"].startswith("system")]
    by = {k: sum(a["decision"] == k for a in human) for k in ("approve", "edit", "reject")}
    over = (by["edit"] + by["reject"]) / len(human) if human else None
    waits = sorted(a["wait_s"] for a in human)
    double = res.get("double-resume", {}).get("ledger_rows")
    lines = [f"# Oversight drill report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Desk {bootstrap.desk_version()}, layer {layer}, model {gd.llm_calls.model_name()}; reviewer scripted (Ana, Ben).", "",
             "| measure | value |", "|---|---|",
             f"| decisions by a person | {len(human)}: approve {by['approve']}, edit {by['edit']}, reject {by['reject']} |",
             f"| system expiries (stale card -> ticket) | {len(final) - len(human)} |",
             f"| override rate (edited + rejected / reviewed) | {'-' if over is None else f'{over:.0%}'} |",
             f"| waiting time p50 / p95 | {fmt_s(pct(waits, 0.5))} / {fmt_s(pct(waits, 0.95))} |",
             f"| double payments (double resume: ledger rows for one card) | {'-' if double is None else double - 1} |",
             f"| refunds executed while stopped | {res.get('stop', {}).get('refunds_while_stopped', '-')} |", ""]
    if "decisions" in res:
        lines += ["## Decisions", "", "| decision | outcome | proposed | final | ledger | reply |", "|---|---|---|---|---|---|"]
        for r in res["decisions"]["rows"]:
            lines.append(f"| {r['decision']} | {r['status']} | {r.get('proposed', '-')} | {r.get('final') or '-'} | "
                         f"{r.get('ledger') or 'no row'} | {str(r['reply'])[:90]} |")
        lines.append("")
    if "restart" in res:
        r = res["restart"]
        lines += ["## Restart", "", f"Card {r['card']} pending before the kill: {r['pending_before_kill']}; after the restart: "
                  f"{r['pending_after_restart']}; approve -> HTTP {r['decision_http']} {r['status']}; ledger rows {r['ledger_rows']}; "
                  f"server start {r['server_start_s'][0]} s and {r['server_start_s'][1]} s.", ""]
    if "double-resume" in res:
        r = res["double-resume"]
        lines += ["## Double resume", "", f"Two concurrent resumes: refund statuses {r['refund_statuses']}; ledger rows "
                  f"{r['ledger_rows']}; a third decision: {r['third']}.", ""]
    if "stop" in res:
        r = res["stop"]
        lines += ["## Stop switch", "", f"Approve while stopped: {r.get('approved_while_stopped', {}).get('status')} "
                  f"(ticket {r.get('approved_while_stopped', {}).get('ticket')}); new request: {r['new_request']['status']}, "
                  f"card {r['new_request']['card']}; expired with a ticket: {len(r['expired'])}; refunds executed while stopped: "
                  f"{r['refunds_while_stopped']}.", ""]
    if "burst" in res:
        r = res["burst"]
        lines += ["## Burst", "", f"{r['requests']} requests: {r['queued']} queued, {r['refused_by_rule']} refused by rule, "
                  f"{r['other']} other (as expected: {r['matches_expectation']}/{r['requests']}); dual approvals {r['dual']}; "
                  f"waiting p50 {fmt_s(r['wait_p50'])}, p95 {fmt_s(r['wait_p95'])}; {r['decisions_per_min']} decisions per minute "
                  f"at {r['pace_s']} s per decision.", "", "| id | customer | expected | got | amount / why |", "|---|---|---|---|---|"]
        for x in r["rows"]:
            lines.append(f"| {x['id']} | {x['customer']} | {x['expect']} | {x['got']} | "
                         f"{refunds.money(x['amount']) if x['amount'] else x['why']} |")
        lines += ["", "Decisions: " + "; ".join(f"{d['id']} {d['decision']} by {d['reviewer']} -> {d['status']}" for d in r["decisions"]), ""]
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text + "\n")
    return text


def fmt_s(v) -> str:
    return "-" if v is None else f"{v:.1f} s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drill", choices=DRILLS + ["all"], default="all")
    ap.add_argument("--layer", default="L4", choices=gd.layers.LAYERS)
    ap.add_argument("--pace", type=float, default=2.0, help="seconds the scripted reviewer takes per decision")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] model: {gd.llm_calls.describe()}; desk {bootstrap.desk_version()}, layer {a.layer}")
    tickets = ensure_tickets()
    todo = DRILLS if a.drill == "all" else [a.drill]
    od = None
    try:
        for d in todo:
            t0 = time.time()
            _made.clear()
            if d == "restart":
                res = restart(a.layer)
            else:
                od = od or ov.OversightDesk(layer=a.layer)
                res = {"decisions": lambda: decisions(od, a.pace), "double-resume": lambda: double_resume(od),
                       "stop": lambda: stop(od), "burst": lambda: burst(od, a.pace)}[d]()
            res["seconds"] = round(time.time() - t0, 1)
            res["threads"] = list(_made)
            (OUT / f"{d}.json").write_text(json.dumps(res, indent=1, default=str))
        print("\n" + report(a.layer))
    finally:
        refunds.set_stop(False)
        refunds.LEDGER = refunds.STATE / "refunds.db"
        if od:
            od.close()
        if tickets:
            tickets.terminate()


if __name__ == "__main__":
    main()
