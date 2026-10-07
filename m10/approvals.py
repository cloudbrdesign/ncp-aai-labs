"""The reviewer's side: the queue, the card, and the decision (lessons 10.1 and 10.4). The web page calls the
same functions through oversight_desk.py --serve.

    python m10/approvals.py pending                              # sorted by criticality, with waiting time
    python m10/approvals.py show t-3f2a9c1e                      # one card: facts, proposal, allowed decisions
    python m10/approvals.py approve t-3f2a9c1e --reviewer "Ana"
    python m10/approvals.py edit t-3f2a9c1e --amount 94.50 --reviewer "Ana"
    python m10/approvals.py reject t-3f2a9c1e --message "Please send the dock back first." --reviewer "Ana"
    python m10/approvals.py expire --older-than 30               # stale cards -> rejection + an M2 ticket
    python m10/approvals.py log                                  # every approval record (who, what, how long)

decide() is the one way a decision reaches a card:
  - the card must still be pending (read from the thread's checkpoint), the reviewer must give a name,
    the decision must be one the card allows (approve, edit, reject); a rejection needs a message the
    customer can read; an edited amount is validated against what can be refunded BEFORE anything resumes
  - dual approval: a refund above refunds.DUAL_APPROVAL_ABOVE needs two different reviewers who agree on the
    amount; the first approval is stored (state/approvals.db) and the card stays pending until the second
  - then Command(resume=...) on the thread, with the decision in the middleware's shape
    ({"decisions": [{"type": "approve"} | {"type": "edit", "edited_action": {...}} |
    {"type": "reject", "message": ...}]}) plus the reviewers' names
  - one M9 audit record per decision (action "approval": reviewer, decision, proposed vs final arguments,
    wait seconds, the refund row, the same request and trace ID as the customer's turn), and one
    `reviewer_decision` record in the feedback store: a second channel, kept apart from the thumbs

Waiting time is the time from the card's creation to the decision. expire turns cards that waited too long
into a rejection with a message and a ticket on M2's ticket API, so nobody waits forever on a person.
"""
import argparse
import json
import sys
import time

import bootstrap  # noqa: F401  (first: state folder, fake server, the v9 desk)
from bootstrap import VERSION, audit_log, escalate, gd

import oversight_desk as ov
import refunds

SCORES = {"approve": 1.0, "edit": 0.5, "reject": 0.0}


class NothingPending(Exception):
    """The thread has no pending card (decided, expired, or never had one)."""


class InvalidDecision(Exception):
    """The decision is not allowed or its arguments are invalid; nothing was resumed."""


def wait_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def partials(card_id: str) -> list[dict]:
    with ov.cards_db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM partials WHERE card_id = ? ORDER BY ts", (card_id,))]


def pending(od: "ov.OversightDesk") -> list[dict]:
    """The queue: every card whose thread still waits on review(), most critical first
    (two approvers needed, then the larger amount, then the oldest)."""
    with ov.cards_db() as con:
        threads = [dict(r) for r in con.execute("SELECT * FROM cards WHERE status = 'pending'")]
    out, now = [], time.time()
    for t in threads:
        payload = od.pending(t["thread_id"])
        if payload is None or payload["card"]["card_id"] != t["card_id"]:
            with ov.cards_db() as con:                       # the checkpoint is the truth; fix the index
                con.execute("UPDATE cards SET status = 'gone' WHERE card_id = ?", (t["card_id"],))
            continue
        card = payload["card"]
        out.append({**card, "args": payload["action_requests"][0]["args"],
                    "description": payload["action_requests"][0]["description"],
                    "allowed_decisions": payload["review_configs"][0]["allowed_decisions"],
                    "args_schema": payload["review_configs"][0].get("args_schema"),
                    "wait_s": round(now - card["created_ts"], 1), "approvals_so_far": partials(card["card_id"])})
    out.sort(key=lambda c: (-c["required_approvals"], -c["args"]["amount"], c["created_ts"]))
    return out


def decide(od: "ov.OversightDesk", thread_id: str, decision: str, reviewer: str, amount=None, message=None,
           guard: bool = True, card_status: str = "decided") -> dict:
    """Apply one reviewer's decision to the thread's pending card. See the module docstring.
    guard=False skips the "still pending?" check: drill.py uses it to show that the ledger's idempotency
    key alone stops a double payment when two resumes get through."""
    payload = od.pending(thread_id)
    if payload is None:
        if guard:
            raise NothingPending(f"nothing pending on thread {thread_id} (decided, expired, or no card)")
        payload = od.app.get_state(od.config(thread_id)).values.get("payload")
        if not payload:
            raise NothingPending(f"thread {thread_id} never had a card")
    card, args = payload["card"], dict(payload["action_requests"][0]["args"])
    allowed = payload["review_configs"][0]["allowed_decisions"]
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise InvalidDecision("a reviewer name is required (who approved what is part of the record)")
    if decision not in allowed:
        raise InvalidDecision(f"decision must be one of {allowed}, got {decision!r}")
    final = dict(args)
    if decision == "edit":
        why = refunds.validate_amount(amount, card["order"]["max_amount"])
        if why:
            raise InvalidDecision(f"edit refused: {why}")
        final["amount"] = float(amount)
    if decision == "reject" and not (message or "").strip():
        raise InvalidDecision("a rejection needs a message the customer can read")
    now = time.time()
    reviewers = [reviewer]
    if decision in ("approve", "edit") and refunds.needs_two(final["amount"]):
        earlier = partials(card["card_id"])
        if any(p["reviewer"] == reviewer for p in earlier):
            raise InvalidDecision(f"{reviewer} already approved this card; a refund above "
                                  f"{refunds.money(refunds.DUAL_APPROVAL_ABOVE)} needs a second, different person")
        with ov.cards_db() as con:
            con.execute("INSERT INTO partials VALUES (?,?,?,?,?)", (card["card_id"], reviewer, decision, final["amount"], now))
        agree = [p for p in earlier if abs(p["amount"] - final["amount"]) < 0.005]
        if not agree:
            out = {"status": "waiting for a second approver", "card_id": card["card_id"], "thread_id": thread_id,
                   "decision": decision, "reviewers": reviewers, "final_args": final,
                   "wait_s": round(now - card["created_ts"], 1)}
            _audit(od, card, args, final, decision, reviewers, message, out, partial=True)
            return out
        reviewers = [agree[0]["reviewer"], reviewer]
    if decision == "approve":
        dec = {"type": "approve"}
    elif decision == "edit":
        dec = {"type": "edit", "edited_action": {"name": "issue_refund", "args": final}}
    else:
        dec = {"type": "reject", "message": message.strip()}
    res = od.resume(thread_id, {"decisions": [dec], "reviewers": reviewers})
    out = {"status": res["status"], "card_id": card["card_id"], "thread_id": thread_id, "decision": decision,
           "reviewers": reviewers, "final_args": final if decision != "reject" else None,
           "refund": res.get("refund"), "ticket_id": res.get("ticket_id"), "reply": res["reply"],
           "wait_s": round(now - card["created_ts"], 1)}
    with ov.cards_db() as con:
        con.execute("UPDATE cards SET status = ?, decided_ts = ?, decision = ? WHERE card_id = ?",
                    (card_status, now, decision, card["card_id"]))
    _audit(od, card, args, final, decision, reviewers, message, out)
    import feedback_loop
    feedback_loop.record_reviewer(card, decision, reviewers, args, out["final_args"], message, out)
    return out


def _audit(od, card: dict, proposed: dict, final: dict, decision: str, reviewers: list[str], message, out: dict,
           partial: bool = False) -> dict:
    refund = out.get("refund") or {}
    tool = {"tool": "issue_refund", "params": final if decision != "reject" else proposed,
            "result": (refund.get("status") or ("not run (rejected)" if decision == "reject" else "not run yet"))
            + (f" {refund.get('refund_id')}" if refund.get("refund_id") else "")
            + (f": {refund.get('why')}" if refund.get("why") else "")}
    return audit_log.write({
        "request_id": card["request_id"], "trace_id": card["trace_id"], "config_version": audit_log.config_version(),
        "desk_version": VERSION, "model": None, "layer": None, "hosted": None, "caller": card["customer"],
        "scope": f"orders of {card['customer']}", "session_id": card["thread_id"],
        "input": f"refund card {card['card_id']}: {card['customer_message']}", "action": "approval",
        "tool_calls": [] if partial else [tool], "reply": out.get("reply"),
        "approval": {"card_id": card["card_id"], "thread_id": card["thread_id"], "decision": decision,
                     "reviewer": reviewers[-1], "reviewers": reviewers, "partial": partial,
                     "required_approvals": card["required_approvals"], "proposed_args": proposed,
                     "final_args": None if decision == "reject" else final, "message": message or "",
                     "wait_s": out["wait_s"], "outcome": out["status"], "refund_id": refund.get("refund_id"),
                     "ticket_id": out.get("ticket_id"), "proposal_source": card["proposal"]["source"]}})


def expire(od: "ov.OversightDesk", older_than_min: float, by: str = "system: expiry") -> list[dict]:
    """Cards older than N minutes: a ticket on M2's ticket API, then a rejection that tells the customer so."""
    done = []
    for c in pending(od):
        if c["wait_s"] < older_than_min * 60:
            continue
        tid, err = escalate.open_ticket(f"Refund card {c['card_id']} expired after {wait_text(c['wait_s'])} without a "
                                        f"decision | order {c['order']['order_id']} | {refunds.money(c['args']['amount'])} "
                                        f"| thread {c['thread_id']} | request {c['request_id']}", c["order"]["order_id"])
        msg = (f"Your request waited more than {older_than_min:g} minutes for a review, so it has gone to a colleague "
               f"as ticket {tid or '(pending: ' + err + ')'}; they will contact you.")
        out = decide(od, c["thread_id"], "reject", by, message=msg, card_status="expired")
        done.append({**out, "ticket_id": tid})
    return done


def show_card(c: dict) -> None:
    o = c["order"]
    print(f"card {c['card_id']}  thread {c['thread_id']}  request {c['request_id']}  waiting {wait_text(c['wait_s'])}")
    print(f"  customer   {c['customer']}: {c['customer_message']!r}")
    print(f"  order      {o['order_id']}: {o['qty']} x {o['item']} ({o['product']}), {o['status']}"
          + (f" {o['delivered']}" if o.get("delivered") else "") + f"; value {refunds.money(o['order_value'])}, "
          f"already refunded {refunds.money(o['already_refunded'])}")
    print(f"  policy     {c['policy']}")
    print(f"  proposal   issue_refund {json.dumps(c['args'], ensure_ascii=False)}  (by the {c['proposal']['source']})")
    for chk in c["checks"]:
        print(f"  CHECK      {chk}")
    print(f"  allowed    {', '.join(c['allowed_decisions'])}; amount must be > 0 and <= {refunds.money(o['max_amount'])}")
    need = c["required_approvals"]
    print(f"  approvers  {need} needed" + (f" (above {refunds.money(c['dual_threshold'])})" if need == 2 else "")
          + (f"; so far: {', '.join(p['reviewer'] for p in c['approvals_so_far'])}" if c["approvals_so_far"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pending", help="the queue, most critical first")
    s = sub.add_parser("show", help="one card")
    s.add_argument("thread")
    for name in ("approve", "edit", "reject"):
        p = sub.add_parser(name, help=f"{name} the thread's pending refund")
        p.add_argument("thread")
        p.add_argument("--reviewer", required=True)
        if name == "edit":
            p.add_argument("--amount", type=float, required=True)
        if name == "reject":
            p.add_argument("--message", required=True)
    e = sub.add_parser("expire", help="stale cards -> rejection + ticket")
    e.add_argument("--older-than", type=float, required=True, metavar="MIN")
    sub.add_parser("log", help="every approval record in the audit log")
    a = ap.parse_args()
    od = ov.OversightDesk()
    try:
        if a.cmd == "pending":
            cards = pending(od)
            st = refunds.stopped()
            if st:
                print(f"[stop] REFUNDS STOPPED since {st['time']}: {st['reason']} ({st['by']})")
            for c in cards:
                print(f"{c['thread_id']:<26} {c['card_id']}  {c['order']['order_id']}  {c['customer']:<9} "
                      f"{refunds.money(c['args']['amount']):>12}  approvers {c['required_approvals']}"
                      f"{' (1 so far)' if c['approvals_so_far'] else ''}  waiting {wait_text(c['wait_s'])}")
            print(f"[queue] {len(cards)} pending card{'s' if len(cards) != 1 else ''}")
        elif a.cmd == "show":
            c = next((c for c in pending(od) if c["thread_id"] == a.thread), None)
            if not c:
                sys.exit(f"[queue] nothing pending on thread {a.thread}")
            show_card(c)
        elif a.cmd in ("approve", "edit", "reject"):
            try:
                out = decide(od, a.thread, a.cmd, a.reviewer, amount=getattr(a, "amount", None),
                             message=getattr(a, "message", None))
            except (NothingPending, InvalidDecision) as err:
                sys.exit(f"[refused] {err}")
            print(f"[{a.cmd}] card {out['card_id']} by {' + '.join(out['reviewers'])} after {wait_text(out['wait_s'])}: "
                  f"{out['status']}" + (f", refund {out['refund'].get('refund_id')}" if (out.get('refund') or {}).get('refund_id') else ""))
            if out.get("reply"):
                print("\n" + gd.desk_graph.wrap(f"Reply to the customer: {out['reply']}"))
        elif a.cmd == "expire":
            done = expire(od, a.older_than)
            for d in done:
                print(f"[expire] {d['card_id']} ({d['thread_id']}) after {wait_text(d['wait_s'])} -> rejected, ticket {d['ticket_id']}")
            print(f"[expire] {len(done)} card{'s' if len(done) != 1 else ''} expired")
        else:
            recs = audit_log.query(action="approval")
            for r in recs:
                ap_ = r.get("approval") or {}
                print(f"{r['time']}  {r['request_id']}  {ap_.get('card_id')}  {ap_.get('decision'):<7} by "
                      f"{' + '.join(ap_.get('reviewers') or []):<20} wait {wait_text(ap_.get('wait_s') or 0):>8}  "
                      f"{ap_.get('outcome')}" + (f"  {json.dumps(ap_.get('final_args'))}" if ap_.get("final_args") else ""))
            print(f"[audit] {len(recs)} approval record{'s' if len(recs) != 1 else ''}")
    finally:
        od.close()


if __name__ == "__main__":
    main()
