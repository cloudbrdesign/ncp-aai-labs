"""The refund tool: eligibility, the amount, the ledger, and the stop switch (lesson 10.4).

    python m10/refunds.py check A1002 --customer "Tom B."     # eligible? how much at most?
    python m10/refunds.py ledger                              # every refund issued
    python m10/refunds.py stop --on --reason "payment provider incident" --by "Ana"
    python m10/refunds.py stop --off

eligibility()   M4's return_check (the 30-day rule on the SQL row) run in the caller's identity scope from
                Module 9, so another customer's order is "not on your account". The most that can be
                refunded is unit price (data/prices.json, fictional) x quantity, minus what the ledger has
                already refunded for that order. Ineligible requests are refused here, before any person is
                asked: a rule decides them, so they don't spend reviewer attention (criticality).
needs_two()     refunds above DUAL_APPROVAL_ABOVE need two different approvers (dual approval for high impact).
issue_refund()  the side effect. It writes one row to state/refunds.db and is idempotent: the key is
                thread + order (UNIQUE), so a second call for the same conversation and order returns the
                first row instead of paying twice. It also refuses an amount above what is left for the
                order, and does nothing while the stop switch is on.
stop switch     state/stop.json, read before every side effect (as M8's fault file is read on every tool
                call). On: no refund runs; the desk tells the customer a colleague will follow up.
"""
import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import sqlite3
import sys
import threading
import uuid

import bootstrap
from bootstrap import gd

HERE = bootstrap.HERE
STATE = bootstrap.STATE
PRICES = json.loads((HERE / "data" / "prices.json").read_text())
CURRENCY = PRICES["currency"]
DUAL_APPROVAL_ABOVE = 250.0          # EUR: one refund above this needs two approvers
LEDGER = pathlib.Path(os.environ.get("M10_LEDGER", STATE / "refunds.db"))   # drill.py: one ledger per drill
STOP_FILE = STATE / "stop.json"
_lock = threading.Lock()
desk_steps = gd.desk_steps           # M4's step handlers (return_check)
orders_sql = gd.orders_sql           # M4's SQL tool, with M9's identity scope on every connection


def money(x: float) -> str:
    return f"{CURRENCY} {x:,.2f}"


# ---- the stop switch -----------------------------------------------------------------------------

def stopped() -> dict | None:
    """The stop switch's state, or None when refunds may run."""
    try:
        s = json.loads(STOP_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None
    return s if s.get("on") else None


def set_stop(on: bool, reason: str = "", by: str = "") -> dict:
    s = {"on": on, "reason": reason, "by": by, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    if on:
        STOP_FILE.write_text(json.dumps(s))
    else:
        STOP_FILE.unlink(missing_ok=True)
    return s


# ---- the ledger ----------------------------------------------------------------------------------

@contextlib.contextmanager
def ledger():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(LEDGER, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS refunds (
        refund_id       TEXT PRIMARY KEY,
        order_id        TEXT NOT NULL,
        customer        TEXT NOT NULL,
        amount          REAL NOT NULL CHECK (amount > 0),
        reason          TEXT,
        approved_by     TEXT NOT NULL,
        decision        TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        thread_id       TEXT,
        request_id      TEXT,
        created_at      TEXT NOT NULL)""")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def rows(order_id: str | None = None, request_id: str | None = None) -> list[dict]:
    with ledger() as con:
        q, args = "SELECT * FROM refunds WHERE 1=1", []
        if order_id:
            q, args = q + " AND order_id = ?", args + [order_id]
        if request_id:
            q, args = q + " AND request_id = ?", args + [request_id]
        return [dict(r) for r in con.execute(q + " ORDER BY created_at", args)]


def refunded(order_id: str) -> float:
    return round(sum(r["amount"] for r in rows(order_id)), 2)


# ---- eligibility ---------------------------------------------------------------------------------

def order_facts(order_id: str, customer: str) -> dict | None:
    """The order as the caller may see it (M9's scope): status, product, quantity. None if not theirs."""
    token = gd._caller.set((customer or "", True))
    try:
        row = orders_sql.order_status(order_id)
        if not row:
            return None
        with contextlib.closing(orders_sql.connect()) as con:
            qty = con.execute("SELECT qty FROM order_items WHERE order_id = ?", (order_id,)).fetchone()["qty"]
        return {**row, "qty": int(qty)}
    finally:
        gd._caller.reset(token)


def eligibility(order_id: str, customer: str) -> dict:
    """Can this caller get a refund for this order, and how much at most? A rule, no model."""
    order_id = order_id.upper()
    out = {"order_id": order_id, "customer": customer, "eligible": False}
    facts = order_facts(order_id, customer)
    if facts is None:
        return {**out, "reason": f"order {order_id} is not on your account"}
    token = gd._caller.set((customer or "", True))
    try:
        check = desk_steps.return_check({"order_id": order_id}, "")        # M4's 30-day rule
    finally:
        gd._caller.reset(token)
    unit = float(PRICES["unit_price"][facts["product"]])
    total = round(unit * facts["qty"], 2)
    done = refunded(order_id)
    out.update(item=facts["item"], product=facts["product"], qty=facts["qty"], unit_price=unit, order_value=total,
               already_refunded=done, max_amount=round(total - done, 2), status=facts["status"],
               delivered=facts.get("delivered"), policy=check)
    if not check.get("returnable"):
        return {**out, "reason": check.get("reason") or check.get("error", "not returnable")}
    if out["max_amount"] <= 0:
        return {**out, "reason": f"order {order_id} has already been refunded in full ({money(done)})"}
    return {**out, "eligible": True,
            "reason": f"delivered {check['delivered_days_ago']} days ago, inside the "
                      f"{desk_steps.RETURN_DAYS}-day window ({check['days_left']} days left)"}


def needs_two(amount: float) -> bool:
    return float(amount) > DUAL_APPROVAL_ABOVE


def validate_amount(amount, max_amount: float) -> str | None:
    """Why an amount can't be refunded, or None. Used for edited arguments before anything resumes."""
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return f"amount must be a number, got {amount!r}"
    if a <= 0:
        return "amount must be more than 0"
    if round(a, 2) != a:
        return "amount has more than two decimals"
    if a > max_amount + 1e-9:
        return f"amount {money(a)} is more than the {money(max_amount)} that can be refunded"
    return None


# ---- the side effect -----------------------------------------------------------------------------

def issue_refund(thread_id: str, order_id: str, customer: str, amount: float, reason: str, approved_by: str,
                 decision: str, request_id: str = "") -> dict:
    """Write one refund row, exactly once per thread + order. Returns what happened:
    issued | duplicate (the row from the first call) | refused (amount above what is left) | stopped."""
    s = stopped()
    if s:
        return {"status": "stopped", "stop": s}
    key = f"{thread_id}:{order_id}"
    with _lock, ledger() as con:
        first = con.execute("SELECT * FROM refunds WHERE idempotency_key = ?", (key,)).fetchone()
        if first:
            return {"status": "duplicate", **dict(first)}
        done = sum(r[0] for r in con.execute("SELECT amount FROM refunds WHERE order_id = ?", (order_id,)))
        facts = order_facts(order_id, customer)
        if facts is None:
            return {"status": "refused", "why": f"order {order_id} is not on {customer}'s account"}
        left = round(float(PRICES["unit_price"][facts["product"]]) * facts["qty"] - done, 2)
        why = validate_amount(amount, left)
        if why:
            return {"status": "refused", "why": why if left > 0 else f"order {order_id} is already refunded in full"}
        row = {"refund_id": "R-" + uuid.uuid4().hex[:8], "order_id": order_id, "customer": customer,
               "amount": float(amount), "reason": reason, "approved_by": approved_by, "decision": decision,
               "idempotency_key": key, "thread_id": thread_id, "request_id": request_id,
               "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        # INSERT OR IGNORE + the UNIQUE key: a second writer that got past the SELECT above (another
        # process, a double click) inserts nothing and gets the first row back.
        cur = con.execute(f"INSERT OR IGNORE INTO refunds ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                          list(row.values()))
        if cur.rowcount == 0:
            first = con.execute("SELECT * FROM refunds WHERE idempotency_key = ?", (key,)).fetchone()
            return {"status": "duplicate", **dict(first)}
    return {"status": "issued", **row}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="eligibility of one order for one customer")
    c.add_argument("order_id")
    c.add_argument("--customer", default="Tom B.")
    sub.add_parser("ledger", help="every refund issued")
    s = sub.add_parser("stop", help="the stop switch")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true")
    g.add_argument("--off", action="store_true")
    s.add_argument("--reason", default="")
    s.add_argument("--by", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        e = eligibility(a.order_id, a.customer)
        print(json.dumps({k: v for k, v in e.items() if k != "policy"}, indent=1, ensure_ascii=False))
    elif a.cmd == "ledger":
        rs = rows()
        for r in rs:
            print(f"{r['created_at']}  {r['refund_id']}  {r['order_id']}  {r['customer']:<9} {money(r['amount']):>12}  "
                  f"{r['decision']:<8} by {r['approved_by']}  key {r['idempotency_key']}")
        print(f"[ledger] {len(rs)} refund{'s' if len(rs) != 1 else ''} in {LEDGER.relative_to(bootstrap.LABS) if LEDGER.is_relative_to(bootstrap.LABS) else LEDGER}")
    else:
        st = set_stop(a.on, a.reason, a.by)
        print(f"[stop] refunds {'STOPPED' if a.on else 'running again'}" + (f": {a.reason}" if a.reason else "")
              + (f" (by {a.by})" if a.by else "") + f" at {st['time']}")
    gd.close()


if __name__ == "__main__":
    sys.exit(main())
