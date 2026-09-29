"""The desk's step handlers: plain Python, no model.

    order_status   the M1 order data (m01/data/orders.json, the data behind lookup_order)
    return_check   the M1 returns policy: 30 days from delivery, refund within 5 business days
    open_ticket    the M2 ticket API (python m02/ticket_api.py)
    answer         a general question: the returns policy is the evidence

Each handler returns a small dict of facts. The draft node may only use these facts,
and the critique node checks the reply against them.
"""
import json
import os
import pathlib
import re

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORDERS = json.loads((ROOT / "m01" / "data" / "orders.json").read_text())
POLICY = (ROOT / "m01" / "data" / "returns_policy.md").read_text()
RETURN_DAYS = 30
TICKET_API = {"url": os.environ.get("TICKET_API_URL", "http://localhost:8765")}


class TicketApiError(RuntimeError):
    """The ticket API failed (HTTP error or not reachable). The graph run stops here."""


def order_status(order_id: str, request: str = "") -> dict:
    order = ORDERS.get(order_id)
    return {"order_id": order_id, **order} if order else {"order_id": order_id, "error": "no such order"}


def return_check(order_id: str, request: str = "") -> dict:
    order = ORDERS.get(order_id)
    if not order:
        return {"order_id": order_id, "error": "no such order"}
    if order["status"] != "delivered":
        return {"order_id": order_id, "item": order["item"], "returnable": False,
                "reason": f"not delivered yet ({order['status']}); it can be cancelled instead"}
    days = int(re.match(r"(\d+)", order["delivered"]).group(1))
    return {"order_id": order_id, "item": order["item"], "returnable": days <= RETURN_DAYS,
            "delivered_days_ago": days, "days_left": max(RETURN_DAYS - days, 0),
            "refund": "within 5 business days of the return arriving, to the original payment method"}


def open_ticket(order_id: str, request: str = "") -> dict:
    try:
        r = httpx.post(f"{TICKET_API['url']}/tickets", json={"order_id": order_id, "issue": request[:200]}, timeout=10)
    except httpx.HTTPError as e:
        raise TicketApiError(f"ticket API not reachable at {TICKET_API['url']} ({type(e).__name__}). "
                             "Start it: python m02/ticket_api.py") from e
    if r.status_code >= 400:
        raise TicketApiError(f"ticket API returned HTTP {r.status_code}")
    t = r.json()
    return {"order_id": order_id, "ticket_id": t["ticket_id"], "ticket_status": t["status"]}


def answer(order_id: str, request: str = "") -> dict:
    if re.search(r"where|when|arriv|status|shipped|track", request, re.I):   # about an order, but which one?
        return {"order_id": "", "need": "the order ID"}
    return {"order_id": "", "policy": " ".join(l[2:] for l in POLICY.splitlines() if l.startswith("- "))}


HANDLERS = {"order_status": order_status, "return_check": return_check, "open_ticket": open_ticket, "answer": answer}


def run_step(step: dict, request: str) -> dict:
    """Run one plan step and return its evidence."""
    return {"action": step["action"], **HANDLERS[step["action"]](step.get("order_id", ""), request)}


def sentence(ev: dict) -> str:
    """One plain sentence per piece of evidence (used by the template reply)."""
    oid = ev.get("order_id")
    if ev.get("error"):
        return f"I couldn't find an order with ID {oid}."
    if ev["action"] == "order_status":
        if ev["status"] == "shipped":
            return f"Order {oid} ({ev['item']}) has shipped with {ev['carrier']} and should arrive in {ev['eta']}."
        if ev["status"] == "delivered":
            return f"Order {oid} ({ev['item']}) was delivered {ev['delivered']}."
        return f"Order {oid} ({ev['item']}) is still {ev['status']} ({ev.get('note', 'no date yet')})."
    if ev["action"] == "return_check":
        if ev["returnable"]:
            return (f"Order {oid} ({ev['item']}) can be returned: you have {ev['days_left']} days left, "
                    f"and the refund arrives {ev['refund']}.")
        return f"Order {oid} ({ev['item']}) can't be returned: {ev.get('reason', 'the 30-day window has passed')}."
    if ev["action"] == "open_ticket":
        return f"I opened ticket {ev['ticket_id']} for order {oid}; our team will follow up."
    if ev.get("need"):
        return "Which order do you mean? Please send me the order ID (the letter A and four digits)."
    return f"Our returns policy: {ev['policy']}"


def short(ev: dict) -> str:
    """A few words per piece of evidence, for the execute log."""
    if ev.get("error"):
        return ev["error"]
    if ev["action"] == "order_status":
        detail = f"ETA {ev['eta']}" if "eta" in ev else ev.get("delivered", ev.get("note", ""))
        return f"{ev['status']}, {detail}"
    if ev["action"] == "return_check":
        return f"returnable, {ev['days_left']} days left" if ev["returnable"] else "not returnable"
    if ev["action"] == "open_ticket":
        return f"ticket {ev['ticket_id']} opened"
    return "needs an order ID" if ev.get("need") else "returns policy"


def template_reply(evidence: list[dict], profile: dict) -> str:
    """The deterministic reply: facts only. Used when the model can't produce a good draft."""
    text = " ".join(sentence(ev) for ev in evidence)
    if not text:
        text = "Could you tell me your order ID (the letter A and four digits)?"
    if profile.get("contact") == "email":
        text += " We'll follow up by email."
    return text
