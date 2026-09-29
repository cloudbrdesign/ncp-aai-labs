"""The M4 desk's step handlers: order facts from SQL, how-to answers from the manuals.

    order_status    orders_sql.order_status (was m01/data/orders.json in M3)
    return_check    the same 30-day rule as M3, on the SQL row
    open_ticket     M3's handler (the M2 ticket API)
    manual_search   hybrid search over the manuals, filtered by product; the product comes
                    from the request, or from SQL when the step names an order instead
    answer          M3's handler (the M1 returns policy, or "which order?")

Each handler returns a small dict of evidence. The draft may only use this evidence,
and the critique checks the reply against it, including every cited chunk ID.
"""
import re

import handlers          # m03/handlers.py
import orders_sql
import retrieve
import vector_store

RETURN_DAYS = 30
TOP_K = 3
_milvus = {}


def milvus():
    """One Milvus Lite client for the whole run (opening the file takes a moment)."""
    if "client" not in _milvus:
        client = vector_store.connect()
        vector_store.load(client)
        _milvus["client"] = client
    return _milvus["client"]


def order_status(step: dict, request: str) -> dict:
    row = orders_sql.order_status(step["order_id"])
    return {"order_id": step["order_id"], **row} if row else {"order_id": step["order_id"], "error": "no such order"}


def return_check(step: dict, request: str) -> dict:
    oid = step["order_id"]
    row = orders_sql.order_status(oid)
    if not row:
        return {"order_id": oid, "error": "no such order"}
    if row["status"] != "delivered":
        return {"order_id": oid, "item": row["item"], "returnable": False,
                "reason": f"not delivered yet ({row['status']}); it can be cancelled instead"}
    days = int(re.match(r"(\d+)", row["delivered"]).group(1))
    if days > RETURN_DAYS:
        return {"order_id": oid, "item": row["item"], "returnable": False, "delivered_days_ago": days,
                "reason": f"it was delivered {days} days ago, outside the {RETURN_DAYS}-day window"}
    return {"order_id": oid, "item": row["item"], "returnable": True, "delivered_days_ago": days,
            "days_left": RETURN_DAYS - days,
            "refund": "within 5 business days of the return arriving, to the original payment method"}


def manual_search(step: dict, request: str, log=print) -> dict:
    """RAG: resolve the product (SQL if needed), then hybrid search filtered to it."""
    product, oid = step.get("product", ""), step.get("order_id", "")
    if oid:
        from_sql = orders_sql.order_product(oid)
        log(f"[sql]   order_product({oid}) -> {from_sql}")
        product = from_sql or product
    hits = retrieve.search(milvus(), request, "hybrid", TOP_K, product=product or None)
    log(f"[rag]   hybrid search, filter {retrieve.product_filter(product) or 'none'}")
    log(f"[rag]   -> {', '.join(h['id'] for h in hits) or 'nothing found'}")
    passages = [{"id": h["id"], "section": h["section"], "text": " ".join(h["text"].split("\n", 1)[-1].split())}
                for h in hits]
    ev = {"order_id": oid, "product": product, "passages": passages}
    if oid:
        row = orders_sql.order_status(oid)
        ev["item"] = row["item"] if row else ""
    return ev


def run_step(step: dict, request: str, log=print) -> dict:
    action = step["action"]
    if action == "order_status":
        ev = order_status(step, request)
    elif action == "return_check":
        ev = return_check(step, request)
    elif action == "manual_search":
        ev = manual_search(step, request, log)
    elif action == "open_ticket":
        ev = handlers.open_ticket(step["order_id"], request)
    else:
        ev = handlers.answer(step.get("order_id", ""), request)
    return {"action": action, **ev}


def sentence(ev: dict) -> str:
    """One plain sentence per fact (manual passages are listed separately)."""
    if ev["action"] == "manual_search":
        if ev.get("order_id") and ev.get("item"):
            return f"Order {ev['order_id']} is a {ev['item']} ({ev['product']})."
        return ""
    return handlers.sentence(ev)


def short(ev: dict) -> str:
    if ev["action"] == "manual_search":
        return f"{len(ev['passages'])} passages"
    return handlers.short(ev)


def passages(evidence: list[dict]) -> list[dict]:
    return [p for ev in evidence if ev["action"] == "manual_search" for p in ev["passages"]]


def template_reply(evidence: list[dict], profile: dict) -> str:
    """The deterministic reply: order facts, plus the top passage of each manual search, cited."""
    parts = [s for s in (sentence(ev) for ev in evidence) if s]
    for ev in evidence:
        if ev["action"] == "manual_search":
            if ev["passages"]:
                p = ev["passages"][0]
                first = " ".join(re.split(r"(?<=[.!?])\s+", p["text"])[:2])
                parts.append(f"From the manual ({p['section']}): {first} [{p['id']}]")
            else:
                parts.append("I couldn't find this in our manuals; a colleague will get back to you.")
    text = " ".join(parts) or "Could you tell me your order ID (the letter A and four digits)?"
    if profile.get("contact") == "email":
        text += " We'll follow up by email."
    return text
