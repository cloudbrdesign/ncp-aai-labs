"""Routing: which source answers each part of a request, the order database or the manuals?

The M3 planner's menu gains one step, manual_search. Each step type goes to one source:

    order_status, return_check   SQL (orders_sql.py, named read-only queries)
    open_ticket                  the M2 ticket API
    manual_search                RAG over the manuals (hybrid search, filtered by product)
    answer                       the M1 returns policy, or a question back

A request can need both: "The dock from order A1002 won't drive my second screen" is
a manual question, but only the order says which dock. The plan is then one
manual_search step with order_id A1002 and no product; the execute step asks SQL for
the order's product (D300) and filters the manual search to it. A structured fact
becomes a retrieval filter.

As in M3, the model only fills a small schema (temperature 0), the code checks the plan
with rules, and a keyword router takes over when the plan is invalid; the log says so.
"""
import pathlib
import re
from typing import Literal

from pydantic import BaseModel, Field

from llm_calls import chat

PROMPT = (pathlib.Path(__file__).resolve().parent / "prompts" / "plan.txt").read_text()
MAX_STEPS = 4
SOURCE = {"order_status": "sql", "return_check": "sql", "open_ticket": "ticket API",
          "manual_search": "rag", "answer": "policy"}

ORDER_ID = re.compile(r"\b[Aa]\d{4}\b")
CODE = re.compile(r"\b(H200|D300|M270)\b", re.I)
RETURN = re.compile(r"\breturn|refund|send (it |something |them )?back|money back", re.I)
PRODUCT_WORDS = [("FAQ", re.compile(r"warranty|guarantee|" + RETURN.pattern, re.I)),  # first: policy beats product
                 ("H200", re.compile(r"head ?set|head ?phones?|earphones?", re.I)),
                 ("D300", re.compile(r"\bdock\b|docking|\bhub\b", re.I)),
                 ("M270", re.compile(r"\bmonitor\b", re.I))]
DAMAGE = re.compile(r"broken|cracked|damaged|smashed|missing|wrong item", re.I)
MANUAL = re.compile(r"how (do|can|to)|error|\bE\d\d\b|won'?t|doesn'?t|not working|pair|firmware|reset|set ?up|"
                    r"blink|flicker|charg|drive|mirror|mount|noise|warranty|guarantee|dead pixel|LED|manual|"
                    r"what does .* mean", re.I)
STATUS = re.compile(r"where|when|arriv|status|shipped|track", re.I)


class Step(BaseModel):
    action: Literal["order_status", "return_check", "open_ticket", "manual_search", "answer"]
    order_id: str = Field(default="", description="the order ID this step is about, like A1001, or empty")
    product: Literal["", "H200", "D300", "M270", "FAQ"] = Field(
        default="", description="manual_search only: the product the customer names, or empty")


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)


def order_ids(text: str) -> list[str]:
    return list(dict.fromkeys(m.upper() for m in ORDER_ID.findall(text)))


def named_product(text: str) -> str:
    """The product the text names: a product code first, then product words ("dock")."""
    code = CODE.search(text)
    if code:
        return code.group(1).upper()
    return next((p for p, words in PRODUCT_WORDS if words.search(text)), "")


def problems(plan: Plan, request: str, carried: list[str]) -> list[str]:
    """Rule checks on the model's plan. An empty list means the plan is accepted."""
    wanted = order_ids(request) or carried[-1:]
    planned = {s.order_id for s in plan.steps if s.order_id}
    actions = {s.action for s in plan.steps}
    found = [f"step {s.action} has no order ID" for s in plan.steps
             if s.action in ("order_status", "return_check", "open_ticket") and not s.order_id]
    found += [f"missing order {o}" for o in wanted if o not in planned]
    found += [f"order {o} is not in the request" for o in planned if o not in wanted]
    if MANUAL.search(request) and not DAMAGE.search(request) and "manual_search" not in actions:
        found.append("manual question but no manual_search step")
    if "manual_search" in actions and not MANUAL.search(request) and not named_product(request):
        found.append("manual_search, but the request is not about using a product")
    for s in plan.steps:
        if s.action != "manual_search" or s.order_id:   # with an order ID, SQL decides the product
            continue
        if s.product and s.product != named_product(request):
            found.append(f"product {s.product} is not the one in the request")
        elif not s.product and named_product(request):
            found.append(f"manual_search has no product, but the request names {named_product(request)}")
    if "open_ticket" in actions and not DAMAGE.search(request):
        found.append("open_ticket, but nothing is reported damaged, missing or wrong")
    if DAMAGE.search(request) and "open_ticket" not in actions:
        found.append("damaged item but no open_ticket step")
    if RETURN.search(request) and wanted and "return_check" not in actions:
        found.append("return question about an order but no return_check step")
    return found


def keyword_plan(request: str, carried: list[str]) -> Plan:
    """The fallback router: split into clauses, then pick a step type by keywords."""
    steps = []
    current = carried[-1] if carried else ""
    for clause in re.split(r"(?<=[.?!])\s+|,|\band\b|\balso\b", request, flags=re.I):
        ids = order_ids(clause)
        current = ids[0] if ids else current
        product = named_product(clause) or named_product(request)
        if DAMAGE.search(clause) and current:
            step = {"action": "open_ticket", "order_id": current}
        elif RETURN.search(clause) and current:
            step = {"action": "return_check", "order_id": current}
        elif MANUAL.search(clause) or (RETURN.search(clause) and not current):
            # an order in the request says which product (SQL resolves it in execute),
            # unless this clause names a product code itself
            oid = current if order_ids(request) and not CODE.search(clause) else ""
            step = {"action": "manual_search", "order_id": oid, "product": "" if oid else product}
        elif ids or (not order_ids(request) and current and STATUS.search(clause)):
            step = {"action": "order_status", "order_id": current}
        else:
            continue
        if step not in steps:
            steps.append(step)
    return Plan(steps=steps[:MAX_STEPS] or [{"action": "answer"}])


def make_plan(request: str, history: str, carried: list[str], log=print) -> tuple[Plan, str]:
    """Ask the model for a plan; fall back to keyword_plan() if it is invalid. Returns (plan, source)."""
    user = (f"Conversation so far:\n{history}\n\n" if history else "") + f"Request: {request}"
    try:
        plan = chat([("system", PROMPT), ("user", user)], schema=Plan)
        for s in plan.steps:
            s.order_id = s.order_id.strip().upper()
        if len(plan.steps) > 1:   # a general "answer" step next to real steps adds nothing
            plan.steps = [s for s in plan.steps if s.action != "answer"] or plan.steps
        issues = problems(plan, request, carried)
    except Exception as e:   # invalid JSON, schema mismatch, model not reachable
        issues = [f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"]
    if not issues:
        return plan, "model"
    log(f"[plan] model plan rejected ({'; '.join(issues)}); using the keyword router")
    return keyword_plan(request, carried), "fallback"


def describe_step(s: dict) -> str:
    what = " ".join(x for x in (s["action"], s.get("order_id", ""), s.get("product", "")) if x)
    return f"{what} ({SOURCE[s['action']]})"
