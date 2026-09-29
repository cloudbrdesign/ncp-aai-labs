"""Planning: turn a multi-part request into at most 4 steps from a fixed menu.

This is decomposition-first planning (plan-and-execute): the whole plan is made before
any step runs. The model only fills in a small schema (structured output, temperature 0);
the code then checks the plan against simple rules. If the model's output is invalid
(bad JSON, unknown step, an order ID it forgot or made up), a regex splitter builds
the plan instead, and the log says so.
"""
import pathlib
import re
from typing import Literal

from pydantic import BaseModel, Field

from llm_calls import chat

STEP_TYPES = ("order_status", "return_check", "open_ticket", "answer")
MAX_STEPS = 4
PROMPT = (pathlib.Path(__file__).resolve().parent / "prompts" / "plan.txt").read_text()

ORDER_ID = re.compile(r"\b[Aa]\d{4}\b")
DAMAGE = re.compile(r"broken|cracked|damaged|missing|wrong item|faulty|doesn'?t work", re.I)
RETURN = re.compile(r"\breturn|refund|send (it )?back", re.I)


class Step(BaseModel):
    action: Literal["order_status", "return_check", "open_ticket", "answer"]
    order_id: str = Field(default="", description="the order ID this step is about, like A1001; empty for answer")


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)


def order_ids(text: str) -> list[str]:
    """Order IDs in the order they appear, without repeats."""
    return list(dict.fromkeys(m.upper() for m in ORDER_ID.findall(text)))


def problems(plan: Plan, request: str, carried: list[str]) -> list[str]:
    """Rule checks on the model's plan. An empty list means the plan is accepted."""
    wanted = order_ids(request) or carried[-1:]
    planned = {s.order_id.upper() for s in plan.steps if s.order_id}
    found = [f"step {s.action} has no order ID" for s in plan.steps if s.action != "answer" and not s.order_id]
    found += [f"missing order {o}" for o in wanted if o not in planned]
    found += [f"order {o} is not in the request" for o in planned if o not in wanted]
    actions = {s.action for s in plan.steps}
    if DAMAGE.search(request) and "open_ticket" not in actions:
        found.append("damaged item but no open_ticket step")
    if RETURN.search(request) and "return_check" not in actions:
        found.append("return question but no return_check step")
    return found


def regex_plan(request: str, carried: list[str]) -> Plan:
    """The fallback planner: split on sentences, commas and 'and', then pick a step type by keywords."""
    steps = []
    current = carried[-1] if carried else ""
    for clause in re.split(r"(?<=[.?!])\s+|,|\band\b|\balso\b", request, flags=re.I):
        ids = order_ids(clause)
        current = ids[0] if ids else current
        if DAMAGE.search(clause):
            action = "open_ticket"
        elif RETURN.search(clause):
            action = "return_check"
        elif ids or (not order_ids(request) and re.search(r"where|when|arriv|status|shipped", clause, re.I)):
            action = "order_status"
        else:
            continue
        for oid in ids or [current]:
            if oid and {"action": action, "order_id": oid} not in steps:
                steps.append({"action": action, "order_id": oid})
    return Plan(steps=steps[:MAX_STEPS] or [{"action": "answer", "order_id": ""}])


def make_plan(request: str, history: str, carried: list[str], log=print) -> tuple[Plan, str]:
    """Ask the model for a plan; fall back to regex_plan() if it is invalid. Returns (plan, source)."""
    user = (f"Conversation so far:\n{history}\n\n" if history else "") + f"Request: {request}"
    try:
        plan = chat([("system", PROMPT), ("user", user)], schema=Plan)
        plan.steps = [Step(action=s.action, order_id=s.order_id.strip().upper()) for s in plan.steps]
        issues = problems(plan, request, carried)
    except Exception as e:  # invalid JSON, schema mismatch, model not reachable
        issues = [f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"]
    if not issues:
        return plan, "model"
    log(f"[plan] model plan rejected ({'; '.join(issues)}); using the regex fallback")
    return regex_plan(request, carried), "fallback"
