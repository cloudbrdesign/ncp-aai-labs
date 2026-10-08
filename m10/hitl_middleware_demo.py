"""Side scene for lesson 10.4: the same approval with LangChain's built-in HumanInTheLoopMiddleware.

    python m10/hitl_middleware_demo.py

An agent from langchain.agents.create_agent with the desk's model (Ollama llama3.2:3b, temperature 0) and two
tools. The middleware interrupts before issue_refund runs and lets order_status run freely:

    HumanInTheLoopMiddleware(interrupt_on={
        "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"], "when": ...},
        "order_status": False})

It needs a checkpointer (InMemorySaver here) and a thread ID to resume. Five runs, each on its own thread:
  approve        the refund runs as the model proposed it
  edit           the reviewer halves the amount; the tool runs with the edited arguments and the model is
                 told (edit_notice) that a person replaced its call
  reject         the tool does not run; the message goes back to the model as the reason
  status         order_status needs no approval: no interrupt
  when           the same refund with a `when` predicate that interrupts only above EUR 200 (langchain >= 1.3.3):
                 the EUR 189 refund runs with no person. A conditional interrupt is a policy decision.
Whether the 3B calls the tool at all is recorded, not assumed (the video shows what happened).
Each run has its own empty ledger, state/demo/<run>.refunds.db, so the runs don't affect each other.

Checked in the installed langchain 1.4.2: HumanInTheLoopMiddleware(interrupt_on, *, description_prefix,
edit_notice); InterruptOnConfig keys allowed_decisions, description, args_schema, when (a predicate on
ToolCallRequest, whose .tool_call has the call's args); the interrupt value is a HITLRequest
{action_requests, review_configs}; the resume value is {"decisions": [...]}, one per action request.
"""
import json
import uuid

import bootstrap  # noqa: F401  (first: state folder, fake server, the v9 desk)
from bootstrap import STATE, gd

import refunds
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

CUSTOMER = "Tom B."
_thread = {"id": ""}


@tool
def order_status(order_id: str) -> str:
    """Look up the status of one of the customer's orders, like A1002."""
    e = refunds.eligibility(order_id, CUSTOMER)
    return json.dumps({k: e.get(k) for k in ("order_id", "item", "status", "delivered", "reason")})


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund an amount in EUR for one of the customer's orders. Needs a person's approval."""
    r = refunds.issue_refund(_thread["id"], order_id.upper(), CUSTOMER, float(amount), reason, "middleware demo",
                             "middleware")
    return f"{r['status']}: " + (f"refund {r['refund_id']} of EUR {r['amount']:.2f}" if r.get("refund_id") else r.get("why", ""))


def agent(when_above: float | None = None):
    cfg = {"allowed_decisions": ["approve", "edit", "reject"]}
    if when_above is not None:
        cfg["when"] = lambda req: float(req.tool_call["args"].get("amount", 0)) > when_above
    mw = HumanInTheLoopMiddleware(interrupt_on={"issue_refund": cfg, "order_status": False})
    return create_agent(gd.llm_calls.chat_model(), tools=[order_status, issue_refund], middleware=[mw],
                        checkpointer=InMemorySaver(),
                        system_prompt=f"You help {CUSTOMER} at an electronics shop. Use the tools; never invent a result.")


def run(name: str, text: str, decide=None, when_above: float | None = None) -> dict:
    _thread["id"] = f"demo-{name}-{uuid.uuid4().hex[:6]}"
    refunds.LEDGER = STATE / "demo" / f"{name}.refunds.db"
    refunds.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    refunds.LEDGER.unlink(missing_ok=True)
    app, cfg = agent(when_above), {"configurable": {"thread_id": _thread["id"]}}
    print(f"\n== {name}: {text}")
    out = app.invoke({"messages": [{"role": "user", "content": text}]}, cfg, version="v2")
    asked = bool(out.interrupts)
    if asked:
        req = out.interrupts[0].value
        print("interrupt payload:", json.dumps(req, indent=1, default=str))
        if decide:
            d = decide(req["action_requests"][0])
            print("decision:", json.dumps(d))
            out = app.invoke(Command(resume={"decisions": [d]}), cfg, version="v2")
    msgs = out.value["messages"]
    calls = [c["name"] for m in msgs for c in (getattr(m, "tool_calls", None) or [])]
    results = [m.content for m in msgs if m.type == "tool"]
    print(f"tool calls by the model: {calls or 'none'}; interrupted: {asked}")
    for r in results:
        print(f"tool result: {r[:200]}")
    print(f"final reply: {msgs[-1].content[:300]}")
    rows = refunds.rows()
    print(f"ledger: {len(rows)} refund{'s' if len(rows) != 1 else ''}" + "".join(f", EUR {r['amount']:.2f}" for r in rows))
    return {"name": name, "tool_calls": calls, "interrupted": asked, "tool_results": results, "reply": msgs[-1].content,
            "refunds": [r["amount"] for r in rows]}


def main():
    print(f"[INFO] model: {gd.llm_calls.describe()}")
    ask = "I want my money back for order A1002, please refund it."
    runs = [
        run("approve", ask, lambda a: {"type": "approve"}),
        run("edit", ask, lambda a: {"type": "edit", "edited_action": {
            "name": a["name"], "args": {**a["args"], "amount": round(float(a["args"].get("amount", 0)) / 2, 2)}}}),
        run("reject", ask, lambda a: {"type": "reject", "message": "Refunds need the dock back first. Explain how to return it."}),
        run("status", "Where is my order A1005?"),
        run("when", ask, lambda a: {"type": "approve"}, when_above=200.0),
    ]
    print("\n[ledger] refunds per run: " + ", ".join(f"{r['name']} {r['refunds'] or 'none'}" for r in runs))
    (STATE / "demo" / "demo.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in runs))
    print(f"[demo] {sum(r['interrupted'] for r in runs)} of {len(runs)} runs interrupted; the model called a tool in "
          f"{sum(bool(r['tool_calls']) for r in runs)} of {len(runs)}; runs in state/demo/demo.jsonl")
    gd.close()


if __name__ == "__main__":
    main()
