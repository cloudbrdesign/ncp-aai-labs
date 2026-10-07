"""Version 10 of the support desk: the v9 desk plus a refund tool that never runs without a person.

    python m10/oversight_desk.py --customer "Tom B." --input "I want my money back for A1002"
    python m10/oversight_desk.py --serve --port 8110          # the web page on http://127.0.0.1:8110/
    python m10/approvals.py pending                            # the reviewer queue from the terminal

The outer graph (LangGraph StateGraph, checkpointed in state/threads.db with SqliteSaver):

    desk_turn -> propose_refund -> review -> execute_refund -> reply
        |              |              |  (reject)               ^
        +--------------+--------------+-------------------------+   (no refund asked, refused by rule, stopped)

  desk_turn       Module 9's Desk.respond(): rails, identity scope, the audit record, escalation, disclosure.
                  Inside it runs the M4 graph (plan, execute, draft, critique) unchanged.
  propose_refund  only when the message matches M4's RETURN pattern and names an order. Eligibility is a
                  rule (refunds.py): ineligible requests end here, no person is asked. Otherwise the 3B fills
                  a small schema (order, amount, reason) and the node builds the approval card.
  review          ONE interrupt() and nothing else. The payload has the shape of LangChain's
                  HumanInTheLoopMiddleware request: action_requests [name, args, description] and
                  review_configs [action_name, allowed_decisions, args_schema], plus the card's facts for the
                  page. On resume LangGraph re-runs this node from the top and interrupt() returns the
                  decision; nothing else is in the node, so nothing runs twice.
  execute_refund  the side effect, AFTER the interrupt, in its own node: refunds.issue_refund() checks the stop
                  switch and is idempotent (thread + order is a UNIQUE key), so a double resume pays once.
                  The reviewer's edited arguments go straight to the tool: the model is not asked again.
  reply           what the customer reads, from the outcome.

Why an outer graph: the M4 desk graph runs inside desk_turn. Had the interrupt been inside the desk (a
subgraph called as a function), resuming would re-run the parent node, and with it the whole desk turn.
Here the desk turn is finished and checkpointed before the card exists.

The thread ID is the conversation (M9's session ID). One pending card per thread: while it waits, a new
message in that thread gets a holding reply. The pending card lives in the checkpoint, so it survives a
restart of this process (drill.py --drill restart). state/approvals.db is an index of the cards
(for the queue and wait times); the card itself is read from get_state(...).tasks[*].interrupts.
"""
import argparse
import asyncio
import contextlib
import contextvars
import sqlite3
import sys
import threading
import time
import uuid
from typing import TypedDict

import bootstrap
from bootstrap import STATE, VERSION, escalate, gd

import refunds

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import router  # noqa: E402  (m04: RETURN pattern, order_ids)

THREADS_DB = STATE / "threads.db"
CARDS_DB = STATE / "approvals.db"
WEB = bootstrap.HERE / "web" / "index.html"
ALLOWED = ["approve", "edit", "reject"]
PROPOSAL_PROMPT = (
    "You prepare refund proposals for a person who approves them, at an electronics shop. Use only the facts "
    "given. amount: the refund in EUR. Propose the most that can be refunded, unless the customer asks for "
    "only part of the order; then propose the unit price times the number of items they name. Never propose "
    "more than the most that can be refunded. reason: one short sentence a reviewer can check.")
gd.desk_obs.WORKLOADS["fill_proposal"] = "refund"     # M8's request log labels this model call "refund"


class RefundProposal(BaseModel):
    """What the model fills for the card (the reviewer sees it and can edit the amount)."""
    order_id: str = Field(description="the order ID, like A1002")
    amount: float = Field(description="the refund amount in EUR")
    reason: str = Field(description="one short sentence: why this refund")


class Turn(TypedDict, total=False):
    thread_id: str
    customer: str
    text: str
    request_id: str | None
    trace_id: str
    desk_reply: str
    desk_action: str
    eligibility: dict | None
    payload: dict | None          # the interrupt payload (the card)
    decision: dict | None         # what review's interrupt() returned
    refund: dict | None           # what issue_refund() returned
    ticket_id: str | None
    status: str                   # answered | refused_by_rule | pending | refunded | rejected | stopped | not_executed
    reply: str


# ---- async: one event loop for the desk, whatever thread the graph runs in --------------------------

_loop: dict = {}
PASSAGES: dict[str, list[str]] = {}      # request_id -> retrieved chunk IDs (feedback records need them)
_m8_turn = gd.desk_obs.turn


async def _turn_with_passages(question: str) -> dict:
    """M8's desk turn, keeping the retrieved chunk IDs for this request (Desk.respond drops them)."""
    rec = await _m8_turn(question)
    PASSAGES[rec.get("request_id", "")] = [p["id"] for p in rec.get("passages", [])]
    return rec


gd.desk_obs.turn = _turn_with_passages


def run_async(coro):
    """Run a coroutine on the desk's own event loop (started once, in a daemon thread).

    It is submitted from an empty contextvars.Context on purpose. LangGraph keeps the running graph's
    config in a context variable, and asyncio copies the caller's context into the new task. Submitted
    from inside desk_turn, the M4/M5 desk graph would see the outer graph's config and run as its
    subgraph, with the outer SqliteSaver (which has no async methods) and the outer thread's checkpoints.
    From an empty context it runs as its own graph, exactly as in Module 9."""
    if "loop" not in _loop:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True, name="desk-loop").start()
        _loop["loop"] = loop
    return contextvars.Context().run(asyncio.run_coroutine_threadsafe, coro, _loop["loop"]).result()


# ---- the card index (approvals.db) ----------------------------------------------------------------

@contextlib.contextmanager
def cards_db():
    CARDS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CARDS_DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS cards (
        card_id TEXT PRIMARY KEY, thread_id TEXT, request_id TEXT, customer TEXT, order_id TEXT, amount REAL,
        required INTEGER, created_ts REAL, status TEXT, decided_ts REAL, decision TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS partials (
        card_id TEXT, reviewer TEXT, decision TEXT, amount REAL, ts REAL, PRIMARY KEY (card_id, reviewer))""")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def register_card(card: dict) -> None:
    with cards_db() as con:
        con.execute("INSERT OR IGNORE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (card["card_id"], card["thread_id"], card["request_id"], card["customer"],
                     card["order"]["order_id"], card["proposal"]["amount"], card["required_approvals"],
                     card["created_ts"], "pending", None, None))


# ---- the nodes ----------------------------------------------------------------------------------

def fill_proposal(text: str, elig: dict) -> tuple[dict, str]:
    """The 3B fills RefundProposal from the facts. Returns (proposal, source). A rule fills it if the
    model's output can't be used, and the card says so."""
    facts = (f"Order: {elig['order_id']}, {elig['qty']} x {elig['item']} ({elig['product']}), unit price: "
             f"{refunds.money(elig['unit_price'])}\nPolicy check: {elig['reason']}\n"
             f"Already refunded: {refunds.money(elig['already_refunded'])}\n"
             f"Most that can be refunded: {refunds.money(elig['max_amount'])}\nCustomer's message: {text}")
    try:
        p = gd.llm_calls.chat([("system", PROPOSAL_PROMPT), ("user", facts)], RefundProposal)
        return {"order_id": p.order_id.upper(), "amount": round(float(p.amount), 2), "reason": p.reason.strip()}, "model"
    except Exception as e:
        return {"order_id": elig["order_id"], "amount": elig["max_amount"],
                "reason": f"Refund requested by the customer (model output unusable: {type(e).__name__})"}, "rule"


def build_card(state: Turn, elig: dict) -> dict:
    proposal, source = fill_proposal(state["text"], elig)
    checks = []
    if proposal["order_id"] != elig["order_id"]:
        checks.append(f"the model named order {proposal['order_id']}; the card uses {elig['order_id']}")
        proposal["order_id"] = elig["order_id"]
    why = refunds.validate_amount(proposal["amount"], elig["max_amount"])
    if why:
        checks.append(f"proposed amount not payable: {why}. Edit or reject.")
    amount = proposal["amount"]
    required = 2 if refunds.needs_two(amount) else 1
    now = time.time()
    args = {"order_id": elig["order_id"], "amount": amount, "reason": proposal["reason"], "customer": state["customer"]}
    card = {"card_id": "c-" + uuid.uuid4().hex[:8], "thread_id": state["thread_id"], "request_id": state["request_id"],
            "trace_id": state["trace_id"], "customer": state["customer"], "customer_message": state["text"],
            "created_ts": round(now, 3),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "order": {k: elig[k] for k in ("order_id", "item", "product", "qty", "unit_price", "order_value",
                                           "already_refunded", "max_amount", "status", "delivered")},
            "policy": elig["reason"], "proposal": {**proposal, "source": source}, "checks": checks,
            "required_approvals": required, "dual_threshold": refunds.DUAL_APPROVAL_ABOVE}
    return {
        "action_requests": [{
            "name": "issue_refund", "args": args,
            "description": (f"Refund {refunds.money(amount)} to {state['customer']} for order {elig['order_id']} "
                            f"({elig['qty']} x {elig['item']}). Policy: {elig['reason']}. "
                            + ("Needs two approvers. " if required == 2 else "") + f"Proposed by the {source}.")}],
        "review_configs": [{
            "action_name": "issue_refund", "allowed_decisions": ALLOWED,
            "args_schema": {"type": "object", "required": ["amount"],
                            "properties": {"amount": {"type": "number", "exclusiveMinimum": 0,
                                                      "maximum": elig["max_amount"]},
                                           "reason": {"type": "string"}}}}],
        "card": card}


def build_graph(od: "OversightDesk") -> StateGraph:

    def desk_turn(state: Turn) -> dict:
        out = run_async(od.desk.respond(state["text"], customer=state["customer"], session=state["thread_id"],
                                        request_id=state.get("request_id")))
        return {"desk_reply": out["reply"], "desk_action": out["action"], "request_id": out["request_id"],
                "trace_id": out["trace_id"], "eligibility": None, "payload": None, "decision": None,
                "refund": None, "ticket_id": None, "status": "answered", "reply": out["reply"]}

    def after_desk(state: Turn) -> str:
        asks = router.RETURN.search(state["text"]) and router.order_ids(state["text"])
        return "propose_refund" if asks and state["desk_action"] in ("answered", "masked") else "reply"

    def propose_refund(state: Turn) -> dict:
        gd.desk_obs._turn.set({"request_id": state["request_id"], "trace_id": state["trace_id"]})
        oid = router.order_ids(state["text"])[0]
        if refunds.stopped():                       # no new cards while refunds are stopped
            tid, _ = escalate.open_ticket(f"Refund request while refunds are stopped | order {oid} | "
                                          f"thread {state['thread_id']} | request {state['request_id']}", oid)
            return {"status": "stopped", "ticket_id": tid, "eligibility": {"order_id": oid}}
        elig = refunds.eligibility(oid, state["customer"])
        if not elig["eligible"]:
            return {"status": "refused_by_rule", "eligibility": elig}
        payload = build_card(state, elig)
        register_card(payload["card"])
        return {"status": "pending", "eligibility": elig, "payload": payload}

    def after_propose(state: Turn) -> str:
        return "review" if state["status"] == "pending" else "reply"

    def review(state: Turn) -> dict:
        # The whole node. On resume it runs again from here and interrupt() returns the decision.
        return {"decision": interrupt(state["payload"])}

    def after_review(state: Turn) -> str:
        return "execute_refund" if state["decision"]["decisions"][0]["type"] in ("approve", "edit") else "reply"

    def execute_refund(state: Turn) -> dict:
        d = state["decision"]
        dec = d["decisions"][0]
        args = dict(state["payload"]["action_requests"][0]["args"])
        if dec["type"] == "edit":
            args.update(dec["edited_action"]["args"])
        res = refunds.issue_refund(state["thread_id"], args["order_id"], state["customer"], float(args["amount"]),
                                   args.get("reason", ""), approved_by=" + ".join(d.get("reviewers", [])),
                                   decision=dec["type"], request_id=state["request_id"])
        status = {"issued": "refunded", "duplicate": "refunded", "refused": "not_executed", "stopped": "stopped"}[res["status"]]
        tid = None
        if res["status"] in ("stopped", "refused"):
            tid, _ = escalate.open_ticket(f"Approved refund not paid ({res['status']}) | order {args['order_id']} | "
                                          f"{refunds.money(float(args['amount']))} | thread {state['thread_id']} | "
                                          f"request {state['request_id']}", args["order_id"])
        return {"refund": {**res, "args": args}, "status": status, "ticket_id": tid}

    def reply(state: Turn) -> dict:
        out = {"reply": compose_reply(state)}
        if state.get("decision") and state["decision"]["decisions"][0]["type"] == "reject":
            out["status"] = "rejected"
        return out

    g = StateGraph(Turn)
    for name, fn in (("desk_turn", desk_turn), ("propose_refund", propose_refund), ("review", review),
                     ("execute_refund", execute_refund), ("reply", reply)):
        g.add_node(name, fn)
    g.add_edge(START, "desk_turn")
    g.add_conditional_edges("desk_turn", after_desk, ["propose_refund", "reply"])
    g.add_conditional_edges("propose_refund", after_propose, ["review", "reply"])
    g.add_conditional_edges("review", after_review, ["execute_refund", "reply"])
    g.add_edge("execute_refund", "reply")
    g.add_edge("reply", END)
    return g


def compose_reply(state: Turn) -> str:
    """The customer's reply for each outcome. Facts only from the eligibility rule and the ledger."""
    base, st = state["desk_reply"], state["status"]
    elig = state.get("eligibility") or {}
    oid = elig.get("order_id", "")
    ticket = f" (ticket {state['ticket_id']})" if state.get("ticket_id") else ""
    if st == "refused_by_rule":
        return f"{base}\n\nI can't start a refund for order {oid}: {elig['reason']}."
    if st == "stopped" and not state.get("decision"):
        return (f"{base}\n\nRefunds are paused at the moment, so I can't start one now. "
                f"A colleague will follow up with you{ticket}.")
    if not state.get("decision"):
        return base
    dec = state["decision"]["decisions"][0]
    proposed = state["payload"]["action_requests"][0]["args"]["amount"]
    if dec["type"] == "reject":
        note = dec.get("message") or ""
        return (f"A colleague reviewed your refund request for order {oid} and did not approve it."
                + (f" Their note: {note}" if note else " You can ask here why."))
    r = state["refund"]
    if st == "refunded":
        amount = r["amount"]
        part = "" if abs(amount - proposed) < 0.005 else f" (the proposal was {refunds.money(proposed)})"
        return (f"A colleague approved a refund of {refunds.money(amount)} for order {oid}{part}, refund "
                f"{r['refund_id']}. It goes back to your original payment method within 5 business days.")
    if st == "stopped":
        return (f"Your refund for order {oid} was approved, but refunds are paused right now, so nothing has "
                f"been paid yet. A colleague will follow up with you{ticket}.")
    return (f"Your refund for order {oid} was approved but could not be paid: {r.get('why')}. "
            f"A colleague will look at it{ticket}.")


# ---- the v10 desk ---------------------------------------------------------------------------------

HOLD = ("Your refund request for order {order} is with a colleague (waiting {wait}). "
        "You'll see their answer here.")


class OversightDesk:
    """The v10 desk. chat() is one customer turn; resume() hands a reviewer's decision to a pending card."""

    def __init__(self, layer: str = "L4", hosted: bool = False):
        self.layer, self.hosted = layer, hosted
        self._desk = None
        THREADS_DB.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(THREADS_DB, check_same_thread=False)
        self.app = build_graph(self).compile(checkpointer=SqliteSaver(self.conn))

    @property
    def desk(self):
        """Module 9's desk, built on first use (the reviewer CLI never needs its rails)."""
        if self._desk is None:
            self._desk = gd.Desk(layer=self.layer, hosted=self.hosted)
        return self._desk

    @staticmethod
    def config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def pending(self, thread_id: str) -> dict | None:
        """The pending interrupt payload of a thread, read from the checkpoint (None if nothing waits)."""
        snap = self.app.get_state(self.config(thread_id))
        for task in snap.tasks:
            for intr in task.interrupts:
                return intr.value
        return None

    def thread(self, thread_id: str) -> dict:
        snap = self.app.get_state(self.config(thread_id))
        v = dict(snap.values or {})
        p = self.pending(thread_id)
        return {"thread_id": thread_id, "status": v.get("status"), "reply": v.get("reply"),
                "request_id": v.get("request_id"), "pending": p["card"] if p else None,
                "refund": v.get("refund"), "next": list(snap.next)}

    def chat(self, text: str, customer: str = "Tom B.", thread_id: str | None = None,
             request_id: str | None = None) -> dict:
        thread_id = thread_id or "t-" + uuid.uuid4().hex[:8]
        waiting = self.pending(thread_id)
        if waiting:
            c = waiting["card"]
            wait = f"{(time.time() - c['created_ts']) / 60:.0f} min"
            return {"reply": HOLD.format(order=c["order"]["order_id"], wait=wait), "thread_id": thread_id,
                    "status": "pending", "pending": c["card_id"], "card": c, "request_id": c["request_id"],
                    "trace_id": c["trace_id"], "action": "holding"}
        out = self.app.invoke({"text": text, "customer": customer, "thread_id": thread_id, "request_id": request_id},
                              self.config(thread_id), version="v2")
        return self._result(thread_id, out)

    def resume(self, thread_id: str, value: dict) -> dict:
        """Command(resume=...) on the thread: review's interrupt() returns `value`, the graph goes on."""
        out = self.app.invoke(Command(resume=value), self.config(thread_id), version="v2")
        return self._result(thread_id, out)

    def _result(self, thread_id: str, out) -> dict:
        v = out.value
        res = {"thread_id": thread_id, "status": v.get("status"), "request_id": v.get("request_id"),
               "trace_id": v.get("trace_id"), "action": v.get("desk_action"), "pending": None, "card": None,
               "refund": v.get("refund"), "ticket_id": v.get("ticket_id"), "reply": v.get("reply")}
        if out.interrupts:                          # GraphOutput.interrupts (invoke version="v2")
            card = out.interrupts[0].value["card"]
            res.update(pending=card["card_id"], card=card, status="pending",
                       reply=(f"{v['desk_reply']}\n\nI've asked a colleague to approve a refund of "
                              f"{refunds.money(card['proposal']['amount'])} for order {card['order']['order_id']}. "
                              "You'll see their answer here."))
        return res

    def close(self) -> None:
        self.conn.close()
        gd.close()


# ---- HTTP -----------------------------------------------------------------------------------------

def make_app(od: OversightDesk):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    import approvals
    import decision_record
    import feedback_loop

    app = FastAPI(title=f"Support desk {VERSION}")

    @app.get("/")
    def page():
        return FileResponse(WEB)

    @app.get("/health")
    def health():
        return {"ok": True, "version": VERSION, "layer": od.layer, "stop": refunds.stopped()}

    @app.post("/v1/chat")
    def chat(body: dict, request: Request):
        msgs = body.get("messages") or []
        text = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        h = request.headers
        out = od.chat(text, customer=h.get("x-customer") or "Tom B.",
                      thread_id=h.get("x-thread-id") or h.get("x-session-id"), request_id=h.get("x-request-id"))
        payload = {"id": "chatcmpl-" + (out["request_id"] or ""), "object": "chat.completion", "model": f"desk-{VERSION}",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": out["reply"]},
                                "finish_reason": "stop"}]}
        headers = {"x-request-id": out["request_id"] or "", "x-trace-id": out["trace_id"] or "",
                   "x-thread-id": out["thread_id"], "x-pending": out["pending"] or "", "x-status": out["status"] or "",
                   "x-action": out["action"] or "", "x-version": VERSION}
        return JSONResponse(payload, headers=headers)

    @app.get("/v1/threads/{thread_id}")
    def thread(thread_id: str):
        return od.thread(thread_id)

    @app.get("/v1/approvals")
    def queue():
        return {"stop": refunds.stopped(), "cards": approvals.pending(od), "now": time.time()}

    @app.post("/v1/approvals/{thread_id}/decision")
    def decide(thread_id: str, body: dict):
        try:
            out = approvals.decide(od, thread_id, body.get("decision", ""), body.get("reviewer", ""),
                                   amount=body.get("amount"), message=body.get("message"))
        except approvals.NothingPending as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except approvals.InvalidDecision as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        return out

    @app.post("/v1/feedback")
    def feedback(body: dict):
        try:
            rec = feedback_loop.record_click(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        return JSONResponse(rec, status_code=201)

    @app.get("/v1/records/{request_id}")
    def record(request_id: str, view: str = "customer"):
        rec = decision_record.build(request_id)
        if not rec["audit"]:
            return JSONResponse({"error": f"no audit record for request {request_id}"}, status_code=404)
        return decision_record.view(rec, view)

    @app.get("/v1/stop")
    def stop_state():
        return {"stop": refunds.stopped()}

    @app.post("/v1/stop")
    def stop(body: dict):
        return {"stop": refunds.set_stop(bool(body.get("on")), body.get("reason", ""), body.get("by", ""))}

    return app


def show(out: dict) -> None:
    print(f"[turn]   thread {out['thread_id']}, request {out['request_id']}, desk action {out['action']}, "
          f"status {out['status']}")
    if out.get("card"):
        c = out["card"]
        print(f"[card]   {c['card_id']}: refund {refunds.money(c['proposal']['amount'])} for {c['order']['order_id']} "
              f"({c['order']['qty']} x {c['order']['item']}), policy: {c['policy']}; "
              f"approvers needed: {c['required_approvals']}; proposed by the {c['proposal']['source']}")
        for chk in c["checks"]:
            print(f"[card]   check: {chk}")
        print(f"[next]   python m10/approvals.py show {out['thread_id']}")
    print("\n" + gd.desk_graph.wrap(f"Reply: {out['reply']}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="I want my money back for order A1002.")
    ap.add_argument("--customer", default="Tom B.")
    ap.add_argument("--thread", help="continue a conversation (the thread ID)")
    ap.add_argument("--layer", default="L4", choices=gd.layers.LAYERS)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8110)
    a = ap.parse_args()
    od = OversightDesk(layer=a.layer)
    print(f"[INFO] model: {gd.llm_calls.describe()}")
    print(f"[INFO] desk {VERSION}, {od.desk.describe()}; threads in {THREADS_DB.relative_to(bootstrap.LABS) if THREADS_DB.is_relative_to(bootstrap.LABS) else THREADS_DB}"
          + ("; REFUNDS STOPPED" if refunds.stopped() else ""))
    if a.serve:
        import uvicorn
        print(f"[INFO] the page: http://127.0.0.1:{a.port}/", flush=True)
        uvicorn.run(make_app(od), host="127.0.0.1", port=a.port)
        return
    print(f"Customer ({a.customer}): {a.input}", flush=True)
    try:
        show(od.chat(a.input, customer=a.customer, thread_id=a.thread))
    finally:
        od.close()


if __name__ == "__main__":
    sys.exit(main())
