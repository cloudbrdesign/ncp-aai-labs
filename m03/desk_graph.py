"""Version 3 of the support desk: a LangGraph state graph that plans, remembers and checks itself.

    python m03/desk_graph.py --input "Where is A1001, and can I return A1002?" --show-plan
    python m03/desk_graph.py --thread t1 --input "Where is order A1001?"
    python m03/desk_graph.py --thread t1 --input "And when will it arrive?"     # after a restart
    python m03/desk_graph.py --bad-draft --input "Can I return order A1002?"   # watch the critique loop
    python m03/desk_graph.py --draw                                            # Mermaid -> m03/graph.mmd

    recall -> plan -> execute (loops once per step) -> draft <-> critique -> remember

Two kinds of memory:
  checkpointer = short-term memory per thread. After every node, the graph state (the
                 conversation, the plan, the evidence) is saved under the thread ID in
                 m03/state/threads.db. Same --thread after a restart: the desk still knows.
  store        = long-term memory per customer, shared by all threads (long_term.py,
                 m03/state/memory.db): profile, past episodes, lessons.
"""
import argparse
import contextlib
import json
import pathlib
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

import critic
import handlers
import long_term
import planner
from llm_calls import chat, describe

HERE = pathlib.Path(__file__).resolve().parent
THREADS_DB = long_term.STATE_DIR / "threads.db"
DRAFT_PROMPT = (HERE / "prompts" / "draft.txt").read_text()
MAX_HISTORY = 6        # messages the model sees from earlier turns (trimmed before each model call)
MAX_REVISIONS = 2      # critique may send the draft back at most this many times
RECURSION_LIMIT = 25   # hard stop for the execute and critique loops
PLANTED_ID = "A1009"   # --bad-draft adds this order ID, which is in no evidence
MULTI_PART = "Where is my order A1001, and can I return A1002?"

SHOW = {"log": True, "plan": False}
TRACE: list[str] = []   # the nodes that ran, in order (time_travel.py and check.py read it)


def log(msg: str):
    if SHOW["log"]:
        print(msg, flush=True)


class DeskState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # the thread's conversation; add_messages appends
    customer: str
    request: str
    bad_draft: bool
    profile: dict          # from long-term memory (+ facts spotted in this message)
    new_facts: dict        # facts spotted in this message, saved by remember
    episode: dict | None   # the most similar past episode
    lessons: list[str]
    plan: list[dict]
    plan_source: str       # "model" or "fallback"
    step: int              # index of the next plan step
    evidence: list[dict]
    draft: str
    feedback: list[str]
    revisions: int
    reply: str


def history(state: DeskState) -> list:
    """Earlier turns only, trimmed to the last MAX_HISTORY messages (token_counter=len counts messages)."""
    return trim_messages(state["messages"][:-1], max_tokens=MAX_HISTORY, token_counter=len,
                         strategy="last", start_on="human")


def as_text(messages: list) -> str:
    return "\n".join(f"{'Customer' if m.type == 'human' else 'Desk'}: {m.content}" for m in messages)


# ---- nodes ------------------------------------------------------------------------

def recall(state: DeskState, runtime: Runtime) -> dict:
    """Load long-term memory for this customer at the start of the turn."""
    TRACE.append("recall")
    store, customer = runtime.store, state["customer"]
    stored = long_term.get_profile(store, customer)
    new = long_term.detect_facts(state["request"])
    episode = long_term.similar_episode(store, customer, state["request"])
    lessons = [l["text"] for l in long_term.lessons(store, customer)]
    log(f"[recall] profile: {long_term.profile_text(stored) or '(empty)'}")
    if new:
        log(f"[recall] new in this message: {new} (remember will save it)")
    if episode:
        log(f"[recall] most similar past episode ({episode['score']}): \"{episode['request']}\"")
    if lessons:
        log(f"[recall] lessons: {len(lessons)}")
    return {"profile": {**stored, **new}, "new_facts": new, "episode": episode, "lessons": lessons}


def plan(state: DeskState) -> dict:
    TRACE.append("plan")
    past = history(state)
    carried = planner.order_ids(as_text(past))     # a follow-up like "when will it arrive?" uses these
    p, source = planner.make_plan(state["request"], as_text(past), carried, log=log)
    steps = [s.model_dump() for s in p.steps]
    listed = ", ".join(f"{s['action']} {s['order_id']}".strip() for s in steps)
    log(f"[plan] {len(steps)} step{'s' if len(steps) > 1 else ''} ({source}): {listed}")
    if SHOW["plan"]:
        print(json.dumps({"steps": steps}, indent=2), flush=True)
    return {"plan": steps, "plan_source": source, "step": 0, "evidence": []}


def execute(state: DeskState) -> dict:
    """Run ONE plan step. The conditional edge below brings us back until the plan is done."""
    TRACE.append("execute")
    i, steps = state["step"], state["plan"]
    ev = handlers.run_step(steps[i], state["request"])   # may raise TicketApiError: the run stops here
    log(f"[execute] step {i + 1}/{len(steps)} {steps[i]['action']} {steps[i]['order_id']} -> {handlers.short(ev)}")
    return {"step": i + 1, "evidence": state["evidence"] + [ev]}


def after_execute(state: DeskState) -> str:
    """The routing function: another step to run, or on to the draft."""
    return "execute" if state["step"] < len(state["plan"]) else "draft"


def draft(state: DeskState) -> dict:
    TRACE.append("draft")
    profile, rules = state["profile"], []
    if profile.get("contact") == "email":
        rules.append("- The customer wants email only. Never offer a phone call.")
    elif profile.get("contact") == "phone":
        rules.append("- The customer prefers a phone call for follow-ups.")
    if profile.get("language"):
        rules.append(f"- Write in {profile['language']}.")
    rules += [f"- Lesson from earlier: {l}" for l in state.get("lessons", [])]
    rules += [f"- Fix this: {f}" for f in state.get("feedback", [])]
    facts = "\n".join(f"- {handlers.sentence(ev)}" for ev in state["evidence"])
    system = DRAFT_PROMPT.format(rules="\n".join(rules), facts=facts)
    if state.get("episode"):   # episodic memory: how a similar request was handled (steps, not the old
        ep = state["episode"]  # reply text, so old facts can't leak into this answer)
        steps = ", ".join(s["action"] for s in ep.get("steps", [])) or "answer"
        system += f"\nA similar past request ({planner.ORDER_ID.sub('<order>', ep['request'])}) was handled with: {steps}.\n"
    user = as_text(history(state) + [HumanMessage(state["request"])])
    try:
        text, source = chat([("system", system), ("user", user)]), "model"
    except Exception as e:
        text, source = handlers.template_reply(state["evidence"], profile), "template"
        log(f"[draft] model failed ({type(e).__name__}); using the template reply")
    if state.get("bad_draft") and state.get("revisions", 0) == 0:
        text += f" Your other order {PLANTED_ID} is on its way too."
        log(f"[draft] --bad-draft: planted order {PLANTED_ID}, which is in no evidence")
    label = f"revision {state['revisions']}" if state.get("revisions") else "first draft"
    log(f"[draft] {label} ({source}): {text}")
    return {"draft": text}


def critique(state: DeskState, runtime: Runtime) -> dict:
    TRACE.append("critique")
    planned = [s["order_id"] for s in state["plan"] if s["order_id"]]
    known = [ev["order_id"] for ev in state["evidence"] if ev.get("order_id") and not ev.get("error")]
    feedback = critic.checks(state["draft"], planned, known, state["profile"])
    facts = "\n".join(handlers.sentence(ev) for ev in state["evidence"])
    score = critic.grade(facts, state["draft"], log=log)
    if score == 0:
        feedback.append("grounding: the reply contradicts the facts; restate only what they say")
    if not feedback:
        log("[critique] PASS: order IDs answered, none invented, contact preference respected, grounded")
        return {"feedback": []}
    for f in feedback:
        log(f"[critique] FAIL {f}")
        kept = long_term.add_lesson(runtime.store, state["customer"], critic.lesson_for(f))
        log(f"[critique] lesson saved ({len(kept)}/{long_term.MAX_LESSONS} kept): {critic.lesson_for(f)}")
    return {"feedback": feedback, "revisions": state.get("revisions", 0) + 1}


def after_critique(state: DeskState) -> str:
    if state["feedback"] and state["revisions"] <= MAX_REVISIONS:
        return "draft"
    return "remember"


def remember(state: DeskState, runtime: Runtime) -> dict:
    """Write long-term memory in the hot path: new profile facts and this episode."""
    TRACE.append("remember")
    store, customer = runtime.store, state["customer"]
    reply = state["draft"]
    if state.get("feedback"):
        reply = handlers.template_reply(state["evidence"], state["profile"])
        log(f"[remember] still failing after {MAX_REVISIONS} revisions: sending the template reply")
    if state.get("new_facts"):
        long_term.save_profile(store, customer, state["new_facts"])
        log(f"[remember] profile updated: {state['new_facts']}")
    long_term.add_episode(store, customer, state["request"], state["plan"], reply)
    log(f"[remember] episode saved ({len(long_term.episodes(store, customer))} for {customer})")
    return {"reply": reply, "messages": [AIMessage(reply)]}


# ---- the graph --------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(DeskState)
    for name, fn in [("recall", recall), ("plan", plan), ("execute", execute), ("draft", draft),
                     ("critique", critique), ("remember", remember)]:
        g.add_node(name, fn)
    g.add_edge(START, "recall")
    g.add_edge("recall", "plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", after_execute, ["execute", "draft"])
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", after_critique, ["draft", "remember"])
    g.add_edge("remember", END)
    return g


@contextlib.contextmanager
def open_desk(memory: str = "sqlite"):
    """Compile the graph with a checkpointer (short-term) and a store (long-term)."""
    long_term.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if memory == "sqlite":
            saver = stack.enter_context(SqliteSaver.from_conn_string(str(THREADS_DB)))
        else:
            saver = InMemorySaver()   # lives in this process only: gone after a restart
        store = stack.enter_context(long_term.open_store())
        yield build_graph().compile(checkpointer=saver, store=store)


def config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": RECURSION_LIMIT}


def run_turn(app, thread: str, customer: str, request: str, bad_draft: bool = False) -> dict:
    """One customer message. durability="sync" writes each checkpoint before the next step starts."""
    turn = {"messages": [HumanMessage(request)], "customer": customer, "request": request,
            "bad_draft": bad_draft, "plan": [], "step": 0, "evidence": [], "draft": "", "feedback": [],
            "revisions": 0, "reply": ""}
    return app.invoke(turn, config(thread), durability="sync")


def draw() -> str:
    mermaid = build_graph().compile().get_graph().draw_mermaid()
    (HERE / "graph.mmd").write_text(mermaid)
    return mermaid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=MULTI_PART, help="the customer's message")
    ap.add_argument("--thread", default="demo", help="thread ID: the conversation to continue")
    ap.add_argument("--customer", default="tom", help="customer ID: whose long-term memory to use")
    ap.add_argument("--memory", choices=["sqlite", "inmemory"], default="sqlite",
                    help="where thread checkpoints live (long-term memory is always SQLite)")
    ap.add_argument("--show-plan", action="store_true", help="print the plan as JSON")
    ap.add_argument("--bad-draft", action="store_true", help=f"plant order {PLANTED_ID} in the first draft")
    ap.add_argument("--resume", action="store_true", help="continue a thread whose last run failed")
    ap.add_argument("--draw", action="store_true", help="print the graph as Mermaid and write m03/graph.mmd")
    a = ap.parse_args()
    if a.draw:
        print(draw())
        return
    SHOW["plan"] = a.show_plan
    print(f"[INFO] model: {describe()} | thread: {a.thread} | customer: {a.customer} | checkpoints: {a.memory}")
    with open_desk(a.memory) as app:
        try:
            if a.resume:
                out = app.invoke(None, config(a.thread), durability="sync")
            else:
                print(f"Customer: {a.input}", flush=True)
                out = run_turn(app, a.thread, a.customer, a.input, a.bad_draft)
        except handlers.TicketApiError as e:
            print(f"[FAIL] the run stopped in execute: {e}\n"
                  f"[INFO] resume from the last checkpoint: python m03/desk_graph.py --thread {a.thread} --resume")
            raise SystemExit(1)
        print(f"\nReply: {out['reply']}")


if __name__ == "__main__":
    main()
