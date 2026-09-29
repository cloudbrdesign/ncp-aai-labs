"""Version 4 of the support desk: the M3 graph, now with SQL for orders and RAG for manuals.

    python m04/desk_graph.py --input "Where is order A1003?"                          # SQL
    python m04/desk_graph.py --input "My D300 dock shows E42. What does it mean?"     # RAG
    python m04/desk_graph.py --input "The dock from order A1002 won't drive my second screen."   # SQL, then RAG
    python m04/desk_graph.py --bad-draft --input "My D300 dock shows E42."            # watch the citation check
    python m04/desk_graph.py --draw                                                   # Mermaid -> m04/graph.mmd

    recall -> plan -> execute (loops once per step) -> draft <-> critique -> remember

What changed from M3 (m03/desk_graph.py):
  plan      router.py: the menu gains manual_search, and each step names its source
            (SQL, RAG, ticket API). A manual step may carry an order ID instead of a
            product: execute then asks SQL for the order's product and filters on it.
  execute   desk_steps.py: order facts from orders.db through named read-only queries;
            manual passages from Milvus Lite through hybrid search.
  draft     gets the order facts AND the passages, each passage with its chunk ID, and
            must cite the IDs it uses.
  critique  M3's rule checks and groundedness grade, plus two rules for RAG: every cited
            chunk ID was retrieved in this turn, and a reply built from passages cites at
            least one. Failures go back to draft as named feedback, as in M3.
Unchanged: the checkpointer (threads), the store (profile, episodes, lessons) and the
revision limit. They come from m03 (long_term.py, critic.py, handlers.py).
"""
import argparse
import contextlib
import json
import os
import pathlib
import re
import sys
import textwrap
from typing import Annotated, TypedDict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "m03"))   # after m04: m04's llm_calls wins, m03 fills the rest

from langchain_core.messages import AIMessage, HumanMessage, trim_messages  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402

import critic  # noqa: E402   m03/critic.py
import desk_steps  # noqa: E402
import long_term  # noqa: E402   m03/long_term.py
import router  # noqa: E402
import vector_store  # noqa: E402
from llm_calls import chat, describe  # noqa: E402

# M4 keeps its own state folder: m04/state/threads.db and memory.db next to manuals.db
long_term.STATE_DIR = vector_store.STATE_DIR
long_term.MEMORY_DB = vector_store.STATE_DIR / "memory.db"
THREADS_DB = vector_store.STATE_DIR / "threads.db"

DRAFT_PROMPT = (HERE / "prompts" / "draft.txt").read_text()
MAX_HISTORY = 6
MAX_REVISIONS = 2
RECURSION_LIMIT = 25
PLANTED_CHUNK = "H200-specifications-1"   # --bad-draft cites this chunk, which a dock question never retrieves
CITATION = re.compile(r"\b((?:H200|D300|M270|FAQ)-[a-z0-9-]+?-\d+)\b")   # a chunk ID, with or without [ ]
DEMO = "My D300 dock shows E42. What does it mean?"

critic.LESSONS.update({
    "unknown_chunk": "Cite only the passage IDs you were given.",
    "missing_citation": "Cite the passage ID after every manual claim.",
})

SHOW = {"log": True, "plan": False}
TRACE: list[str] = []


def wrap(text: str) -> str:
    """Long model text on several lines, so the log fits a screen."""
    return textwrap.fill(text, width=100, subsequent_indent="    ")


def log(msg: str):
    if SHOW["log"]:
        print(msg, flush=True)


class DeskState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    customer: str
    request: str
    bad_draft: bool
    profile: dict
    new_facts: dict
    episode: dict | None
    lessons: list[str]
    plan: list[dict]
    plan_source: str
    step: int
    evidence: list[dict]
    draft: str
    feedback: list[str]
    revisions: int
    reply: str


def history(state: DeskState) -> list:
    return trim_messages(state["messages"][:-1], max_tokens=MAX_HISTORY, token_counter=len,
                         strategy="last", start_on="human")


def as_text(messages: list) -> str:
    return "\n".join(f"{'Customer' if m.type == 'human' else 'Desk'}: {m.content}" for m in messages)


def facts_text(evidence: list[dict]) -> str:
    return "\n".join(f"- {s}" for s in (desk_steps.sentence(ev) for ev in evidence) if s)


def passages_text(evidence: list[dict]) -> str:
    return "\n".join(f"[{p['id']}] {p['text']}" for p in desk_steps.passages(evidence)) or "(none)"


# ---- nodes ------------------------------------------------------------------------

def recall(state: DeskState, runtime: Runtime) -> dict:
    TRACE.append("recall")
    store, customer = runtime.store, state["customer"]
    stored = long_term.get_profile(store, customer)
    new = long_term.detect_facts(state["request"])
    lessons = [l["text"] for l in long_term.lessons(store, customer)]
    log(f"[recall] profile: {long_term.profile_text(stored) or '(empty)'}"
        + (f" | lessons: {len(lessons)}" if lessons else ""))
    return {"profile": {**stored, **new}, "new_facts": new, "lessons": lessons,
            "episode": long_term.similar_episode(store, customer, state["request"])}


def plan(state: DeskState) -> dict:
    TRACE.append("plan")
    past = history(state)
    carried = router.order_ids(as_text(past))
    p, source = router.make_plan(state["request"], as_text(past), carried, log=log)
    steps = [s.model_dump() for s in p.steps]
    log(f"[plan] {len(steps)} step{'s' if len(steps) > 1 else ''} ({source}): "
        + ", ".join(router.describe_step(s) for s in steps))
    if SHOW["plan"]:
        print(json.dumps({"steps": steps}), flush=True)
    return {"plan": steps, "plan_source": source, "step": 0, "evidence": []}


def execute(state: DeskState) -> dict:
    TRACE.append("execute")
    i, steps = state["step"], state["plan"]
    s = steps[i]
    log(f"[execute] step {i + 1}/{len(steps)} {router.describe_step(s)}")
    ev = desk_steps.run_step(s, state["request"], log=log)
    log(f"[execute]   -> {desk_steps.short(ev)}")
    return {"step": i + 1, "evidence": state["evidence"] + [ev]}


def after_execute(state: DeskState) -> str:
    return "execute" if state["step"] < len(state["plan"]) else "draft"


def draft(state: DeskState) -> dict:
    TRACE.append("draft")
    profile, rules = state["profile"], []
    if profile.get("contact") == "email":
        rules.append("- The customer wants email only. Never offer a phone call.")
    rules += [f"- Lesson from earlier: {l}" for l in state.get("lessons", [])]
    rules += [f"- Fix this: {f}" for f in state.get("feedback", [])]
    system = DRAFT_PROMPT.format(rules="\n".join(rules), facts=facts_text(state["evidence"]) or "(none)",
                                 passages=passages_text(state["evidence"]))
    user = as_text(history(state) + [HumanMessage(state["request"])])
    try:
        text, source = chat([("system", system), ("user", user)]), "model"
    except Exception as e:
        text, source = desk_steps.template_reply(state["evidence"], profile), "template"
        log(f"[draft] model failed ({type(e).__name__}); using the template reply")
    if state.get("bad_draft") and state.get("revisions", 0) == 0:
        text += f" The specifications list the details [{PLANTED_CHUNK}]."
        log(f"[draft] --bad-draft: cited {PLANTED_CHUNK}, which was not retrieved")
    label = f"revision {state['revisions']}" if state.get("revisions") else "first draft"
    log(wrap(f"[draft] {label} ({source}): {text}"))
    return {"draft": text}


def rag_checks(draft_text: str, retrieved: list[str]) -> list[str]:
    """The two RAG rules: cite only retrieved chunks, and cite at least one when passages were used."""
    cited = list(dict.fromkeys(CITATION.findall(draft_text)))
    feedback = [f"unknown_chunk: [{c}] was not retrieved; cite only the passage IDs given" for c in cited
                if c not in retrieved]
    if retrieved and not cited:
        feedback.append("missing_citation: the reply cites no passage; add the passage ID after each manual claim")
    return feedback


def critique(state: DeskState, runtime: Runtime) -> dict:
    TRACE.append("critique")
    ev = state["evidence"]
    planned = [s["order_id"] for s in state["plan"] if s.get("order_id")]
    known = [e["order_id"] for e in ev if e.get("order_id") and not e.get("error")]
    retrieved = [p["id"] for p in desk_steps.passages(ev)]
    feedback = critic.checks(state["draft"], planned, known, state["profile"])
    feedback += rag_checks(state["draft"], retrieved)
    score = critic.grade(f"{facts_text(ev)}\nManual passages:\n{passages_text(ev)}", state["draft"], log=log)
    if score == 0:
        feedback.append("grounding: the reply contradicts the facts or passages; restate only what they say")
    if not feedback:
        cited = CITATION.findall(state["draft"])
        cites = f"{len(cited)} citation{'s' if len(cited) != 1 else ''}, all retrieved" if retrieved else "no passages"
        log(f"[critique] PASS: order IDs answered, none invented, {cites}, grounded")
        return {"feedback": []}
    for f in feedback:
        log(f"[critique] FAIL {f}")
        kept = long_term.add_lesson(runtime.store, state["customer"], critic.lesson_for(f))
        log(f"[critique] lesson saved ({len(kept)}/{long_term.MAX_LESSONS} kept): {critic.lesson_for(f)}")
    return {"feedback": feedback, "revisions": state.get("revisions", 0) + 1}


def after_critique(state: DeskState) -> str:
    return "draft" if state["feedback"] and state["revisions"] <= MAX_REVISIONS else "remember"


def remember(state: DeskState, runtime: Runtime) -> dict:
    TRACE.append("remember")
    store, customer = runtime.store, state["customer"]
    reply = state["draft"]
    if state.get("feedback"):
        reply = desk_steps.template_reply(state["evidence"], state["profile"])
        log(f"[remember] still failing after {MAX_REVISIONS} revisions: sending the template reply")
    if state.get("new_facts"):
        long_term.save_profile(store, customer, state["new_facts"])
    long_term.add_episode(store, customer, state["request"], state["plan"], reply)
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
    vector_store.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        saver = (stack.enter_context(SqliteSaver.from_conn_string(str(THREADS_DB))) if memory == "sqlite"
                 else InMemorySaver())
        store = stack.enter_context(long_term.open_store())
        yield build_graph().compile(checkpointer=saver, store=store)


def run_turn(app, thread: str, customer: str, request: str, bad_draft: bool = False) -> dict:
    turn = {"messages": [HumanMessage(request)], "customer": customer, "request": request,
            "bad_draft": bad_draft, "plan": [], "step": 0, "evidence": [], "draft": "", "feedback": [],
            "revisions": 0, "reply": ""}
    return app.invoke(turn, {"configurable": {"thread_id": thread}, "recursion_limit": RECURSION_LIMIT},
                      durability="sync")


def draw() -> str:
    mermaid = build_graph().compile().get_graph().draw_mermaid()
    (HERE / "graph.mmd").write_text(mermaid)
    return mermaid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEMO, help="the customer's message")
    ap.add_argument("--thread", default=f"m4-{os.getpid()}", help="thread ID (default: a new thread per run)")
    ap.add_argument("--customer", default="tom", help="customer ID: whose long-term memory to use")
    ap.add_argument("--memory", choices=["sqlite", "inmemory"], default="sqlite")
    ap.add_argument("--show-plan", action="store_true", help="print the plan as JSON")
    ap.add_argument("--bad-draft", action="store_true", help=f"cite {PLANTED_CHUNK} in the first draft")
    ap.add_argument("--draw", action="store_true", help="print the graph as Mermaid and write m04/graph.mmd")
    a = ap.parse_args()
    if a.draw:
        print(draw())
        return
    SHOW["plan"] = a.show_plan
    print(f"[INFO] model: {describe()} | thread: {a.thread} | customer: {a.customer}")
    print(f"Customer: {a.input}", flush=True)
    with open_desk(a.memory) as app:
        try:
            out = run_turn(app, a.thread, a.customer, a.input, a.bad_draft)
        except desk_steps.handlers.TicketApiError as e:
            print(f"[FAIL] the run stopped in execute: {e}")
            raise SystemExit(1)
    print("\n" + wrap(f"Reply: {out['reply']}"))


if __name__ == "__main__":
    main()
