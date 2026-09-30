"""The M4 desk as a graph NeMo Agent Toolkit can run: `langgraph_wrapper` loads `agent` from here.

    nat run  --config_file m05/configs/desk_eval.yml --input "Where is order A1003?"
    nat eval --config_file m05/configs/desk_eval.yml
    python m05/desk_app.py --input "Where is order A1003?"      # the same graph without NAT

The wrapper sends the graph one thing, {"messages": [...]}, and reads the last message
back. The M4 graph needs more: the customer, the request text and empty turn fields
(m04/desk_graph.run_turn fills them). So `agent` is a small parent graph:

    START -> prepare (fills the M4 turn from the last message) -> desk (the M4 graph, unchanged) -> END

It is compiled without a checkpointer (every eval item is a new conversation) and with an
in-memory store, because the M4 recall and remember steps read and write long-term memory.
Each run gets its own customer ID, so lessons from one eval item don't leak into the next.

`guarded_agent` is the same desk behind NeMo Guardrails (guarded_desk.py), for
configs/desk_eval_guarded.yml. NAT accepts a function that returns a compiled graph, which
lets guarded_desk.py load only when that config asks for it.
"""
import argparse
import os
import pathlib
import sys
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402,F401  (first: the M4 desk must get M5's chat())

sys.path.insert(1, str(HERE.parent / "m04"))
import desk_graph  # noqa: E402   m04/desk_graph.py (puts m03 on the path too)
import desk_steps  # noqa: E402
import make_orders_db  # noqa: E402
import vector_store  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

desk_graph.SHOW["log"] = os.environ.get("M05_DESK_LOG") == "1"   # quiet inside NAT; M05_DESK_LOG=1 to see the steps


def index_ready() -> bool:
    if not (make_orders_db.DB.exists() and vector_store.DB_PATH.exists()):
        return False
    client = vector_store.connect()
    try:
        return client.has_collection(vector_store.COLLECTION)
    finally:
        client.close()


def build_index(say=print) -> None:
    """Rebuild what the M4 desk reads: the cleaned manuals, the Milvus Lite index and orders.db."""
    import clean
    import ingest
    quiet = lambda *a, **k: None  # noqa: E731
    clean.clean(say=quiet)
    res = ingest.ingest(clean.CLEAN_DIR, vector_store.COLLECTION, say=quiet)
    make_orders_db.build()
    say(f"[index] {res['count']} manual chunks in {vector_store.DB_PATH}; orders in {make_orders_db.DB}")


def close() -> None:
    """Free the Milvus Lite file, so another process (nat) can open it.

    Milvus Lite runs a small local server per .db file and locks the file until that server
    stops; closing the client is not enough. Only one process can use the file at a time.
    """
    from milvus_lite.server_manager import server_manager_instance
    client = desk_steps._milvus.pop("client", None)
    if client is not None:
        client.close()
    server_manager_instance.release_server(str(vector_store.DB_PATH))


def prepare(state: desk_graph.DeskState) -> dict:
    request = state["messages"][-1].content
    return {"customer": f"nat-{uuid.uuid4().hex[:8]}", "request": request, "bad_draft": False, "plan": [],
            "step": 0, "evidence": [], "draft": "", "feedback": [], "revisions": 0, "reply": ""}


def build_bare():
    g = StateGraph(desk_graph.DeskState)
    g.add_node("prepare", prepare)
    g.add_node("desk", desk_graph.build_graph().compile())   # the M4 graph as one node
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "desk")
    g.add_edge("desk", END)
    return g.compile(store=InMemoryStore())


agent = build_bare()


async def ask(question: str) -> dict:
    """One desk turn through `agent`; returns the final M4 state (reply, evidence, plan...)."""
    return await agent.ainvoke({"messages": [HumanMessage(question)]},
                               {"recursion_limit": desk_graph.RECURSION_LIMIT})


def guarded_agent(config=None):
    """The guarded desk as a one-node graph: the node sends the last message through the rails."""
    import guarded_desk
    rails = guarded_desk.build_rails()

    async def guarded(state: MessagesState) -> dict:
        out = await guarded_desk.respond(rails, state["messages"][-1].content)
        return {"messages": [AIMessage(out["reply"])]}

    g = StateGraph(MessagesState)
    g.add_node("guarded", guarded)
    g.add_edge(START, "guarded")
    g.add_edge("guarded", END)
    # The desk graph runs inside this graph's node, so LangGraph treats it as a subgraph and
    # hands it this graph's store. Without one here, recall would get store=None.
    return g.compile(store=InMemoryStore())


def main():
    import asyncio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Where is order A1003?")
    a = ap.parse_args()
    desk_graph.SHOW["log"] = True
    if not index_ready():
        build_index()
    print(f"[INFO] model: {llm_calls.describe()}")
    out = asyncio.run(ask(a.input))
    print(desk_graph.wrap(f"Reply: {out['reply']}"))


if __name__ == "__main__":
    main()
