"""The M5 desk for `nat eval`, plus a sidecar file with what each turn retrieved.

    python m06/desk_app.py --input "My D300 dock shows E42. What does it mean?"
    OLLAMA_MODEL=llama3.2:1b python m06/desk_app.py --input "Where is order A1003?"
    nat eval --config_file m06/configs/eval_A.yml --reps 3        # loads `agent` from here

`agent` is a one-node graph around M5's `agent` (m05/desk_app.py: the M4 desk as a
LangGraph graph). NAT's langgraph_wrapper sees only the final reply: the wrapped graph's
steps (the plan, the SQL rows, the retrieved passages) are not visible to NAT's
evaluators. Ragas' context metrics need the passages, so every turn under `nat eval` also
appends one record to a sidecar file, <the eval's output_dir>/passages.jsonl:

    item_id, rep      the test-set item (looked up by its question) and which repetition
    question, reply   what went in and came out
    route             the plan's steps, like "manual_search D300 (rag)", and who made the plan
    passages          the retrieved chunks the draft saw: [{"id", "text"}, ...]
    latency_s         wall-clock time of the whole turn
    model             the desk's model and sampling (DESK_TEMPERATURE, DESK_SEED)
    error             set when the turn failed

How the sidecar finds the output folder: the same way nat does, from the command line
(`--override eval.general.output_dir DIR` if given, else the config file's value).
M06_SIDECAR=path overrides it. `nat run` and plain Python calls write no sidecar. The
first record of a run empties the file, so the file always holds one run.

The desk's model comes from m06/llm_calls.py: OLLAMA_MODEL (configuration B is
llama3.2:1b), DESK_TEMPERATURE and DESK_SEED.
"""
import argparse
import collections
import importlib.util
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402  (first: the M4 desk must get this chat())
import testset  # noqa: E402

_spec = importlib.util.spec_from_file_location("m05_desk_app", LABS / "m05" / "desk_app.py")
m05 = importlib.util.module_from_spec(_spec)
sys.modules["m05_desk_app"] = m05
_spec.loader.exec_module(m05)          # M5's desk: the M4 graph as `agent`

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

desk_graph = m05.desk_graph            # m04/desk_graph.py
desk_steps = m05.desk_steps            # m04/desk_steps.py
build_index = m05.build_index
router = desk_graph.router             # m04/router.py
_run = {"path": None, "started": False, "reps": collections.Counter(), "items": None}


def close() -> None:
    """Free the Milvus Lite file so another process (nat) can open it (see m05/desk_app.py)."""
    m05.close()


# ---- the sidecar ------------------------------------------------------------------------

def _argv_value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def output_dir_from_argv(argv: list[str] | None = None) -> pathlib.Path | None:
    """The eval output folder of this `nat eval` process, or None when not under nat eval."""
    argv = sys.argv if argv is None else argv
    if "eval" not in argv[1:3]:
        return None
    for i, a in enumerate(argv):
        if a == "--override" and i + 2 < len(argv) and argv[i + 1] == "eval.general.output_dir":
            return pathlib.Path(argv[i + 2])
    config = _argv_value(argv, "--config_file")
    if not config or not pathlib.Path(config).exists():
        return None
    import yaml
    data = yaml.safe_load(pathlib.Path(config).read_text()) or {}
    out = ((data.get("eval") or {}).get("general") or {}).get("output_dir")
    return pathlib.Path(out) if out else None


def sidecar_path() -> pathlib.Path | None:
    if os.environ.get("M06_SIDECAR"):
        return pathlib.Path(os.environ["M06_SIDECAR"])
    out = output_dir_from_argv()
    return out / "passages.jsonl" if out else None


def write_sidecar(record: dict) -> None:
    path = _run["path"] if _run["started"] else sidecar_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if _run["started"] else "w") as f:     # the first record of a run empties the file
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _run.update(path=path, started=True)


def item_for(question: str) -> dict | None:
    if _run["items"] is None:
        _run["items"] = testset.by_question()
    return _run["items"].get(testset.norm(question))


# ---- one turn ---------------------------------------------------------------------------

async def answer(question: str) -> dict:
    """One desk turn through M5's agent; returns the sidecar record (reply, route, passages...)."""
    item = item_for(question) or {}
    t = time.perf_counter()
    rec = {"item_id": item.get("id"), "category": item.get("category"), "question": question,
           "model": llm_calls.model_name(), "temperature": llm_calls.temperature(), "seed": llm_calls.seed()}
    try:
        out = await m05.ask(question)
    except Exception as e:                       # recorded for triage (infra), then passed on to NAT
        rec.update(reply="", route=[], plan_source="", passages=[], latency_s=round(time.perf_counter() - t, 3),
                   error=f"{type(e).__name__}: {str(e)[:200]}")
        raise_later = e
    else:
        raise_later = None
        passages = desk_steps.passages(out.get("evidence", []))
        rec.update(reply=out.get("reply", ""),
                   route=[router.describe_step(s) for s in out.get("plan", [])],
                   plan_source=out.get("plan_source", ""),
                   passages=[{"id": p["id"], "text": p["text"]} for p in passages],
                   latency_s=round(time.perf_counter() - t, 3), error="")
    if rec["item_id"]:
        rec["rep"] = _run["reps"][rec["item_id"]]
        _run["reps"][rec["item_id"]] += 1
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["_raise"] = raise_later
    return rec


async def desk(state: MessagesState) -> dict:
    rec = await answer(state["messages"][-1].content)
    err = rec.pop("_raise")
    write_sidecar(rec)
    if err is not None:
        raise err
    return {"messages": [AIMessage(rec["reply"])]}


def build():
    g = StateGraph(MessagesState)
    g.add_node("desk", desk)
    g.add_edge(START, "desk")
    g.add_edge("desk", END)
    # M5's graph runs inside this graph's node, so LangGraph treats it as a subgraph and hands it
    # this graph's store (the M4 recall and remember steps need one). See m05/desk_app.guarded_agent.
    return g.compile(store=InMemoryStore())


agent = build()


def ask(question: str) -> dict:
    """One turn outside NAT (feedback.py, the CLI). Returns the record; raises if the turn failed."""
    import asyncio
    rec = asyncio.run(answer(question))
    err = rec.pop("_raise")
    if err is not None:
        raise err
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="My D300 dock shows E42. What does it mean?")
    ap.add_argument("--log", action="store_true", help="show the desk's steps (like M05_DESK_LOG=1)")
    a = ap.parse_args()
    desk_graph.SHOW["log"] = a.log
    testset.ensure_index()
    print(f"[INFO] model: {llm_calls.describe()}")
    rec = ask(a.input)
    print(f"[route] {', '.join(rec['route']) or '-'} ({rec['plan_source']})")
    print(f"[rag]   {', '.join(p['id'] for p in rec['passages']) or 'no passages'}")
    print(f"[time]  {rec['latency_s']} s" + (f" | test-set item {rec['item_id']}" if rec["item_id"] else ""))
    print(desk_graph.wrap(f"Reply: {rec['reply']}"))
    close()


if __name__ == "__main__":
    main()
