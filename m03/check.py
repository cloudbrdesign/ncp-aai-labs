"""Check that the M03 lab works end to end.

    python m03/check.py          # Ollama running (or the NVIDIA variables set); nothing else to start

0. The model answers (otherwise every check below would only test the fallbacks).
1. The reasoning check scores direct, chain-of-thought and program-aided answers (3.1).
2. The plan is valid JSON with 1-4 known steps; an invalid plan falls back to the regex planner (3.2).
3. A thread survives a restart (SqliteSaver); a new thread and InMemorySaver start empty (3.3).
4. A new thread for the same customer recalls the profile; an episode is written (3.3).
5. The thread has more than one checkpoint; a fork leaves the original checkpoint unchanged (3.4).
6. After a forced ticket-API failure, re-invoking resumes without re-running plan (3.4).
7. --bad-draft is caught by the critique and revised; lessons never exceed 3 (3.5).
8. The NAT agent's get_memory returns the preference the LangGraph desk saved (3.3).

It uses the ticket API from m02 on port 8767 (it starts and stops it itself).
"""
import asyncio
import logging
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_graph  # noqa: E402
import long_term  # noqa: E402
import planner  # noqa: E402
import reasoning_check  # noqa: E402
import time_travel  # noqa: E402
from llm_calls import chat, describe, fake_mode  # noqa: E402

RUN = str(int(time.time()))
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)


def desk(*args) -> str:
    """Run desk_graph.py as a new process (a real restart) and return its output."""
    p = subprocess.run([sys.executable, str(HERE / "desk_graph.py"), *args],
                       capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr


def plan_line(out: str) -> str:
    return next((l for l in out.splitlines() if l.startswith("[plan]") and "step" in l), "(no plan line)")


def main():
    # 0. the model must answer, or every step below would quietly use its fallback
    try:
        reply = chat([("user", "Reply with the single word OK.")])
        check(f"Model answers ({describe()})", bool(reply.strip()))
    except Exception as e:
        check(f"Model answers ({describe()})", False,
              f"{type(e).__name__}: {e}. Start Ollama (ollama serve) and pull llama3.2:3b, "
              "or set the NVIDIA variables.")
        return

    # 1. reasoning
    scores = reasoning_check.run(show=lambda *_: None)
    check("Reasoning check scored all three methods (" + ", ".join(f"{m} {s}/5" for m, s in scores.items()) + ")",
          len(scores) == 3)

    # 2. planning
    lines = []
    plan, source = planner.make_plan(desk_graph.MULTI_PART, "", [], log=lines.append)
    valid = planner.Plan.model_validate_json(plan.model_dump_json())
    check(f"Plan is valid JSON with 1-4 known steps ({len(valid.steps)} steps, from the {source})",
          1 <= len(valid.steps) <= 4 and all(s.action in planner.STEP_TYPES for s in valid.steps), plan)
    saved = planner.chat
    if fake_mode():
        os.environ["M03_FAKE_BAD_PLAN"] = "1"     # the fake model returns an invalid plan
    else:                                        # stand in for an invalid model output
        planner.chat = lambda *a, **k: planner.Plan.model_validate({"steps": [{"action": "refund_now"}] * 5})
    try:
        lines.clear()
        plan, source = planner.make_plan(desk_graph.MULTI_PART, "", [], log=lines.append)
    finally:
        planner.chat = saved
        os.environ.pop("M03_FAKE_BAD_PLAN", None)
    check("An invalid plan is rejected and the regex fallback is logged",
          source == "fallback" and any("regex fallback" in l for l in lines) and len(plan.steps) == 2, lines)
    mermaid = desk_graph.draw()
    check("Graph drawn as Mermaid (m03/graph.mmd) with the execute loop and the critique edge",
          "execute -.-> execute" in mermaid and "critique -.-> draft" in mermaid, mermaid)

    # 3. short-term memory across a restart
    thread = f"chk-{RUN}"
    desk("--thread", thread, "--input", "Where is order A1001?")
    out = desk("--thread", thread, "--input", "And when will it arrive?")
    check("Same thread after a restart still knows order A1001 (SqliteSaver)", "A1001" in plan_line(out), out[-600:])
    out = desk("--thread", f"{thread}-new", "--input", "And when will it arrive?")
    check("A new thread starts empty", "A1001" not in plan_line(out), plan_line(out))
    desk("--thread", f"{thread}-ram", "--memory", "inmemory", "--input", "Where is order A1001?")
    out = desk("--thread", f"{thread}-ram", "--memory", "inmemory", "--input", "And when will it arrive?")
    check("InMemorySaver forgets the thread after a restart", "A1001" not in plan_line(out), plan_line(out))

    # 4. long-term memory, 5. time travel, 6. resume, 7. critique
    desk_graph.SHOW["log"] = False
    time_travel.say = lambda *a, **k: None
    customer = f"chk-{RUN}"
    with desk_graph.open_desk() as app:
        desk_graph.run_turn(app, f"{thread}-a", customer, "Can I return order A1002? Please contact me by email only.")
        out = desk_graph.run_turn(app, f"{thread}-b", customer, "Where is order A1001?")
        check("A new thread for the same customer recalls 'email only' from the profile",
              out["profile"].get("contact") == "email", out["profile"])
        with long_term.open_store() as store:
            n = len(long_term.episodes(store, customer))
        check(f"Episodes are written to long-term memory ({n} for this customer)", n == 2, n)

        tt = time_travel.history_part(app)
        check(f"Thread history has more than one checkpoint ({tt['checkpoints']})", tt["checkpoints"] > 1)
        check("A fork (update_state) leaves the original checkpoint unchanged", tt["fork_kept_original"], tt)

        r = time_travel.resume_part(app)
        check("Forced ticket-API failure stopped the run in execute", r["failed"], r)
        check("Re-invoking resumed from the checkpoint without re-running plan",
              r["failed"] and "plan" not in r["resumed"] and "execute" in r["resumed"], r)

        lines = []
        desk_graph.log = lines.append
        desk_graph.TRACE.clear()
        out = desk_graph.run_turn(app, f"{thread}-bad", customer, "Can I return order A1002?", bad_draft=True)
        caught = any("FAIL unknown_order: A1009" in l for l in lines)
        check("--bad-draft: the critique caught the planted order ID and the draft was revised",
              caught and desk_graph.TRACE.count("draft") >= 2 and "A1009" not in out["reply"], "\n".join(lines))
        for i in range(5):
            long_term.add_lesson(app.store, customer, f"test lesson {i}")
        kept = [l["text"] for l in long_term.lessons(app.store, customer)]
        check("Lessons never exceed 3 (the oldest drops off)",
              kept == ["test lesson 2", "test lesson 3", "test lesson 4"], kept)

        # 8. NAT memory: the desk saves tom's preference, the NAT agent's tool reads it
        desk_graph.run_turn(app, f"{thread}-tom", "tom", "Where is order A1002? Please contact me by email only.")
    asyncio.run(nat_memory())


async def nat_memory():
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    logging.getLogger("nat").setLevel(logging.CRITICAL)
    try:
        cfg = load_config(HERE / "configs" / "support_memory.yml")
        async with WorkflowBuilder.from_config(cfg) as builder:
            get_memory = await builder.get_function("get_memory")
            found = await get_memory.ainvoke({"query": "contact preference", "top_k": 3})
            check("NAT get_memory returns the preference saved by desk_graph.py (email only)", "email only" in found,
                  found[:300])
            add_memory = await builder.get_function("add_memory")
            await add_memory.ainvoke({"memory": f"Check note {RUN}: prefers morning deliveries."})
        with long_term.open_store() as store:
            notes = [i.value["text"] for i in store.search(long_term.ns("tom", "notes"), limit=100)]
        check("NAT add_memory writes into the same store", any(RUN in t for t in notes), notes[-3:])
    except Exception as e:
        check("NAT memory provider desk_memory works", False,
              f"{type(e).__name__}: {e}. Install it first: pip install -e m03/desk_memory")


if __name__ == "__main__":
    print(f"NCP-AAI M03 lab check  [model: {describe()}]\n")
    main()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
