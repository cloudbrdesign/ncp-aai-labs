"""Stateful orchestration: checkpoint history, replay, fork, and resume after a failure.

    python m03/time_travel.py                  # both parts
    python m03/time_travel.py --part history   # list, replay, fork
    python m03/time_travel.py --part resume    # forced failure, then resume

Part 1: run one request, list the thread's checkpoints (one per super-step), replay
from the checkpoint just before `draft`, then fork that checkpoint with update_state
(the customer now prefers a phone call). The fork is a new branch; the original
checkpoint is never changed.

Part 2: start the M2 ticket API with --fail-first 1, so the first ticket request gets
HTTP 503. The run stops inside `execute`. Re-invoking the thread with no new input
continues from the last checkpoint: `plan` does not run again. durability="sync"
makes sure every finished step was written before the next one started.
"""
import argparse
import pathlib
import subprocess
import sys
import time

import httpx

import desk_graph
import handlers
from desk_graph import TRACE, config, open_desk, run_turn
from llm_calls import describe

HERE = pathlib.Path(__file__).resolve().parent
CUSTOMER = "lena"
PORT = 8767   # its own ticket API, so it doesn't clash with one you already run on 8765


def say(msg=""):
    print(msg, flush=True)


def history_part(app) -> dict:
    thread = f"tt-{int(time.time())}"
    request = "Where is order A1001, and can I return A1002? Please reply by email only."
    say(f"[INFO] run on thread {thread}: {request}")
    first = run_turn(app, thread, CUSTOMER, request)
    say(f"[INFO] reply: {first['reply']}\n")

    snapshots = list(app.get_state_history(config(thread)))       # newest first
    say(f"[INFO] {len(snapshots)} checkpoints in thread {thread} (newest first):")
    for s in snapshots:
        nxt = ",".join(s.next) or "(end)"
        say(f"       step {s.metadata['step']:>2}  next={nxt:<9} id=...{s.config['configurable']['checkpoint_id'][-8:]}")
    before_draft = [s for s in snapshots if s.next == ("draft",)][-1]   # the earliest one
    cid = before_draft.config["configurable"]["checkpoint_id"][-8:]

    TRACE.clear()
    replay = app.invoke(None, before_draft.config, durability="sync")
    say(f"\n[INFO] replay from ...{cid} (next=draft): ran {' -> '.join(TRACE)}; recall, plan, execute were not re-run")
    say(f"[INFO] replayed reply: {replay['reply']}")

    phone = {**before_draft.values["profile"], "contact": "phone"}
    fork_config = app.update_state(before_draft.config, {"profile": phone})
    TRACE.clear()
    fork = app.invoke(None, fork_config, durability="sync")
    say(f"\n[INFO] fork: update_state(profile.contact=phone) on ...{cid} -> new checkpoint "
        f"...{fork_config['configurable']['checkpoint_id'][-8:]}; ran {' -> '.join(TRACE)}")
    say(f"[INFO] forked reply: {fork['reply']}")

    original = app.get_state(before_draft.config).values["profile"].get("contact")
    kept = original == "email"
    say(f"[{'PASS' if kept else 'FAIL'}] original checkpoint ...{cid} still has contact={original} "
        "(update_state branches, it never rolls back)")
    total = len(list(app.get_state_history(config(thread))))
    say(f"[INFO] thread {thread} now has {total} checkpoints: the first run, the replay and the fork")
    return {"checkpoints": len(snapshots), "fork_kept_original": kept, "fork_contact": fork["profile"]["contact"]}


def start_ticket_api(fail_first: int) -> subprocess.Popen:
    api = subprocess.Popen([sys.executable, str(HERE.parent / "m02" / "ticket_api.py"), "--port", str(PORT),
                            "--fail-first", str(fail_first)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            httpx.get(f"http://localhost:{PORT}/health", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    handlers.TICKET_API["url"] = f"http://localhost:{PORT}"
    return api


def resume_part(app) -> dict:
    api = start_ticket_api(fail_first=1)
    say(f"[INFO] started the M2 ticket API on port {PORT} with --fail-first 1")
    try:
        thread = f"tr-{int(time.time())}"
        request = "Where is order A1001? Also, my USB-C dock from order A1002 arrived cracked."
        say(f"[INFO] run on thread {thread}: {request}")
        TRACE.clear()
        failed = False
        try:
            run_turn(app, thread, CUSTOMER, request)
        except handlers.TicketApiError as e:
            failed = True
            say(f"[FAIL] the run stopped in execute: {e} (simulated outage)")
        first_run = list(TRACE)
        state = app.get_state(config(thread))
        say(f"[INFO] last checkpoint: next={','.join(state.next)}, plan has {len(state.values['plan'])} steps, "
            f"{state.values['step']} done")

        TRACE.clear()
        out = app.invoke(None, config(thread), durability="sync")   # no new input: continue the thread
        resumed = list(TRACE)
        say(f"[INFO] re-invoked the thread: ran {' -> '.join(resumed)}")
        ok = failed and "plan" not in resumed and "plan" in first_run
        say(f"[{'PASS' if ok else 'FAIL'}] resumed from the last checkpoint; plan did not run again")
        say(f"[INFO] reply: {out['reply']}")
        return {"failed": failed, "first_run": first_run, "resumed": resumed, "reply": out["reply"]}
    finally:
        api.terminate()
        handlers.TICKET_API["url"] = "http://localhost:8765"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["history", "resume", "both"], default="both")
    a = ap.parse_args()
    desk_graph.SHOW["log"] = False   # keep this demo to its own lines
    say(f"[INFO] model: {describe()} | customer: {CUSTOMER} | checkpoints: {desk_graph.THREADS_DB.name}\n")
    with open_desk("sqlite") as app:
        if a.part in ("history", "both"):
            history_part(app)
        if a.part == "both":
            say()
        if a.part in ("resume", "both"):
            resume_part(app)


if __name__ == "__main__":
    main()
