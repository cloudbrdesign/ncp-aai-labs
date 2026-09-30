"""Run the labelled cases in data/rails_cases.jsonl through the rails and count right and wrong.

    python m05/rails_check.py
    python m05/rails_check.py --verbose        # one line per case

User messages (direction "input", labels allow / block / off_topic):
  1. check_async(messages, rail_types=[RailType.INPUT]) runs only the input rail (self
     check input) and returns PASSED or BLOCKED with the rail's name. No reply is generated.
  2. If the input rail passes, the dialog rail decides: generate_async with only the dialog
     rails switched on. The canned off-topic refusal means "off_topic". The desk is not
     called here: desk_answer is a stub that returns a fixed text.

Replies (direction "output", labels allow / block): check_async with RailType.OUTPUT runs
self check output and self check facts. The retrieved passages go in first as a
{"role": "context", "content": {"relevant_chunks": ..., "check_facts": True}} message:
the facts rail compares the reply with relevant_chunks, and it only runs when
$check_facts is True (without it, check_async skips the facts rail and says PASSED).

A 3B model does not answer the yes/no checks the same way every time: run it twice and
compare. Self-check rails show how the rails work; they are not security on their own.
"""
import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402
import guarded_desk  # noqa: E402
from nemoguardrails.actions.actions import ActionResult  # noqa: E402
from nemoguardrails.rails.llm.options import RailStatus, RailType  # noqa: E402

CASES = HERE / "data" / "rails_cases.jsonl"
DIALOG_ONLY = {"rails": {"input": False, "output": False, "dialog": True}, "log": {"llm_calls": True}}
STUB = "(stub: the desk would answer here)"


async def stub_answer(question: str) -> ActionResult:
    return ActionResult(return_value=STUB)


def load_cases(path: pathlib.Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build() -> "guarded_desk.LLMRails":
    return guarded_desk.build_rails(answer=stub_answer)


async def check_input(rails, text: str) -> dict:
    res = await rails.check_async([{"role": "user", "content": text}], rail_types=[RailType.INPUT])
    if res.status == RailStatus.BLOCKED:
        return {"predicted": "block", "rail": res.rail, "llm_calls": None}
    gen = await rails.generate_async(messages=[{"role": "user", "content": text}], options=DIALOG_ONLY)
    reply = gen.response[-1]["content"] if gen.response else ""
    calls = [c.task for c in gen.log.llm_calls]
    predicted = "off_topic" if reply.startswith(guarded_desk.REFUSALS[0]) else "allow"
    return {"predicted": predicted, "rail": "dialog" if predicted == "off_topic" else "", "llm_calls": calls,
            "reply": reply}


async def check_output(rails, case: dict) -> dict:
    messages = [{"role": "context", "content": {"relevant_chunks": case["context"], "check_facts": True}},
                {"role": "user", "content": case["question"]},
                {"role": "assistant", "content": case["text"]}]
    res = await rails.check_async(messages, rail_types=[RailType.OUTPUT])
    return {"predicted": "block" if res.status == RailStatus.BLOCKED else "allow", "rail": res.rail or ""}


async def run_cases(rails, cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        r = await (check_input(rails, c["text"]) if c["direction"] == "input" else check_output(rails, c))
        out.append({**c, **r})
    return out


def confusion(results: list[dict], direction: str, labels: list[str], say=print) -> int:
    rows = [r for r in results if r["direction"] == direction]
    counts = Counter((r["label"], r["predicted"]) for r in rows)
    say(f"\n{direction} ({len(rows)} cases): rows = label, columns = what the rails did")
    say(f"{'':>12}" + "".join(f"{p:>11}" for p in labels))
    for label in labels:
        say(f"{label:>12}" + "".join(f"{counts[(label, p)]:>11}" for p in labels))
    right = sum(counts[(l, l)] for l in labels)
    say(f"{right} of {len(rows)} right")
    return right


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    print(f"[INFO] model: {llm_calls.describe()} | cases: {CASES}")
    rails = build()
    results = asyncio.run(run_cases(rails, load_cases()))
    if a.verbose:
        for r in results:
            mark = "ok " if r["label"] == r["predicted"] else "XX "
            extra = f" by {r['rail']}" if r.get("rail") else ""
            print(f"{mark}{r['id']:<6} {r['label']:>9} -> {r['predicted']:<9}{extra:<28} {r['text'][:60]}")
    confusion(results, "input", ["allow", "block", "off_topic"])
    confusion(results, "output", ["allow", "block"])


if __name__ == "__main__":
    main()
