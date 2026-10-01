"""Step 8: every failed item in one bucket, and what to fix first.

    python m06/triage.py                          # configuration A (m06/state/runs/A)
    python m06/triage.py m06/state/runs/B

It joins what steps 2 to 6 wrote for one run: the evaluator scores and the profiler traces
(step 3), the passage sidecar (what the desk retrieved and how it routed), Ragas
groundedness (step 4, if run) and step 2's retrieval results. An item fails when it failed
in more than half of its reps (compare_configs.majority). Each failed item goes into the
first bucket that fits, in this order:

    infra           the turn errored or the reply is empty (timeout, model not reachable)
    routing         the plan used the wrong source: an order question without SQL, a manual
                    or mixed question without a manual search
    wrong_refusal   it declined an answerable question, or answered one it should decline
    retrieval_miss  the item has reference chunks and none of them was among the passages
                    the desk retrieved
    citation        it cited a chunk from the wrong manual, or cited when it shouldn't
    judge_disagrees the deterministic checks failed it, but the judge's Answer Accuracy says
                    the reply matches the reference (>= 0.5): read it; the keywords may be too strict
    not_grounded    the facts were available (retrieved, or from SQL) but the reply doesn't
                    carry them: a generation problem

The order matters: a routing error also shows up as a retrieval miss, and fixing the
earlier cause usually fixes the later symptom. The report prints bucket x category counts,
two examples per bucket, where the time goes (LLM calls vs the rest, from the profiler's
traces) and "fix first": the buckets by size.

Step 2 searches without the desk's product filter, so its misses (hybrid, k = 3) are
printed next to the desk's own retrieval misses for comparison; the bucket itself is
decided by what the desk actually retrieved.
"""
import argparse
import collections
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_configs  # noqa: E402
import testset  # noqa: E402

BUCKETS = ["infra", "routing", "wrong_refusal", "retrieval_miss", "citation", "judge_disagrees", "not_grounded"]
MANUAL = {"manual_exact", "manual_paraphrase", "mixed"}
OUT = testset.STATE / "triage.json"


def bucket(r: dict, ref_ids: list[str]) -> tuple[str, str]:
    """(bucket, why) for one failed item-rep."""
    route = " ".join(r["route"])
    if r["error"] or not r["reply"].strip():
        return "infra", r["error"] or "empty reply"
    if r["category"] == "order" and "(sql)" not in route:
        return "routing", f"order question, route: {route or 'none'}"
    if r["category"] in MANUAL and "(rag)" not in route:
        return "routing", f"{r['category']} question without a manual search, route: {route or 'none'}"
    if r["refusal"] == 0.0:
        return "wrong_refusal", ("declined an answerable question" if r["category"] not in testset.REFUSE
                                 else "answered instead of declining")
    if ref_ids and not set(ref_ids) & set(r["passages"]):
        return "retrieval_miss", f"wanted {' '.join(ref_ids)}, got {' '.join(r['passages']) or 'nothing'}"
    if r["cites_right_manual"] == 0.0:
        return "citation", "cites the wrong manual (or cites when it shouldn't)"
    if (r["answer_accuracy"] or 0) >= 0.5:
        return "judge_disagrees", f"keywords failed, Answer Accuracy {r['answer_accuracy']:.2f}"
    g = f", groundedness {r['groundedness']:.2f}" if r["groundedness"] is not None else ""
    return "not_grounded", f"the facts were available but the reply misses them{g}"


def step2_misses(k: int = 3, mode: str = "hybrid") -> set[str] | None:
    path = testset.STATE / "retrieval_items.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return {r["item"] for r in csv.DictReader(f) if r["mode"] == mode and int(r["k"]) == k and float(r["hit"]) == 0}


def triage(run_dir: pathlib.Path, say=print) -> dict:
    run = compare_configs.load_run(run_dir)
    rows = run["rows"]
    items = {i["id"]: i for i in testset.load_all()}
    refs = testset.reference_ids([items[i] for i in {k[0] for k in rows} if i in items])
    maj = compare_configs.majority(rows)
    failed = sorted(i for i, ok in maj.items() if not ok)
    assigned = {}
    for item in failed:
        reps = sorted((r for (i, _), r in rows.items() if i == item), key=lambda r: r["rep"])
        first_fail = next(r for r in reps if not r["pass"])
        b, why = bucket(first_fail, refs.get(item, []))
        assigned[item] = {"bucket": b, "why": why, "category": first_fail["category"], "rep": first_fail["rep"],
                          "question": first_fail["question"], "reply": first_fail["reply"]}
    cats = [c for c in testset.CATEGORIES if any(a["category"] == c for a in assigned.values())]
    counts = collections.Counter((a["bucket"], a["category"]) for a in assigned.values())
    totals = collections.Counter(a["bucket"] for a in assigned.values())
    say(f"[INFO] {testset.rel(run_dir)}: {run['model']}; {len(maj)} items, {len(failed)} failed (majority over reps)")
    say(f"\n{'bucket':<17}" + "".join(f"{c[:11]:>12}" for c in cats) + f"{'total':>8}")
    for b in BUCKETS:
        if totals[b]:
            say(f"{b:<17}" + "".join(f"{counts[(b, c)] or '.':>12}" for c in cats) + f"{totals[b]:>8}")
    for b in BUCKETS:
        examples = [(i, a) for i, a in assigned.items() if a["bucket"] == b][:2]
        if examples:
            say(f"\n{b}:")
            for i, a in examples:
                say(f"  {i} ({a['category']}) {a['question']}")
                say(f"      reply: {a['reply'][:110]!r}")
                say(f"      why:   {a['why']}")
    llm_s = [r["llm_seconds"] for r in rows.values() if r["llm_seconds"] is not None]
    wf_s = [r["workflow_seconds"] for r in rows.values() if r["workflow_seconds"] is not None]
    calls = [r["llm_calls"] for r in rows.values() if r["llm_calls"] is not None]
    if wf_s and llm_s:
        llm_m, wf_m = sum(llm_s) / len(llm_s), sum(wf_s) / len(wf_s)
        say(f"\nWhere the time goes (profiler traces, mean per item): {wf_m:.2f} s per turn, of which "
            f"{llm_m:.2f} s in {sum(calls) / len(calls):.1f} LLM calls ({100 * llm_m / wf_m:.0f}%) and "
            f"{wf_m - llm_m:.2f} s in everything else (retrieval, SQL, Python)")
        slowest = "LLM calls" if llm_m >= wf_m - llm_m else "the non-LLM steps (retrieval, SQL, Python)"
        say(f"Slowest step: {slowest}. NAT sees the wrapped graph as one function, so it cannot name the M4 node.")
    order = [b for b, _ in totals.most_common()]
    say(f"\nFix first: {', '.join(f'{b} ({totals[b]})' for b in order) or 'nothing failed'}")
    desk_miss = {i for i, a in assigned.items() if a["bucket"] == "retrieval_miss"}
    s2 = step2_misses()
    if s2 is not None:
        say(f"[INFO] reference chunks missed by the desk's own search (filtered by product): "
            f"{', '.join(sorted(desk_miss)) or 'none'}; by step 2's hybrid search (k=3, no filter): "
            f"{', '.join(sorted(s2 & set(maj))) or 'none'}")
    result = {"run": str(run_dir), "items": len(maj), "failed": failed, "assigned": assigned,
              "totals": dict(totals), "fix_first": order, "retrieval_miss": sorted(desk_miss),
              "step2_misses": sorted(s2) if s2 is not None else None}
    OUT.write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", type=pathlib.Path, default=compare_configs.RUNS / "A",
                    help="a nat eval output folder (default m06/state/runs/A)")
    a = ap.parse_args()
    triage(a.run)


if __name__ == "__main__":
    main()
