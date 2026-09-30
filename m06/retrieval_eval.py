"""Step 2: retrieval on its own, no LLM: recall@k, precision@k and hit@k for three search modes.

    python m06/retrieval_eval.py                 # all items with reference sections
    python m06/retrieval_eval.py --split dev
    python m06/retrieval_eval.py --verbose       # each item's top 5 per mode

For every test-set item with reference sections (testset.py resolves them to chunk IDs),
it searches the M4 index with dense, keyword and hybrid search (m04/retrieve.search, no
product filter) and scores the top k for k = 1, 3, 5, 10:

    recall@k     reference chunks in the top k / all reference chunks
    precision@k  reference chunks in the top k / k       (how much of what we hand the model is on topic)
    hit@k        1 if any reference chunk is in the top k (M4's hit@3 was this at k = 3)

Recall can only grow with k; precision usually falls, because a question has one or two
right chunks and the rest of the top 10 is noise. The desk hands the model k = 3.

Ragas has the same two ideas as ID-based metrics that need no LLM: IDBasedContextRecall and
IDBasedContextPrecision compare retrieved IDs with reference IDs. They are computed on the
same IDs as a cross-check (they count each ID once, like this script).

Writes m06/state/retrieval.csv (mode, k, category: averages) and m06/state/retrieval_items.csv
(one row per item, mode and k), which triage.py reads.
"""
import argparse
import asyncio
import collections
import csv
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402,F401  (first)
import testset  # noqa: E402

import retrieve  # noqa: E402   m04/retrieve.py (testset.py put m04 on the path)

MODES = ["dense", "keyword", "hybrid"]
KS = [1, 3, 5, 10]
OUT = testset.STATE / "retrieval.csv"
ITEMS_OUT = testset.STATE / "retrieval_items.csv"


def scores(top: list[str], ref: list[str], k: int) -> dict:
    got = set(top[:k]) & set(ref)
    n = len(top[:k])
    return {"recall": len(got) / len(ref), "precision": len(got) / n if n else 0.0, "hit": 1.0 if got else 0.0}


def ragas_id_scores(top: list[str], ref: list[str]) -> tuple[float, float]:
    """Ragas' IDBasedContextRecall and IDBasedContextPrecision on the same IDs.

    Ragas 0.4.3 warns that importing them from ragas.metrics is deprecated and points to
    ragas.metrics.collections, but 0.4.3's collections module doesn't have them yet, so the
    old import is the only one that works; the warning is silenced here.
    """
    import warnings
    from ragas import SingleTurnSample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
    sample = SingleTurnSample(retrieved_context_ids=top, reference_context_ids=ref)
    recall = asyncio.run(IDBasedContextRecall().single_turn_ascore(sample))
    precision = asyncio.run(IDBasedContextPrecision().single_turn_ascore(sample))
    return float(recall), float(precision)


def planted_example() -> dict:
    """Two retrieved chunks, one of them right: recall 1.0, precision 0.5 both ways."""
    top, ref = ["D300-troubleshooting-1", "H200-pairing-1"], ["D300-troubleshooting-1"]
    own = scores(top, ref, len(top))
    recall, precision = ragas_id_scores(top, ref)
    return {"own": (own["recall"], own["precision"]), "ragas": (recall, precision)}


def run(split: str | None = None, verbose: bool = False, say=print) -> dict:
    items = [i for i in testset.load() if i.get("ref_sections") and (not split or i["split"] == split)]
    testset.ensure_index(say=say)
    rows, per_item, mismatches, ms = [], [], 0, collections.defaultdict(float)
    with testset.open_index() as client:
        refs = testset.resolve(items, testset.chunk_rows(client))
        qvecs = llm_calls.embed([i["question"] for i in items])
        index = retrieve.vector_store.dense_index(client)
        tops = {}
        for mode in MODES:
            for item, v in zip(items, qvecs):
                t = time.perf_counter()
                hits = retrieve.search(client, item["question"], mode, max(KS), qvec=v, index=index)
                ms[mode] += (time.perf_counter() - t) * 1000
                tops[(mode, item["id"])] = [h["id"] for h in hits]
    for mode in MODES:
        for item in items:
            top, ref = tops[(mode, item["id"])], refs[item["id"]]
            for k in KS:
                s = scores(top, ref, k)
                r_recall, r_precision = ragas_id_scores(top[:k], ref)
                if abs(r_recall - s["recall"]) > 1e-9 or abs(r_precision - s["precision"]) > 1e-9:
                    mismatches += 1
                per_item.append({"item": item["id"], "category": item["category"], "split": item["split"],
                                 "mode": mode, "k": k, **{m: round(v, 4) for m, v in s.items()},
                                 "ragas_recall": round(r_recall, 4), "ragas_precision": round(r_precision, 4),
                                 "top": " ".join(top[:k]), "reference": " ".join(ref)})
    groups = collections.defaultdict(list)
    for r in per_item:
        groups[(r["mode"], r["k"], "all")].append(r)
        groups[(r["mode"], r["k"], r["category"])].append(r)
    for (mode, k, cat), rs in groups.items():
        rows.append({"mode": mode, "k": k, "category": cat, "n": len(rs),
                     **{m: round(sum(r[m] for r in rs) / len(rs), 3) for m in ("recall", "precision", "hit")}})
    rows.sort(key=lambda r: (MODES.index(r["mode"]), r["k"], r["category"] != "all", r["category"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT, rows), (ITEMS_OUT, per_item)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    show(rows, items, ms, mismatches, say)
    if verbose:
        for item in items:
            say(f"{item['id']} {item['category']:<18} want {' '.join(refs[item['id']])}")
            for mode in MODES:
                say(f"     {mode:<8} {' '.join(tops[(mode, item['id'])][:5])}")
    return {"rows": rows, "items": per_item, "mismatches": mismatches, "n": len(items), "refs": refs}


def show(rows, items, ms, mismatches, say=print) -> None:
    cats = [c for c in testset.CATEGORIES if any(r["category"] == c for r in rows)]
    say(f"[INFO] {len(items)} items with reference sections | {llm_calls.describe()}")
    say(f"\n{'mode':<9}{'k':>3}{'recall':>8}{'precision':>11}{'hit':>6}   recall@k per category")
    say(f"{'':<37}" + "".join(f"{c[:12]:>13}" for c in cats))
    for r in rows:
        if r["category"] != "all":
            continue
        per = {x["category"]: x["recall"] for x in rows if x["mode"] == r["mode"] and x["k"] == r["k"]}
        say(f"{r['mode']:<9}{r['k']:>3}{r['recall']:>8.2f}{r['precision']:>11.2f}{r['hit']:>6.2f}   "
            + "".join(f"{per.get(c, float('nan')):>13.2f}" for c in cats))
    say("\n" + ", ".join(f"{m} {ms[m] / max(len(items), 1):.1f} ms/query" for m in MODES)
        + " (dense and hybrid: the question embedding is not included)")
    say(f"[INFO] Ragas IDBasedContextRecall/Precision differ from this script's numbers on {mismatches} "
        f"item x mode x k rows")
    say(f"[INFO] wrote {testset.rel(OUT)} and {testset.rel(ITEMS_OUT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=testset.SPLITS, help="only this split (default: both)")
    ap.add_argument("--verbose", action="store_true", help="each item's top 5 chunks per mode")
    a = ap.parse_args()
    run(a.split, a.verbose)


if __name__ == "__main__":
    main()
