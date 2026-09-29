"""Step 3b: measure dense, keyword and hybrid search on the labelled questions.

    python m04/compare_retrieval.py                  # hybrid with RRF
    python m04/compare_retrieval.py --ranker weighted --weights 0.7 0.3
    python m04/compare_retrieval.py --verbose        # every question's top result per mode

m04/data/questions.jsonl holds 20 questions, each labelled with the manual section that
answers it. Half are paraphrases in everyday words ("headphones", "hang the screen on
the wall"), where dense search should do better; half name exact codes and model
numbers (E42, MST, VESA), where keyword search should do better.

hit@3: the question counts as answered when a chunk from the right product and section
is in the top 3. (Recall@k is the general form of the same idea: how many of the
relevant chunks are in the top k. Module 6 builds a full evaluation.)

It also prints:
  HNSW vs FLAT   FLAT compares every vector, so its top 3 is exact. How many of FLAT's
                 top-3 chunks does HNSW find (recall@3 against FLAT)?
  top 20 vs 3    how often the right chunk is in the hybrid top 20 but not in the top 3.
                 Those are the only questions a reranker could fix: it re-scores the 20
                 retrieved chunks and keeps the best 3. (Reranking is not run in this lab.)
"""
import argparse
import collections
import json
import pathlib
import time

import vector_store
from llm_calls import describe, embed
from retrieve import search

HERE = pathlib.Path(__file__).resolve().parent
QUESTIONS = HERE / "data" / "questions.jsonl"
MODES = ["dense", "keyword", "hybrid"]
FLAT_COPY = "manuals_flat"


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def is_hit(q: dict, hits: list[dict]) -> bool:
    return any(h["product"] == q["product"] and h["section"] == q["section"] for h in hits)


def evaluate(client, questions, qvecs, ranker="rrf", weights=(0.5, 0.5), k=3) -> dict:
    """hit@k per mode and question type, misses, and search time per query."""
    index = vector_store.dense_index(client)
    out = {}
    for mode in MODES:
        hits, misses, secs, tops = collections.Counter(), [], 0.0, {}
        for q, v in zip(questions, qvecs):
            t = time.perf_counter()
            res = search(client, q["question"], mode, k, ranker=ranker, weights=weights, qvec=v, index=index)
            secs += time.perf_counter() - t
            tops[q["id"]] = res[0]["id"] if res else "(none)"
            if is_hit(q, res):
                hits[q["type"]] += 1
            else:
                misses.append(q["id"])
        out[mode] = {"hits": sum(hits.values()), "by_type": dict(hits), "misses": misses,
                     "ms": secs * 1000 / len(questions), "top": tops}
    return out


def flat_copy(client) -> None:
    """Copy the chunks into a second collection with a FLAT (exact) index."""
    rows = client.query(vector_store.COLLECTION, filter='id != ""', limit=10000,
                        output_fields=["id", "text", "product", "section", "source", "dense"])
    vector_store.create(client, FLAT_COPY, len(rows[0]["dense"]), "FLAT")
    client.insert(FLAT_COPY, [{k: r[k] for k in ("id", "text", "product", "section", "source", "dense")}
                              for r in rows])
    client.load_collection(FLAT_COPY)


def hnsw_vs_flat(client, questions, qvecs, k=3) -> dict:
    flat_copy(client)
    index = vector_store.dense_index(client)
    found, times = 0, {"index": 0.0, "FLAT": 0.0}
    for q, v in zip(questions, qvecs):
        t = time.perf_counter()
        approx = {h["id"] for h in search(client, q["question"], "dense", k, qvec=v, index=index)}
        times["index"] += time.perf_counter() - t
        t = time.perf_counter()
        exact = {h["id"] for h in search(client, q["question"], "dense", k, qvec=v,
                                         collection=FLAT_COPY, index="FLAT")}
        times["FLAT"] += time.perf_counter() - t
        found += len(approx & exact)
    n = len(questions)
    return {"index": index, "recall": found / (k * n), "ms": {i: s * 1000 / n for i, s in times.items()}}


def rerank_room(client, questions, qvecs, ranker, weights) -> dict:
    index = vector_store.dense_index(client)
    top20 = sum(is_hit(q, search(client, q["question"], "hybrid", 20, ranker=ranker, weights=weights,
                                 qvec=v, index=index)) for q, v in zip(questions, qvecs))
    return {"top20": top20}


def run(ranker="rrf", weights=(0.5, 0.5), verbose=False, say=print) -> dict:
    questions = load_questions()
    types = collections.Counter(q["type"] for q in questions)
    client = vector_store.connect()
    vector_store.load(client)
    say(f"[INFO] {len(questions)} questions ({types['paraphrase']} paraphrase, {types['exact']} exact codes) | "
        f"{vector_store.count(client)} chunks | {describe()}")
    t = time.perf_counter()
    qvecs = embed([q["question"] for q in questions])
    embed_ms = (time.perf_counter() - t) * 1000 / len(questions)
    how = "RRF k=60" if ranker == "rrf" else f"weighted {weights[0]}/{weights[1]}"
    res = evaluate(client, questions, qvecs, ranker, weights)
    say(f"\nhit@3 (hybrid: {how})")
    say(f"{'mode':<9}{'all':>7}{'paraphrase':>12}{'exact':>8}{'ms/query':>10}")
    for mode in MODES:
        r = res[mode]
        say(f"{mode:<9}{r['hits']:>4}/{len(questions):<2}{r['by_type'].get('paraphrase', 0):>8}/{types['paraphrase']:<3}"
            f"{r['by_type'].get('exact', 0):>5}/{types['exact']:<2}{r['ms']:>8.1f}")
    say(f"(dense and hybrid also embed the question: {embed_ms:.0f} ms per question, not in ms/query)")
    for mode in MODES:
        say(f"[INFO] {mode} misses: {' '.join(res[mode]['misses']) or 'none'}")
    if verbose:
        for q in questions:
            say(f"{q['id']} {q['type']:<10} want {q['product']}/{q['section']}")
            say("     " + " | ".join(f"{m}: {res[m]['top'][q['id']]}" for m in MODES))

    hf = hnsw_vs_flat(client, questions, qvecs)
    say(f"\n[INFO] {hf['index']} vs FLAT (exact): {hf['index']} found {hf['recall']:.0%} of FLAT's top-3 chunks "
        f"({hf['ms']['index']:.1f} vs {hf['ms']['FLAT']:.1f} ms/query)")
    rr = rerank_room(client, questions, qvecs, ranker, weights)
    gap = rr["top20"] - res["hybrid"]["hits"]
    say(f"[INFO] hybrid: right chunk in the top 20 for {rr['top20']}/{len(questions)}, in the top 3 for "
        f"{res['hybrid']['hits']}/{len(questions)}")
    say(f"[INFO] a reranker that re-scores the 20 and keeps 3 could fix at most {gap} more (not run here)")
    client.drop_collection(FLAT_COPY)
    client.close()
    return {**res, "hnsw_vs_flat": hf, "top20": rr["top20"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranker", choices=["rrf", "weighted"], default="rrf")
    ap.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5], metavar=("DENSE", "SPARSE"))
    ap.add_argument("--verbose", action="store_true", help="print each question's top chunk per mode")
    a = ap.parse_args()
    run(a.ranker, tuple(a.weights), a.verbose)


if __name__ == "__main__":
    main()
