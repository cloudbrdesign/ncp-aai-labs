"""Step 4: score saved runs with Ragas and the local judge, without running the desk again.

    python m06/ragas_eval.py                        # m06/state/runs/A and B (whichever exist)
    python m06/ragas_eval.py m06/state/runs/A --all-reps
    M06_JUDGE_MODEL=nemotron-mini python m06/ragas_eval.py      # another judge (its own cache)

Input: the passage sidecar a `nat eval` run wrote (<run>/passages.jsonl: question, reply and
the passages the desk retrieved, per item and rep) and the references in data/testset.jsonl.
By default only rep 0 of each item is scored (every rep costs the same judge calls again).

Metrics (Ragas 0.4.3, the `ragas.metrics.collections` API), and judge calls per item:

    answer_accuracy          every item            reference vs reply, two prompts (0/2/4)   2 calls
    context_relevance        items with passages   question vs passages (0/1/2, twice)       2 calls
    response_groundedness    items with passages   reply vs passages (0/1/2, twice)          2 calls
    faithfulness             dev split only        reply's claims supported by passages      2+ calls
    context_recall           dev split only        reference's claims found in passages      1+ calls

The first three are NVIDIA's metrics in Ragas: few tokens per call. The last two break
text into claims first, so they cost more; they run on the dev split only.

The judge: llm_factory(model, provider="openai", client=AsyncOpenAI(base_url=Ollama /v1)).
Ragas sends Instructor's JSON mode (response_format json_object) with the schema in the
prompt. qwen3 gets reasoning_effort "none" (thinking off, see llm_calls.judge_args).

The cache: DiskCacheBackend stores every judge reply under a hash of the prompt and the
reply schema, so a second run of the same items makes no judge calls. The key does not
include the model name (checked in ragas 0.4.3, ragas/cache.py), so each judge model gets
its own cache folder here: m06/state/ragas_cache/<model>/. The cache only repeats a score;
it does not make it more correct, and a new run without it may score differently.

Ragas returns NaN when the judge's reply can't be parsed, and it does not clamp scores to
[0, 1]; both are counted. Writes m06/state/ragas/experiments/ragas_<run>.csv (one row per
item: Ragas' experiment format) and m06/state/ragas/ragas_<run>.json (averages, NaN counts,
judge calls, tokens, minutes).
"""
import argparse
import asyncio
import json
import math
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402  (first)
import testset  # noqa: E402

RUNS = testset.STATE / "runs"
OUT = testset.STATE / "ragas"
CACHE = testset.STATE / "ragas_cache"
NVIDIA = ["answer_accuracy", "context_relevance", "response_groundedness"]
DEV_ONLY = ["faithfulness", "context_recall"]
METRICS = NVIDIA + DEV_ONLY


def read_sidecar(run: pathlib.Path) -> list[dict]:
    path = run / "passages.jsonl"
    if not path.exists():
        raise SystemExit(f"[FAIL] {path} not found. Run nat eval first (m06/README.md, step 3).")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rows_for(run: pathlib.Path, all_reps: bool = False) -> list[dict]:
    items = {i["id"]: i for i in testset.load_all()}
    rows = []
    for rec in read_sidecar(run):
        item = items.get(rec.get("item_id"))
        if not item or (rec.get("rep", 0) != 0 and not all_reps):
            continue
        rows.append({"item_id": item["id"], "rep": rec.get("rep", 0), "split": item["split"],
                     "category": item["category"], "question": rec["question"], "reference": item["answer"],
                     "response": rec.get("reply", ""),
                     "contexts": json.dumps([p["text"] for p in rec.get("passages", [])], ensure_ascii=False)})
    return rows


class Counter:
    """Counts the judge's HTTP calls and tokens (httpx event hook on the OpenAI client)."""

    def __init__(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    async def on_response(self, response):
        if response.request.url.path.endswith("/chat/completions"):
            self.calls += 1
            try:
                await response.aread()
                usage = response.json().get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
            except Exception:
                pass


def judge_llm(model: str, counter: Counter, cache: bool = True, cache_root: pathlib.Path = CACHE):
    import httpx
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    http = httpx.AsyncClient(timeout=600, event_hooks={"response": [counter.on_response]})
    client = llm_calls.judge_client(async_client=True, http_client=http)
    backend = DiskCacheBackend(cache_dir=str(cache_root / re.sub(r"[^A-Za-z0-9_.-]", "_", model))) if cache else None
    return llm_factory(model, provider="openai", client=client, cache=backend,
                       temperature=0.0, max_tokens=1024, **llm_calls.judge_args(model))


def metric_objects(llm) -> dict:
    from ragas.metrics.collections import (AnswerAccuracy, ContextRecall, ContextRelevance, Faithfulness,
                                           ResponseGroundedness)
    return {"answer_accuracy": AnswerAccuracy(llm=llm), "context_relevance": ContextRelevance(llm=llm),
            "response_groundedness": ResponseGroundedness(llm=llm), "faithfulness": Faithfulness(llm=llm),
            "context_recall": ContextRecall(llm=llm)}


async def score_row(row: dict, metrics: dict, sem: asyncio.Semaphore) -> dict:
    contexts = json.loads(row["contexts"])
    out = {**row}
    wanted = {"answer_accuracy": bool(row["response"])}
    for m in ("context_relevance", "response_groundedness"):
        wanted[m] = bool(contexts) and bool(row["response"])
    for m in DEV_ONLY:
        wanted[m] = bool(contexts) and bool(row["response"]) and row["split"] == "dev"
    errors = []
    async with sem:
        for name in METRICS:
            if not wanted[name]:
                out[name] = ""                     # not applicable (no reply, no passages, or not dev)
                continue
            args = {"answer_accuracy": dict(user_input=row["question"], response=row["response"],
                                            reference=row["reference"]),
                    "context_relevance": dict(user_input=row["question"], retrieved_contexts=contexts),
                    "response_groundedness": dict(response=row["response"], retrieved_contexts=contexts),
                    "faithfulness": dict(user_input=row["question"], response=row["response"],
                                         retrieved_contexts=contexts),
                    "context_recall": dict(user_input=row["question"], retrieved_contexts=contexts,
                                           reference=row["reference"])}[name]
            try:
                value = (await metrics[name].ascore(**args)).value
                out[name] = float("nan") if value is None else float(value)
            except Exception as e:                 # the judge's reply didn't fit the schema, even after retries
                out[name] = float("nan")
                errors.append(f"{name}: {type(e).__name__}")
    out["errors"] = "; ".join(errors)
    return out


def run(run_dir: pathlib.Path, model: str | None = None, all_reps: bool = False, cache: bool = True,
        concurrency: int = 2, cache_root: pathlib.Path = CACHE, say=print) -> dict:
    from ragas import Dataset, experiment
    model = model or llm_calls.judge_model()
    rows = rows_for(run_dir, all_reps)
    if not rows:
        raise SystemExit(f"[FAIL] no test-set items in {run_dir / 'passages.jsonl'}")
    counter = Counter()
    metrics = metric_objects(judge_llm(model, counter, cache, cache_root))
    sem = asyncio.Semaphore(concurrency)
    name = f"ragas_{run_dir.name}"
    dataset = Dataset(name=f"items_{run_dir.name}", backend="local/csv", root_dir=str(OUT))
    for r in rows:
        dataset.append(r)

    @experiment()
    async def score(row):
        return await score_row(row, metrics, sem)

    say(f"[INFO] {testset.rel(run_dir)}: {len(rows)} items ({'all reps' if all_reps else 'rep 0'}); judge {model} at "
        f"{llm_calls.judge_base_url()}; cache {'on' if cache else 'off'}")
    t = time.perf_counter()
    exp = asyncio.run(score.arun(dataset, name=name))
    minutes = (time.perf_counter() - t) / 60
    scored = [dict(r) for r in exp]
    summary = {"run": str(run_dir), "judge": model, "items": len(scored), "all_reps": all_reps, "cache": cache,
               "judge_calls": counter.calls, "prompt_tokens": counter.prompt_tokens,
               "completion_tokens": counter.completion_tokens, "minutes": round(minutes, 3), "metrics": {}}
    for m in METRICS:
        vals = [r[m] for r in scored if r.get(m) not in ("", None)]
        vals = [float(v) for v in vals]
        ok = [v for v in vals if not math.isnan(v)]
        summary["metrics"][m] = {"scored": len(vals), "nan": len(vals) - len(ok),
                                 "out_of_range": sum(1 for v in ok if not 0.0 <= v <= 1.0),
                                 "mean": round(sum(ok) / len(ok), 3) if ok else None}
    summary["per_item"] = {f"{r['item_id']}_rep{r['rep']}": {m: r.get(m) for m in METRICS} for r in scored}
    summary["csv"] = str(OUT / "experiments" / f"{name}.csv")
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1, default=str))
    show(summary, say)
    return summary


def show(s: dict, say=print) -> None:
    say(f"{'metric':<24}{'scored':>7}{'mean':>8}{'NaN':>5}{'>1 or <0':>10}")
    for m, v in s["metrics"].items():
        mean = f"{v['mean']:.3f}" if v["mean"] is not None else "-"
        say(f"{m:<24}{v['scored']:>7}{mean:>8}{v['nan']:>5}{v['out_of_range']:>10}")
    say(f"[INFO] judge calls {s['judge_calls']}, tokens {s['prompt_tokens']} in / {s['completion_tokens']} out, "
        f"{s['minutes']:.2f} min" + (" (0 calls: every reply came from the cache)" if s["cache"] and not s["judge_calls"] else ""))
    say(f"[INFO] wrote {testset.rel(pathlib.Path(s['csv']))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=pathlib.Path, help="nat eval output folders (default: runs/A and runs/B)")
    ap.add_argument("--judge", help=f"judge model (default M06_JUDGE_MODEL or {llm_calls.JUDGE_DEFAULT})")
    ap.add_argument("--all-reps", action="store_true", help="score every rep, not only rep 0")
    ap.add_argument("--no-cache", action="store_true", help="call the judge even for prompts it has answered")
    ap.add_argument("--concurrency", type=int, default=2, help="items scored at the same time (default 2)")
    a = ap.parse_args()
    runs = a.runs or [p for p in (RUNS / "A", RUNS / "B") if (p / "passages.jsonl").exists()]
    if not runs:
        raise SystemExit("[FAIL] no runs found in m06/state/runs. Run nat eval first (m06/README.md, step 3).")
    for r in runs:
        run(r, a.judge, a.all_reps, not a.no_cache, a.concurrency)


if __name__ == "__main__":
    main()
