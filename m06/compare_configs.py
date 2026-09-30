"""Step 6: configuration A vs B on the same items: per category, paired flips, McNemar, Pareto.

    python m06/compare_configs.py                                   # m06/state/runs/A vs m06/state/runs/B
    python m06/compare_configs.py m06/state/runs/A m06/state/runs/B --show-flips

Both folders are `nat eval` outputs of the same test set (step 3); B changes one thing (by
default the desk's model, llama3.2:1b instead of llama3.2:3b). Per item and rep it reads:

    pass            keywords_all and refuses_when_unanswerable both 1 (evaluators.item_passes)
    answer_accuracy NAT's `ragas` evaluator (the judge, reference vs reply)
    groundedness    ragas_eval.py's response_groundedness (rep 0), if step 4 was run
    latency         the passage sidecar (wall-clock per turn)
    LLM calls, tokens   the profiler's traces (LLM_END events per request)

and prints, per category and overall: the pass rate (an item passes when it passed in more
than half of its reps; 1 of 2 is not a pass), Answer Accuracy, groundedness, p50/p95
latency, LLM calls and tokens per item, and the judge calls each run cost.

Then, on the items both runs have:
  rep-to-rep spread  items whose verdict changed between reps of the same configuration
  paired flips       regressions (A passed, B failed) and improvements (A failed, B passed)
  McNemar exact      only the discordant pairs carry information; under "no difference" a
                     flip is equally likely to go either way, so the p-value is a two-sided
                     binomial test of the improvements out of all flips (scipy.stats.binomtest)
  Pareto             pass rate vs p50 latency: is either configuration better on both?

The verdict line says INCONCLUSIVE unless the flips are lopsided enough (p < 0.05), and even
then it is one small test set: confirm on the held-out split before switching. With about
40 items few items flip, and small samples only detect large differences.

It also writes NeMo Evaluator's format for each run, m06/state/nel/<run>/results.jsonl (one
record per item and rep: problem_idx, repeat, reward 1.0/0.0, metadata.category, ...) and
eval-desk.json (benchmark name and mean_reward), so `nel compare` can run on the same data
in the optional Python 3.12 venv (README). NEL averages the repeats first, so it may pick a
sign or permutation test instead of McNemar.
"""
import argparse
import collections
import csv
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluators  # noqa: E402   (item_passes; importing it loads NAT, which the lab venv has)
import testset  # noqa: E402

RUNS = testset.STATE / "runs"
NEL = testset.STATE / "nel"
RAGAS = testset.STATE / "ragas"
REP = re.compile(r"^(.*)_rep(\d+)$")


def split_id(item_id: str) -> tuple[str, int]:
    m = REP.match(str(item_id))
    return (m.group(1), int(m.group(2))) if m else (str(item_id), 0)


def scores(run: pathlib.Path, name: str) -> dict:
    path = run / f"{name}_output.json"
    if not path.exists():
        return {}
    return {split_id(i["id"]): i for i in json.loads(path.read_text()).get("eval_output_items", [])}


def traces(run: pathlib.Path, questions: dict) -> dict:
    """LLM calls, tokens, LLM seconds and workflow seconds per (item, rep), from the profiler's traces."""
    path = run / "all_requests_profiler_traces.json"
    out, seen = {}, collections.Counter()
    if not path.exists():
        return out
    for req in sorted(json.loads(path.read_text()), key=lambda r: r["request_number"]):
        steps = [s["payload"] for s in req["intermediate_steps"]]
        start = next((s for s in steps if s["event_type"] == "WORKFLOW_START"), None)
        end = next((s for s in steps if s["event_type"] == "WORKFLOW_END"), None)
        item = questions.get(testset.norm((start or {}).get("data", {}).get("input", "")))
        if not item:
            continue
        key = (item["id"], seen[item["id"]])
        seen[item["id"]] += 1
        llm = [s for s in steps if s["event_type"] == "LLM_END"]
        tokens = sum(((s.get("usage_info") or {}).get("token_usage") or {}).get("total_tokens", 0) for s in llm)
        llm_s = sum(s["event_timestamp"] - s["span_event_timestamp"] for s in llm if s.get("span_event_timestamp"))
        out[key] = {"llm_calls": len(llm), "tokens": tokens, "llm_seconds": llm_s,
                    "workflow_seconds": (end["event_timestamp"] - start["event_timestamp"]) if start and end else None,
                    "llm_names": collections.Counter(s.get("name") for s in llm)}
    return out


def sidecar(run: pathlib.Path) -> dict:
    path = run / "passages.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("item_id") is not None:
                out[(r["item_id"], r.get("rep", 0))] = r
    return out


def ragas_scores(run: pathlib.Path) -> dict:
    path = RAGAS / f"ragas_{run.name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_run(run: pathlib.Path) -> dict:
    """Everything step 3 (and step 4, if run) recorded for one configuration, per (item, rep)."""
    if not (run / "workflow_output.json").exists():
        raise SystemExit(f"[FAIL] {run}/workflow_output.json not found. Run nat eval first (README, step 3).")
    items = {i["id"]: i for i in testset.load_all()}
    questions = testset.by_question(list(items.values()))
    out = json.loads((run / "workflow_output.json").read_text())
    kw, ref, aa = scores(run, "keywords_all"), scores(run, "refuses_when_unanswerable"), scores(run, "answer_accuracy")
    cites, inv = scores(run, "cites_right_manual"), scores(run, "no_invented_order")
    tr, side, rg = traces(run, questions), sidecar(run), ragas_scores(run)
    per_item = rg.get("per_item", {})
    rows = {}
    for w in out:
        key = split_id(w["id"])
        item = items.get(key[0], {})
        k, r = kw.get(key, {}).get("score"), ref.get(key, {}).get("score")
        s = side.get(key, {})
        rg_item = per_item.get(f"{key[0]}_rep{key[1]}", {})
        rows[key] = {"item": key[0], "rep": key[1], "category": item.get("category", w.get("category", "?")),
                     "split": item.get("split", w.get("split", "?")), "question": w.get("question", ""),
                     "answer": w.get("answer", ""), "reply": w.get("generated_answer") or "",
                     "keywords": k, "refusal": r, "pass": evaluators.item_passes(k, r),
                     "answer_accuracy": aa.get(key, {}).get("score"),
                     "aa_error": (aa.get(key, {}).get("error") or ""),
                     "cites_right_manual": cites.get(key, {}).get("score"),
                     "no_invented_order": inv.get(key, {}).get("score"),
                     "groundedness": _num(rg_item.get("response_groundedness")),
                     "latency": s.get("latency_s", (tr.get(key) or {}).get("workflow_seconds")),
                     "route": s.get("route", []), "passages": [p["id"] for p in s.get("passages", [])],
                     "error": s.get("error", ""), **{m: (tr.get(key) or {}).get(m) for m in
                                                    ("llm_calls", "tokens", "llm_seconds", "workflow_seconds")}}
    return {"run": run, "rows": rows, "ragas": rg, "model": next((s.get("model") for s in side.values()), "?")}


def _fmt(v, f: str) -> str:
    return "-" if v is None else format(v, f)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def majority(rows: dict) -> dict[str, bool]:
    """item -> passed in more than half of its reps."""
    by_item = collections.defaultdict(list)
    for (item, _), r in rows.items():
        by_item[item].append(r["pass"])
    return {item: sum(v) * 2 > len(v) for item, v in by_item.items()}


def spread(rows: dict) -> list[str]:
    by_item = collections.defaultdict(set)
    for (item, _), r in rows.items():
        by_item[item].add(r["pass"])
    return sorted(i for i, v in by_item.items() if len(v) > 1)


def pct(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summary(run: dict, items: set[str], category: str | None = None) -> dict:
    rows = [r for (i, _), r in run["rows"].items() if i in items and (category is None or r["category"] == category)]
    maj = majority({(r["item"], r["rep"]): r for r in rows})
    n_items = len(maj)
    per_item = collections.defaultdict(list)
    for r in rows:
        per_item[r["item"]].append(r)
    return {"items": n_items, "pass_rate": sum(maj.values()) / n_items if n_items else None,
            "answer_accuracy": mean(r["answer_accuracy"] for r in rows),
            "groundedness": mean(r["groundedness"] for r in rows),
            "p50": pct([r["latency"] for r in rows], 0.5), "p95": pct([r["latency"] for r in rows], 0.95),
            "calls": mean(r["llm_calls"] for r in rows), "tokens": mean(r["tokens"] for r in rows)}


def mcnemar(regressions: int, improvements: int) -> float | None:
    from scipy.stats import binomtest
    n = regressions + improvements
    return None if n == 0 else float(binomtest(improvements, n, 0.5).pvalue)


def verdict(regressions: int, improvements: int, p: float | None, n: int) -> str:
    if p is None:
        return (f"INCONCLUSIVE: no item changed its verdict between A and B on these {n} items. Equal pass "
                "rates here do not show that the two configurations are equally good.")
    flips = f"{regressions} regression{'s' if regressions != 1 else ''}, {improvements} improvement{'s' if improvements != 1 else ''}"
    if p >= 0.05:
        return (f"INCONCLUSIVE: {flips}, exact McNemar p = {p:.3f}. {regressions + improvements} discordant "
                f"pairs cannot tell the two apart; more items (or more flips) would be needed.")
    side = "B lost" if regressions > improvements else "B gained"
    return (f"{side} more items than it gained or lost the other way ({flips}, exact McNemar p = {p:.3g}). "
            f"One {n}-item test set: confirm on the held-out split before deciding.")


def pareto(a: dict, b: dict, names=("A", "B")) -> str:
    pa, pb = (a["pass_rate"] or 0, a["p50"] or 0), (b["pass_rate"] or 0, b["p50"] or 0)

    def dominates(x, y):   # higher pass rate and lower latency, at least one strictly
        return x[0] >= y[0] and x[1] <= y[1] and (x[0] > y[0] or x[1] < y[1])
    if dominates(pa, pb):
        return f"{names[0]} dominates {names[1]} (at least as accurate and at least as fast): only {names[0]} is on the Pareto front"
    if dominates(pb, pa):
        return f"{names[1]} dominates {names[0]} (at least as accurate and at least as fast): only {names[1]} is on the Pareto front"
    return "both are on the Pareto front: one is more accurate, the other faster; the choice depends on the priority"


# ---- NeMo Evaluator's format --------------------------------------------------------------

def write_nel(run: dict, order: list[str]) -> pathlib.Path:
    folder = NEL / run["run"].name
    folder.mkdir(parents=True, exist_ok=True)
    idx = {item: i for i, item in enumerate(order)}
    records = []
    for (item, rep), r in sorted(run["rows"].items(), key=lambda kv: (idx.get(kv[0][0], 10**6), kv[0][1])):
        if item not in idx:
            continue
        records.append({"problem_idx": idx[item], "repeat": rep, "reward": 1.0 if r["pass"] else 0.0,
                        "model_response": r["reply"], "expected_answer": r["answer"],
                        "metadata": {"category": r["category"], "item_id": item, "split": r["split"]},
                        "scoring_details": {"keywords_all": r["keywords"], "refuses_when_unanswerable": r["refusal"]}})
    with (folder / "results.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cats = collections.defaultdict(list)
    for rec in records:
        cats[rec["metadata"]["category"]].append(rec["reward"])
    rewards = [rec["reward"] for rec in records]
    bundle = {"run_id": f"desk-{run['run'].name}", "config": {"model": run["model"], "source": str(run["run"])},
              "benchmark": {"name": "desk", "samples": len({rec['problem_idx'] for rec in records}),
                            "repeats": 1 + max((rec["repeat"] for rec in records), default=0),
                            "scores": {"mean_reward": {"value": round(sum(rewards) / len(rewards), 4) if rewards else 0.0}},
                            "categories": {c: {"mean_reward": round(sum(v) / len(v), 4)} for c, v in cats.items()}}}
    (folder / "eval-desk.json").write_text(json.dumps(bundle, indent=1))
    return folder


def validate_nel(folder: pathlib.Path) -> list[str]:
    """The fields NeMo Evaluator's `nel compare` reads (checked against nemo-evaluator 0.3.0's comparison.py)."""
    problems = []
    try:
        bundle = json.loads((folder / "eval-desk.json").read_text())
        if not isinstance(bundle["benchmark"]["name"], str):
            problems.append("benchmark.name is not a string")
        if not isinstance(bundle["benchmark"]["scores"]["mean_reward"]["value"], (int, float)):
            problems.append("benchmark.scores.mean_reward.value is not a number")
    except (OSError, KeyError, TypeError, ValueError) as e:
        problems.append(f"eval-desk.json: {type(e).__name__}: {e}")
    try:
        lines = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as e:
        return problems + [f"results.jsonl: {type(e).__name__}: {e}"]
    if not lines:
        problems.append("results.jsonl is empty")
    for rec in lines:
        if not isinstance(rec.get("problem_idx"), int) or not isinstance(rec.get("repeat"), int):
            problems.append(f"problem_idx/repeat not integers: {rec.get('problem_idx')}, {rec.get('repeat')}")
        if not isinstance(rec.get("reward"), (int, float)):
            problems.append(f"reward missing for problem {rec.get('problem_idx')}")
        if not isinstance((rec.get("metadata") or {}).get("category"), str):
            problems.append(f"metadata.category missing for problem {rec.get('problem_idx')}")
    return problems[:5]


# ---- the comparison -------------------------------------------------------------------------

def compare(a_dir: pathlib.Path, b_dir: pathlib.Path, show_flips: bool = False, say=print) -> dict:
    a, b = load_run(a_dir), load_run(b_dir)
    names = (a_dir.name, b_dir.name)
    common = {i for i, _ in a["rows"]} & {i for i, _ in b["rows"]}
    if not common:
        raise SystemExit("[FAIL] the two runs have no items in common")
    n_reps = (1 + max(r for _, r in a["rows"]), 1 + max(r for _, r in b["rows"]))
    say(f"[INFO] {names[0]}: {a['model']}, {n_reps[0]} reps | {names[1]}: {b['model']}, {n_reps[1]} reps | "
        f"{len(common)} items in both")
    cats = [c for c in testset.CATEGORIES if any(r["category"] == c for (i, _), r in a["rows"].items() if i in common)]
    cols = [("pass rate", "pass_rate", ".2f"), ("answer acc.", "answer_accuracy", ".2f"),
            ("grounded", "groundedness", ".2f"), ("p50 s", "p50", ".2f"), ("p95 s", "p95", ".2f"),
            ("LLM calls", "calls", ".1f"), ("tokens", "tokens", ".0f")]
    say("\n" + f"{'':<22}" + "".join(f"{label:>14}" for label, _, _ in cols))
    say(f"{'category':<18}{'n':>4}" + f"{names[0]:>7}{names[1]:>7}" * len(cols))
    table = {}
    for cat in cats + [None]:
        sa, sb = summary(a, common, cat), summary(b, common, cat)
        table[cat or "all"] = {"A": sa, "B": sb}
        cells = "".join(f"{_fmt(sa[key], f):>7}{_fmt(sb[key], f):>7}" for _, key, f in cols)
        say(f"{(cat or 'ALL'):<18}{sa['items']:>4}{cells}")
    judge = {}
    for name, run in zip(names, (a, b)):
        nat_items = sum(1 for (i, _), r in run["rows"].items() if i in common)
        judge[name] = {"nat_answer_accuracy_calls_min": 2 * nat_items,
                       "ragas_eval_calls": run["ragas"].get("judge_calls"),
                       "ragas_eval_minutes": run["ragas"].get("minutes")}
    say("\nJudge calls: " + "; ".join(
        f"{n}: NAT Answer Accuracy at least {j['nat_answer_accuracy_calls_min']} (2 per item and rep), "
        f"ragas_eval {j['ragas_eval_calls'] if j['ragas_eval_calls'] is not None else 'not run'}"
        + (f" ({j['ragas_eval_minutes']} min)" if j["ragas_eval_minutes"] is not None else "")
        for n, j in judge.items()))

    ma, mb = majority({k: v for k, v in a["rows"].items() if k[0] in common}), \
        majority({k: v for k, v in b["rows"].items() if k[0] in common})
    sp = {names[0]: spread({k: v for k, v in a["rows"].items() if k[0] in common}),
          names[1]: spread({k: v for k, v in b["rows"].items() if k[0] in common})}
    say("Rep-to-rep spread (items whose verdict changed between reps): "
        + "; ".join(f"{n}: {len(v)}" + (f" ({', '.join(v)})" if v else "") for n, v in sp.items()))
    regressions = sorted(i for i in common if ma[i] and not mb[i])
    improvements = sorted(i for i in common if not ma[i] and mb[i])
    p = mcnemar(len(regressions), len(improvements))
    both_pass = sum(1 for i in common if ma[i] and mb[i])
    both_fail = sum(1 for i in common if not ma[i] and not mb[i])
    say(f"\nPaired flips ({names[0]} -> {names[1]}): {len(regressions)} regressions, {len(improvements)} improvements, "
        f"{both_pass} pass in both, {both_fail} fail in both")
    if show_flips or regressions or improvements:
        for label, ids in (("regression", regressions), ("improvement", improvements)):
            for i in ids:
                cat = next(r["category"] for (x, _), r in a["rows"].items() if x == i)
                say(f"  {label:<12} {i:<5} {cat}")
    say(f"McNemar exact (binomial on the discordant pairs): p = {p:.4f}" if p is not None
        else "McNemar exact: no discordant pairs, no test")
    par = pareto(table["all"]["A"], table["all"]["B"], names)
    say(f"Pareto (pass rate vs p50 latency): {par}")
    v = verdict(len(regressions), len(improvements), p, len(common))
    say(f"VERDICT: {v}")
    folders = [write_nel(run, [i["id"] for i in testset.load_all()]) for run in (a, b)]
    problems = {str(f): validate_nel(f) for f in folders}
    say(f"[INFO] NeMo Evaluator format: {', '.join(str(testset.rel(f)) for f in folders)} "
        + ("(required fields present)" if not any(problems.values()) else f"PROBLEMS: {problems}"))
    say(f"[INFO] optional: nel compare {testset.rel(folders[0])} {testset.rel(folders[1])} "
        "--correct-above 0.5 --show-flips --no-strict")
    result = {"a": str(a_dir), "b": str(b_dir), "items": len(common), "table": table, "judge": judge,
              "spread": sp, "regressions": regressions, "improvements": improvements, "p_value": p,
              "pareto": par, "verdict": v, "nel": [str(f) for f in folders], "nel_problems": problems}
    (testset.STATE / "compare.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=pathlib.Path, default=[RUNS / "A", RUNS / "B"],
                    help="two nat eval output folders (default m06/state/runs/A and B)")
    ap.add_argument("--show-flips", action="store_true", help="list every flipped item")
    a = ap.parse_args()
    if len(a.runs) != 2:
        ap.error("give two output folders (or none for runs/A vs runs/B)")
    compare(*a.runs, show_flips=a.show_flips)


if __name__ == "__main__":
    main()
