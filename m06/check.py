"""Check that the M06 lab works end to end (free mode: everything local on Ollama).

    python m06/check.py        # Ollama running with llama3.2:3b, llama3.2:1b, qwen3:4b, embeddinggemma

1.  llama3.2:3b, llama3.2:1b, the judge model and embeddinggemma answer on Ollama.
2.  The test set loads: about 40 items, every category has at least 3, both splits, every
    reference section resolves to chunks in the index (6.1).
3.  retrieval_eval writes rows for 3 modes x 4 k values; recall@k never falls as k grows; Ragas'
    ID-based metrics equal the script's precision/recall on a planted example and on every row.
4.  `nat validate` passes on configs/eval_A.yml; keywords_all and refuses_when_unanswerable are
    registered evaluators (`nat info components -t evaluator`).
5.  `nat eval --reps 2` for A (dev split) writes items x reps outputs (IDs end _rep0, _rep1), the
    profiler CSV with LLM_END rows and every evaluator's output file.
6.  The passage sidecar has one record per item and rep, with passage IDs that exist in the index.
7.  NAT's `ragas` Answer Accuracy scored every item-rep, all in [0, 1], no evaluator errors.
8.  ragas_eval scores the three NVIDIA metrics for every item they apply to, all in [0, 1]; a second
    run is served from the cache (same scores, 0 judge calls, faster) (6.1, 6.4).
9.  `nat eval --skip_workflow` on the saved workflow_output.json re-scores without running the desk.
10. judge_check gets a parseable verdict for at least 90% of the labelled replies for each judge and
    prints agreement, false passes and false fails (6.5).
11. The pairwise test prints a position-consistency rate for each judge.
12. B's run exists (OLLAMA_MODEL=llama3.2:1b) and compare_configs prints the side-by-side table with
    latency, LLM calls and tokens (6.2, 6.4).
13. Paired flips and the McNemar p-value are computed; results.jsonl and eval-desk.json carry the
    fields NeMo Evaluator's `nel compare` reads.
14. With DESK_TEMPERATURE=0 DESK_SEED=42 the same direct model call twice returns the same text;
    A's rep-to-rep spread is reported (no assertion).
15. `feedback.py rate --answer down ...` appends a valid record; `report` counts it (6.3).
16. `promote` turns the seeded thumbs-down records with corrections into candidates and skips the
    planted near-duplicate.
17. `nat eval` on the promoted file scores the new items (the loop closes).
18. triage puts every failed item in exactly one bucket; its retrieval_miss items are the ones whose
    reference chunks the desk did not retrieve (step 2's unfiltered misses printed next to them) (6.5).

It rebuilds m04/state (clean manuals, manuals.db) and m04/data/orders.db, like m04/m05 check.py, and
writes m06/state/ (runs/A, runs/B, runs/feedback, ragas/, nel/, feedback.jsonl, ...). It uses the dev
split and 2 reps to keep the time down; the README's full run uses every item and 3 reps.
"""
import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402  (first; in fake mode this starts tests/fake_oai.py)
import compare_configs  # noqa: E402
import desk_app  # noqa: E402
import feedback  # noqa: E402
import judge_check  # noqa: E402
import ragas_eval  # noqa: E402
import retrieval_eval  # noqa: E402
import testset  # noqa: E402
import triage  # noqa: E402

results = []
quiet = lambda *a, **k: None  # noqa: E731
CFG = "m06/configs/eval_A.yml"
RUNS = testset.STATE / "runs"
DEV = ["--override", "eval.general.dataset.filter.allowlist.field.split", "dev"]
NAT_TIMEOUT = 3600
LOG = testset.STATE / "check.log"
EVALUATORS = ["keywords_all", "refuses_when_unanswerable", "cites_right_manual", "no_invented_order",
              "answer_accuracy", "llm_latency", "llm_calls", "tokens_per_call"]
DESK_MODELS = ["llama3.2:3b", "llama3.2:1b"]


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail else "") + "\n")


def info(text):
    print(f"       [INFO] {text}", flush=True)
    with LOG.open("a") as f:
        f.write(f"       [INFO] {text}\n")


def nat(*args, env=None) -> subprocess.CompletedProcess:
    """Run the nat CLI from the repo root (the config uses paths like m06/data/testset.jsonl)."""
    desk_app.close()   # let the nat process open the Milvus Lite file
    exe = shutil.which("nat", path=str(pathlib.Path(sys.executable).parent)) or "nat"
    return subprocess.run([exe, *args], cwd=LABS, capture_output=True, text=True, timeout=NAT_TIMEOUT,
                          env={**os.environ, "COLUMNS": "250", **(env or {})})


def tail(p: subprocess.CompletedProcess, n=4) -> list[str]:
    return (p.stdout + p.stderr).strip().splitlines()[-n:]


def outputs(folder: pathlib.Path, name: str) -> list[dict]:
    path = folder / f"{name}_output.json"
    return json.loads(path.read_text()).get("eval_output_items", []) if path.exists() else []


def models_answer() -> bool:
    client = llm_calls.judge_client()
    replies = {}
    try:
        judge = llm_calls.judge_model()
        for model in DESK_MODELS:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content":
                                               "Reply with the single word OK."}], temperature=0, max_tokens=32)
            replies[model] = (r.choices[0].message.content or "").strip()
        # The judge is always asked for schema-constrained JSON (as in judge_check and ragas_eval),
        # so test it the same way: {"word": "OK"} within 64 tokens, no thinking out loud first.
        r = client.chat.completions.create(model=judge, messages=[{"role": "user", "content":
                                           'Reply with JSON: {"word": "OK"}'}], temperature=0, max_tokens=64,
                                           response_format=llm_calls.json_schema_format(
                                               "word", {"word": {"type": "string"}}, ["word"]),
                                           **llm_calls.judge_args(judge))
        try:
            replies[judge] = str(json.loads(r.choices[0].message.content or "{}").get("word", ""))
        except ValueError:
            replies[judge] = (r.choices[0].message.content or "").strip()
        dim = len(llm_calls.embed(["test"])[0])
    except Exception as e:
        check("Models answer", False, f"{type(e).__name__}: {e}. Start Ollama (ollama serve), then ollama pull "
              f"llama3.2:3b, llama3.2:1b, {llm_calls.judge_model()} and embeddinggemma.")
        return False
    # "OK" (any case, maybe with a full stop) is the only right answer. A long or empty reply means the
    # model is still thinking out loud: as a judge it would be slow and miss the JSON fields.
    short = {m: v.strip(" .!\"'").lower() == "ok" for m, v in replies.items()}
    check("Models answer: " + ", ".join(f"{m} {v[:12]!r}" for m, v in replies.items())
          + f"; {llm_calls.EMBED_MODEL} (dimension {dim})", all(short.values()) and dim > 0,
          "a model that doesn't reply just OK is thinking (or the thinking switch isn't honoured): "
          f"{ {m: v[:80] for m, v in replies.items() if not short[m]} }. "
          "Try another judge: M06_JUDGE_MODEL=nemotron-mini python m06/check.py")
    return all(short.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  model: {llm_calls.describe()}\n")
    print(f"NCP-AAI M06 lab check  [{llm_calls.describe()}]\n")

    # 1. the models
    if not models_answer():
        return finish()
    desk_app.build_index(say=quiet)

    # 2. the test set
    items = testset.load()
    refs = testset.reference_ids(items)
    s = testset.stats(items, refs)
    small = {c: n for c, n in s["per_category"].items() if n < 3}
    check(f"Test set: {s['n']} items, {len(s['per_category'])} categories (smallest has "
          f"{min(s['per_category'].values())}), dev {s['splits']['dev']} / test {s['splits']['test']}, "
          f"every reference section resolves", 35 <= s["n"] <= 50 and not small and s["splits"]["dev"]
          and s["splits"]["test"] and not s["unresolved"], {"small": small, "unresolved": s["unresolved"]})

    # 3. retrieval
    r = retrieval_eval.run(say=quiet)
    alls = [x for x in r["rows"] if x["category"] == "all"]
    grid = {(x["mode"], x["k"]) for x in alls}
    monotone = all(x["recall"] <= y["recall"] + 1e-9 for m in retrieval_eval.MODES
                   for x, y in zip(sorted((a for a in alls if a["mode"] == m), key=lambda a: a["k"]),
                                   sorted((a for a in alls if a["mode"] == m), key=lambda a: a["k"])[1:]))
    planted = retrieval_eval.planted_example()
    check(f"retrieval_eval: {len(grid)} mode x k rows, recall@k never falls as k grows, Ragas ID-based = own "
          f"on the planted example {planted['own']} and on all rows ({r['mismatches']} differ); hybrid recall@3 "
          f"{next(x['recall'] for x in alls if x['mode'] == 'hybrid' and x['k'] == 3):.2f}",
          grid == {(m, k) for m in retrieval_eval.MODES for k in retrieval_eval.KS} and monotone
          and planted["own"] == planted["ragas"] == (1.0, 0.5) and r["mismatches"] == 0, planted)

    # 4. NAT config and plugins
    p = nat("validate", "--config_file", CFG)
    q = nat("info", "components", "-t", "evaluator")
    registered = [n for n in ("keywords_all", "refuses_when_unanswerable") if n in q.stdout]
    check(f"nat validate {CFG}; custom evaluators registered: {', '.join(registered) or 'none'}",
          p.returncode == 0 and "valid" in (p.stdout + p.stderr).lower() and len(registered) == 2,
          tail(p) + ([] if len(registered) == 2 else ["not registered: run pip install --no-deps -e m06"]))

    # 5-7. nat eval, configuration A
    a_dir = RUNS / "A"
    shutil.rmtree(a_dir, ignore_errors=True)
    p = nat("eval", "--config_file", CFG, "--reps", "2", *DEV)
    wo = json.loads((a_dir / "workflow_output.json").read_text()) if (a_dir / "workflow_output.json").exists() else []
    dev = [i for i in items if i["split"] == "dev"]
    ids = [w["id"] for w in wo]
    missing = [n for n in EVALUATORS if not (a_dir / f"{n}_output.json").exists()]
    prof = compare_configs.traces(a_dir, testset.by_question(items))
    llm_end = 0
    if (a_dir / "standardized_data_all.csv").exists():
        with (a_dir / "standardized_data_all.csv").open() as f:
            llm_end = sum(1 for row in csv.DictReader(f) if row["event_type"] == "LLM_END")
    check(f"nat eval --reps 2 (A, dev): {len(wo)} outputs for {len(dev)} items, IDs like {', '.join(ids[:2])}; "
          f"{llm_end} LLM_END rows; {len(EVALUATORS) - len(missing)}/{len(EVALUATORS)} evaluator files",
          p.returncode == 0 and len(wo) == 2 * len(dev) and all(i.endswith(("_rep0", "_rep1")) for i in ids)
          and llm_end > 0 and not missing and len(prof) == len(wo), missing or tail(p))
    side = compare_configs.sidecar(a_dir)
    with testset.open_index() as client:
        known = {row["id"] for row in testset.chunk_rows(client)}
    wanted = {compare_configs.split_id(i) for i in ids}
    passage_ids = [pid for rec in side.values() for pid in (x["id"] for x in rec.get("passages", []))]
    check(f"Passage sidecar: {len(side)} records for {len(wanted)} item-reps, {len(passage_ids)} passage IDs, "
          f"all in the index", set(side) == wanted and wanted and set(passage_ids) <= known,
          {"missing": sorted(wanted - set(side))[:5], "unknown": sorted(set(passage_ids) - known)[:5]})
    aa = outputs(a_dir, "answer_accuracy")
    bad = [x["id"] for x in aa if not isinstance(x.get("score"), (int, float)) or not 0 <= x["score"] <= 1
           or x.get("error") or "error" in (x.get("reasoning") or {})]
    check(f"NAT ragas Answer Accuracy: {len(aa)} item-reps scored, mean "
          f"{sum(x['score'] for x in aa) / max(len(aa), 1):.2f}, all in [0, 1] "
          f"(NAT turns an unparseable judge reply into 0; ragas_eval counts those as NaN)",
          len(aa) == len(wo) and not bad, bad[:5])

    # 8. ragas_eval, twice (the second from the cache)
    cache = testset.STATE / "check_ragas_cache"
    shutil.rmtree(cache, ignore_errors=True)
    first = ragas_eval.run(a_dir, cache_root=cache, say=quiet)
    second = ragas_eval.run(a_dir, cache_root=cache, say=quiet)
    nv = {m: first["metrics"][m] for m in ragas_eval.NVIDIA}
    with_passages = sum(1 for rec in side.values() if rec.get("rep", 0) == 0 and rec.get("passages") and rec.get("reply"))
    same = all(first["per_item"][k] == second["per_item"][k] for k in first["per_item"])
    ok = (nv["answer_accuracy"]["scored"] == len(dev) and all(v["nan"] == 0 and v["out_of_range"] == 0 for v in nv.values())
          and nv["context_relevance"]["scored"] == nv["response_groundedness"]["scored"] == with_passages
          and same and second["judge_calls"] == 0 and first["judge_calls"] > 0 and second["minutes"] <= first["minutes"])
    check(f"ragas_eval: " + ", ".join(f"{m} {v['scored']} scored (mean {v['mean']})" for m, v in nv.items())
          + f"; judge calls {first['judge_calls']} ({first['minutes']:.2f} min), then {second['judge_calls']} "
          f"from the cache ({second['minutes']:.2f} min), same scores: {same}", ok,
          {"metrics": first["metrics"], "second_calls": second["judge_calls"]})

    # 9. re-score without the desk
    r_dir = RUNS / "A_rescored"
    shutil.rmtree(r_dir, ignore_errors=True)
    p = nat("eval", "--config_file", CFG, "--skip_workflow", "--dataset", str(a_dir / "workflow_output.json"),
            "--override", "eval.general.output_dir", str(r_dir.relative_to(LABS)))
    before = {x["id"]: x["score"] for x in outputs(a_dir, "keywords_all")}
    after = {x["id"]: x["score"] for x in outputs(r_dir, "keywords_all")}
    check(f"nat eval --skip_workflow re-scored {len(after)} saved replies without running the desk "
          f"(no sidecar written: {not (r_dir / 'passages.jsonl').exists()}; same keywords_all scores: {before == after})",
          p.returncode == 0 and after and before == after and not (r_dir / "passages.jsonl").exists(), tail(p))

    # 10-11. the judges
    jc = judge_check.run([llm_calls.judge_model(), llm_calls.SELF_JUDGE], reps=3, say=quiet)
    parts, ok = [], True
    for model, res in jc["judges"].items():
        s1 = res["single"]
        parts.append(f"{model}: {s1['parseable']}/{s1['n']} parseable, agree {s1['agree']}, "
                     f"false pass {s1['false_pass']}, false fail {s1['false_fail']}")
        ok = ok and s1["parseable"] >= 0.9 * s1["n"]
    check("judge_check: " + "; ".join(parts), ok, {m: r["single"]["parseable"] for m, r in jc["judges"].items()})
    parts = [f"{m}: {r['pairs']['position_consistent']}/{r['pairs']['parseable']} position-consistent, "
             f"{r['pairs']['first_slot_picks']}/{r['pairs']['slots']} first-slot picks" for m, r in jc["judges"].items()]
    check("Pairwise (A/B and B/A): " + "; ".join(parts),
          all(r["pairs"]["parseable"] > 0 for r in jc["judges"].values()),
          {m: r["pairs"]["items"] for m, r in jc["judges"].items()})

    # 12-13. configuration B and the comparison
    b_dir = RUNS / "B"
    shutil.rmtree(b_dir, ignore_errors=True)
    p = nat("eval", "--config_file", CFG, "--reps", "2", *DEV, "--override", "eval.general.output_dir",
            str(b_dir.relative_to(LABS)), env={"OLLAMA_MODEL": "llama3.2:1b"})
    try:
        cmp = compare_configs.compare(a_dir, b_dir, say=quiet)
    except SystemExit as e:
        cmp = None
        check("B's run and compare_configs", False, f"{e}; {tail(p)}")
    if cmp:
        t = cmp["table"]["all"]
        filled = all(t[x][k] is not None for x in ("A", "B") for k in ("p50", "p95", "calls", "tokens", "pass_rate"))
        b_model = next((r.get("model") for r in compare_configs.sidecar(b_dir).values()), "?")
        check(f"B ({b_model}) vs A: pass rate {t['A']['pass_rate']:.2f} vs {t['B']['pass_rate']:.2f}, p50 "
              f"{t['A']['p50']:.2f} vs {t['B']['p50']:.2f} s, LLM calls {t['A']['calls']:.1f} vs {t['B']['calls']:.1f}, "
              f"tokens {t['A']['tokens']:.0f} vs {t['B']['tokens']:.0f} per item",
              p.returncode == 0 and b_model == "llama3.2:1b" and filled, tail(p))
        problems = [x for v in cmp["nel_problems"].values() for x in v]
        pv = f"{cmp['p_value']:.3f}" if cmp["p_value"] is not None else "none (no discordant pairs)"
        check(f"Paired flips: {len(cmp['regressions'])} regressions, {len(cmp['improvements'])} improvements; "
              f"McNemar p = {pv}; NEL results.jsonl + eval-desk.json fields present",
              not problems and (cmp["p_value"] is None) == (not cmp["regressions"] and not cmp["improvements"]),
              problems)
        info(f"VERDICT: {cmp['verdict']}")

    # 14. sampling: temperature 0 and a seed
    saved = {k: os.environ.get(k) for k in ("DESK_TEMPERATURE", "DESK_SEED")}
    os.environ.update(DESK_TEMPERATURE="0", DESK_SEED="42")
    prompt = [("user", "In one sentence, what does a USB-C dock do?")]
    one, two = llm_calls.chat(prompt), llm_calls.chat(prompt)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    spread = cmp["spread"].get("A", []) if cmp else []
    check(f"DESK_TEMPERATURE=0 DESK_SEED=42: the same call twice gives the same text ({one[:40]!r})",
          one == two and one, {"first": one, "second": two})
    info(f"A's rep-to-rep spread: {len(spread)} of {len(dev)} dev items changed verdict between reps"
         + (f" ({', '.join(spread)})" if spread else ""))

    # 15-17. feedback into the test set
    before = len(feedback.load(feedback.LOG))
    rec = feedback.rate("Where is order A1003?", "down", "wrong_fact", "check.py: planted rating",
                        "Order A1003 is still processing, awaiting stock.", say=quiet)
    after = feedback.load(feedback.LOG)
    rep = feedback.report(say=quiet)
    valid = (rec["key"] == "user_rating" and rec["score"] == 0 and rec["reason"] in feedback.REASONS
             and rec["trace_id"] and rec["reply"])
    check(f"feedback rate --answer down: record {rec['id']} appended ({len(after) - before} new); report counts "
          f"{rep['thumbs'].get('down', 0)} down, {rep['reasons'].get('wrong_fact', 0)} wrong_fact",
          valid and len(after) == before + 1 and after[-1]["id"] == rec["id"], rec)
    out = testset.STATE / "testset_feedback.jsonl"
    pr = feedback.promote(out, say=quiet)
    promoted = {k["feedback_id"] for k in pr["kept"]}
    skipped = dict(pr["skipped"])
    check(f"promote: {len(pr['kept'])} candidates ({', '.join(sorted(promoted))}); planted duplicate skipped: "
          f"{skipped.get('fb-seed-05', 'NO')}",
          {"fb-seed-02", "fb-seed-03"} <= promoted and "near-duplicate" in skipped.get("fb-seed-05", "")
          and "fb-seed-04" in skipped and all(k["needs_review"] and k["source"] == "feedback" for k in pr["kept"]),
          pr)
    f_dir = RUNS / "feedback"
    shutil.rmtree(f_dir, ignore_errors=True)
    p = nat("eval", "--config_file", CFG, "--override", "eval.general.dataset.file_path", str(out.relative_to(LABS)),
            "--override", "eval.general.output_dir", str(f_dir.relative_to(LABS)))
    scored = {x["id"] for x in outputs(f_dir, "keywords_all")}
    check(f"nat eval on the promoted file scored {len(scored)} new items ({', '.join(sorted(scored))})",
          p.returncode == 0 and scored == {k["id"] for k in pr["kept"]}, tail(p))

    # 18. triage
    tr = triage.triage(a_dir, say=quiet)
    per_item = tr["assigned"]
    exactly_one = len(per_item) == len(tr["failed"]) and all(v["bucket"] in triage.BUCKETS for v in per_item.values())
    side0 = compare_configs.sidecar(a_dir)
    consistent = all(not set(refs.get(i, [])) & set(p["id"] for p in side0[(i, per_item[i]["rep"])]["passages"])
                     for i in tr["retrieval_miss"])
    check(f"triage: {len(tr['failed'])} failed items, each in one bucket ({tr['totals']}); retrieval_miss "
          f"{tr['retrieval_miss'] or 'none'} = reference chunks not retrieved by the desk; fix first: "
          f"{', '.join(tr['fix_first'][:3]) or '-'}", exactly_one and consistent, tr["totals"])
    s2 = set(tr["step2_misses"] or [])
    info(f"step 2 (hybrid, k=3, no product filter) also missed {len(s2 & set(tr['retrieval_miss']))} of them")
    return finish()


def finish():
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
