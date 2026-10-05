"""Step 6: a small data flywheel: production logs -> datasets -> a cheaper model -> a report for a person.

    python m08/flywheel.py run                              # candidate llama3.2:1b on the v7 traffic logged so far
    python m08/flywheel.py run --candidate llama3.2:1b --eval-size 8 --icl 2
    python m08/flywheel.py decide --workload draft --decision reject --by "your name" --note "..."
    python m08/flywheel.py registry                         # what runs, what was tried, who decided

The idea is NVIDIA's Data Flywheel Blueprint, on a laptop. Production traffic is logged (desk_obs.py
writes every model call to m08/state/logstore/requests.jsonl in the blueprint's log shape:
workload_id, client_id, request, response). From those logs, for each workload (each desk node
that calls the model: plan, draft, critique, sql), the flywheel:

  1. builds datasets   deduplicates the requests, holds out an evaluation set (--eval-size per
                       workload) and keeps the rest as a pool of in-context examples
  2. runs experiments  the candidate model answers each held-out request twice, as the blueprint's
                       evaluation types do:
                         base-eval  the logged request as it is
                         icl-eval   with --icl examples from the pool added to the system message
                       The production model's logged response is the reference.
  3. scores            free-text workloads (draft): an LLM judge (M6's, qwen3:4b) rates how similar
                       the candidate's answer is to the reference, 1 to 10, as the blueprint's
                       `similarity` metric does; structured workloads (plan, critique, sql): exact
                       match of the parsed fields (plan steps; critique score; SQL result rows)
  4. reports           m08/state/flywheel/report.md: per workload and evaluation type the score,
                       the candidate's latency next to production's, and a suggestion. It promotes
                       nothing: a person reads the report and records a decision with `decide`.

Everything is versioned in m08/state/registry.json: the release in production (its model, a hash
of each workload's prompt, a hash of its config), every flywheel run (candidate, dataset hash,
scores) and every decision with who made it and when. The blueprint's third evaluation type,
customized-eval (after fine-tuning with NeMo Customizer on GPUs), is taught from the docs; this lab
does not fine-tune.

Offline self-test: M08_FAKE_LLM=1 (M6's scripted model and judge; it proves the plumbing only).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import obs_common as oc

if os.environ.get("M08_FAKE_LLM") == "1":
    os.environ["M06_FAKE_LLM"] = "1"
for sub in ("m03", "m04", "m06"):                 # router.py and orders_sql.py (m04), critic.py (m03)
    sys.path.insert(0, str(oc.LABS / sub))
import llm_calls  # noqa: E402  (m06: the desk's model switch and the judge client)

LOG = oc.STATE / "logstore" / "requests.jsonl"
OUT = oc.STATE / "flywheel"
REGISTRY = oc.STATE / "registry.json"
STRUCTURED = {"plan", "critique", "sql"}
JUDGE_PROMPT = """You compare two answers from a customer support desk to the same request.
Rate from 1 to 10 how closely the CANDIDATE answer matches the REFERENCE answer in meaning and facts
(1 = unrelated or contradicts it, 10 = the same facts and advice; wording does not matter).

REQUEST:
{request}

REFERENCE:
{reference}

CANDIDATE:
{candidate}"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_logs(client_id: str) -> dict[str, list[dict]]:
    if not LOG.exists():
        sys.exit("[ERROR] no request log yet: run the fleet and send traffic first (lessons 8.1-8.3)")
    by_w: dict[str, list[dict]] = {}
    seen = set()
    for line in LOG.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("client_id") != client_id:
            continue
        key = (rec["workload_id"], json.dumps(rec["request"]["messages"], sort_keys=True))
        if key in seen:                     # deduplicate: the same request asked again adds nothing
            continue
        seen.add(key)
        by_w.setdefault(rec["workload_id"], []).append(rec)
    return by_w


def split(recs: list[dict], eval_size: int, seed: int = 8) -> tuple[list[dict], list[dict]]:
    recs = sorted(recs, key=lambda r: json.dumps(r["request"]["messages"]))
    random.Random(seed).shuffle(recs)
    return recs[:eval_size], recs[eval_size:]


def with_icl(messages: list[dict], examples: list[dict]) -> list[dict]:
    """The blueprint's icl-eval: example request/response pairs added to the system message."""
    shots = "\n\n".join(
        f"Example {i + 1}\nRequest: {next(m['content'] for m in ex['request']['messages'] if m['role'] == 'user')}\n"
        f"Response: {ex['response']['choices'][0]['message']['content']}" for i, ex in enumerate(examples))
    out = [dict(m) for m in messages]
    sys_msg = next((m for m in out if m["role"] == "system"), None)
    if sys_msg is None:
        out.insert(0, {"role": "system", "content": ""})
        sys_msg = out[0]
    sys_msg["content"] = sys_msg["content"] + "\n\nExamples of good responses:\n\n" + shots
    return out


def schema_for(workload: str):
    import importlib
    if workload == "plan":
        return importlib.import_module("router").Plan
    if workload == "critique":
        return importlib.import_module("critic").Grade
    if workload == "sql":
        return importlib.import_module("orders_sql").SqlQuery
    return None


def candidate_answer(workload: str, messages: list[dict]) -> tuple[str, float]:
    model = llm_calls.chat_model()
    pairs = [(m["role"], m["content"]) for m in messages]
    t = time.perf_counter()
    schema = schema_for(workload)
    if schema is None:
        content = model.invoke(pairs).content.strip()
    else:
        parsed = model.with_structured_output(schema).invoke(pairs)
        content = parsed.model_dump_json() if parsed is not None else "{}"
    return content, time.perf_counter() - t


def same_structured(workload: str, ref: str, cand: str) -> bool:
    try:
        a, b = json.loads(ref), json.loads(cand)
    except ValueError:
        return False
    if workload == "plan":
        key = lambda p: [(s.get("action"), s.get("order_id", ""), s.get("product", "")) for s in p.get("steps", [])]  # noqa: E731
        return key(a) == key(b)
    if workload == "critique":
        return a.get("score") == b.get("score")
    if workload == "sql":
        import orders_sql
        try:
            return orders_sql.run_readonly(a["sql"]) == orders_sql.run_readonly(b["sql"])
        except Exception:
            return False
    return a == b


def judge(request: str, reference: str, candidate: str, client) -> int:
    model = llm_calls.judge_model()
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=200, **llm_calls.judge_args(model),
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(request=request, reference=reference,
                                                                   candidate=candidate)}],
        response_format=llm_calls.json_schema_format(
            "similarity", {"similarity": {"type": "integer"}, "reason": {"type": "string"}}, ["similarity", "reason"]))
    try:
        return max(1, min(10, int(json.loads(r.choices[0].message.content or "{}").get("similarity", 1))))
    except (ValueError, TypeError):
        return 1


def registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"releases": [], "flywheel_runs": [], "decisions": []}


def record_release(reg: dict, client_id: str, by_w: dict) -> None:
    fleet = json.loads((oc.STATE / "fleet.json").read_text()) if (oc.STATE / "fleet.json").exists() else []
    version = client_id.removeprefix("desk-")
    model = next((r["model"] for r in fleet if r["version"] == version), "")
    prompts = {w: sha(next((m["content"] for m in recs[0]["request"]["messages"] if m["role"] == "system"), ""))
               for w, recs in by_w.items()}
    entry = {"version": version, "model": model, "prompt_sha": prompts,
             "config_sha": sha((oc.HERE / "configs" / "desk_obs.yml").read_text()),
             "code_sha": sha((oc.HERE / "desk_obs.py").read_text()), "status": "production"}
    if not any(r["version"] == version and r["prompt_sha"] == prompts and r["model"] == model for r in reg["releases"]):
        entry["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        reg["releases"].append(entry)


def run(a) -> None:
    os.environ["OLLAMA_MODEL"] = a.candidate            # the desk's chat model is now the candidate
    by_w = load_logs(a.client)
    if not by_w:
        sys.exit(f"[ERROR] no logged requests from {a.client}")
    client = llm_calls.judge_client()
    oc.say(f"[flywheel] logs: {sum(len(v) for v in by_w.values())} distinct requests from {a.client} "
           f"({', '.join(f'{w} {len(v)}' for w, v in sorted(by_w.items()))})")
    oc.say(f"[flywheel] candidate {a.candidate}; judge {llm_calls.judge_model()}; eval {a.eval_size} per workload, "
           f"icl {a.icl} examples")
    results, details = {}, []
    for w in sorted(by_w):
        held, pool = split(by_w[w], a.eval_size)
        if not held:
            continue
        data_sha = sha(json.dumps([r["request"]["messages"] for r in held], sort_keys=True))
        res = {"n": len(held), "pool": len(pool), "dataset_sha": data_sha,
               "prod_latency_s": round(sum(r.get("latency_s", 0) for r in held) / len(held), 2)}
        for etype in ("base-eval", "icl-eval"):
            if etype == "icl-eval" and (a.icl == 0 or not pool):
                continue
            scores, lat = [], []
            for r in held:
                msgs = r["request"]["messages"] if etype == "base-eval" else with_icl(r["request"]["messages"], pool[:a.icl])
                ref = r["response"]["choices"][0]["message"]["content"]
                try:
                    cand, took = candidate_answer(w, msgs)
                except Exception as e:
                    cand, took = f"error: {type(e).__name__}", 0.0
                if w in STRUCTURED:
                    s = 1.0 if same_structured(w, ref, cand) else 0.0
                else:
                    user = next((m["content"] for m in r["request"]["messages"] if m["role"] == "user"), "")
                    s = judge(user[:1500], ref, cand, client) / 10
                scores.append(s)
                lat.append(took)
                details.append({"workload": w, "eval": etype, "reference": ref[:300], "candidate": cand[:300], "score": s})
            res[etype] = {"score": round(sum(scores) / len(scores), 3),
                          "metric": "exact match" if w in STRUCTURED else "judge similarity / 10",
                          "latency_s": round(sum(lat) / len(lat), 2)}
            oc.say(f"[{w:8}] {etype:9} {res[etype]['score']:5.2f} ({res[etype]['metric']}), "
                   f"latency {res[etype]['latency_s']:.2f} s vs production {res['prod_latency_s']:.2f} s, n={len(held)}")
        best = max((res[e]["score"] for e in ("base-eval", "icl-eval") if e in res), default=0)
        res["suggestion"] = ("worth a closer look" if best >= a.threshold else "keep the production model")
        results[w] = res
    reg = registry()
    record_release(reg, a.client, by_w)
    run_id = time.strftime("fw-%Y%m%d-%H%M%S")
    reg["flywheel_runs"].append({"run_id": run_id, "client_id": a.client, "candidate": a.candidate,
                                 "judge": llm_calls.judge_model(), "threshold": a.threshold, "results": results,
                                 "status": "awaiting human review"})
    OUT.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1))
    (OUT / f"{run_id}_details.jsonl").write_text("".join(json.dumps(d) + "\n" for d in details))
    lines = [f"# Flywheel run {run_id}", "",
             f"Production: `{a.client}`. Candidate: `{a.candidate}`. Judge: `{llm_calls.judge_model()}`.", "",
             "| workload | n | eval | score | candidate latency | production latency | suggestion |",
             "|---|---|---|---|---|---|---|"]
    for w, r in results.items():
        for e in ("base-eval", "icl-eval"):
            if e in r:
                lines.append(f"| {w} | {r['n']} | {e} | {r[e]['score']:.2f} ({r[e]['metric']}) | "
                             f"{r[e]['latency_s']:.2f} s | {r['prod_latency_s']:.2f} s | {r['suggestion']} |")
    lines += ["", f"Suggestion rule: best score >= {a.threshold}. Nothing is promoted automatically: read the",
              f"examples in {run_id}_details.jsonl, then record a decision per workload:", "",
              "    python m08/flywheel.py decide --workload draft --decision promote|reject --by NAME --note TEXT", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    oc.say(f"[flywheel] {run_id}: report m08/state/flywheel/report.md, examples {run_id}_details.jsonl; "
           "registry m08/state/registry.json (status: awaiting human review)")


def decide(a) -> None:
    reg = registry()
    if not reg["flywheel_runs"]:
        sys.exit("[ERROR] no flywheel run to decide on")
    last = reg["flywheel_runs"][-1]
    if a.workload not in last["results"]:
        sys.exit(f"[ERROR] workload {a.workload} is not in run {last['run_id']}")
    reg["decisions"].append({"run_id": last["run_id"], "workload": a.workload, "candidate": last["candidate"],
                             "decision": a.decision, "by": a.by, "note": a.note,
                             "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    decided = {d["workload"] for d in reg["decisions"] if d["run_id"] == last["run_id"]}
    if decided >= set(last["results"]):
        last["status"] = "reviewed"
    REGISTRY.write_text(json.dumps(reg, indent=1))
    oc.say(f"[registry] {a.workload}: {a.decision} {last['candidate']} (run {last['run_id']}), by {a.by}")


def show(_a) -> None:
    reg = registry()
    for r in reg["releases"]:
        oc.say(f"[release] {r['version']} {r['model']} ({r['status']}): config {r['config_sha']}, "
               f"code {r['code_sha']}, prompts " + ", ".join(f"{w} {h}" for w, h in sorted(r["prompt_sha"].items())))
    for f in reg["flywheel_runs"]:
        oc.say(f"[run]     {f['run_id']}: {f['candidate']} vs {f['client_id']}, {f['status']}")
    for d in reg["decisions"]:
        oc.say(f"[decided] {d['time']} {d['workload']}: {d['decision']} {d['candidate']} by {d['by']}"
               + (f" ({d['note']})" if d["note"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--client", default="desk-v7", help="whose logged traffic (client_id)")
    r.add_argument("--candidate", default="llama3.2:1b")
    r.add_argument("--eval-size", type=int, default=8)
    r.add_argument("--icl", type=int, default=2)
    r.add_argument("--threshold", type=float, default=0.8)
    d = sub.add_parser("decide")
    d.add_argument("--workload", required=True)
    d.add_argument("--decision", choices=["promote", "reject"], required=True)
    d.add_argument("--by", required=True)
    d.add_argument("--note", default="")
    sub.add_parser("registry")
    a = ap.parse_args()
    {"run": run, "decide": decide, "registry": show}[a.cmd](a)


if __name__ == "__main__":
    main()
