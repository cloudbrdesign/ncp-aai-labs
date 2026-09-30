"""Print two `nat eval` output folders side by side: rails off vs on, or Ollama vs NIM.

    python m05/compare_runs.py                                            # m05/state/nat/bare vs guarded
    python m05/compare_runs.py m05/state/nat/bare m05/state/nat/bare_nim  # any two output_dir folders

For every evaluator both runs have, the average score (from <name>_output.json), then the
LLM calls and tokens the profiler recorded (LLM_END rows in standardized_data_all.csv)
and the p90 workflow time from inference_optimization.json. With about a dozen items at
concurrency 1 these are one small sample: the confidence intervals and forecasts in the
profiler's files mean little here.

For an Ollama vs NIM comparison, write the second run to its own folder:
    LLM_PROVIDER=nim nat eval --config_file m05/configs/desk_eval.yml \
        --override eval.general.output_dir m05/state/nat/bare_nim
"""
import argparse
import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT = [HERE / "state" / "nat" / "bare", HERE / "state" / "nat" / "guarded"]
NOT_EVALUATORS = {"workflow_output.json", "all_requests_profiler_traces.json", "inference_optimization.json",
                  "workflow_profiling_metrics.json", "config_metadata.json"}


def scores(folder: pathlib.Path) -> dict:
    out = {}
    for f in sorted(folder.glob("*_output.json")):
        if f.name in NOT_EVALUATORS:
            continue
        data = json.loads(f.read_text())
        if isinstance(data, dict) and "average_score" in data:
            out[f.name[:-len("_output.json")]] = data["average_score"]
    return out


def profile(folder: pathlib.Path) -> dict:
    """LLM calls and tokens from the profiler CSV, per run and per item."""
    calls = tokens = 0
    items = set()
    path = folder / "standardized_data_all.csv"
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                items.add(row.get("example_number"))
                if row.get("event_type") == "LLM_END":
                    calls += 1
                    tokens += int(float(row.get("total_tokens") or 0))
    n = max(len(items), 1)
    p90 = None
    opt = folder / "inference_optimization.json"
    if opt.exists():
        ci = json.loads(opt.read_text()).get("confidence_intervals", {})
        p90 = ci.get("workflow_run_time_confidence_intervals", {}).get("p90")
    return {"items": len(items), "llm_calls": calls, "llm_calls_per_item": round(calls / n, 2),
            "tokens": tokens, "tokens_per_item": round(tokens / n, 1),
            "workflow_p90_s": round(p90, 3) if p90 is not None else None}


def compare(a: pathlib.Path, b: pathlib.Path) -> dict:
    return {"a": {"scores": scores(a), "profile": profile(a)}, "b": {"scores": scores(b), "profile": profile(b)}}


def show(a: pathlib.Path, b: pathlib.Path, say=print) -> dict:
    r = compare(a, b)
    width = 24
    say(f"{'':<{width}}{a.name:>14}{b.name:>14}")
    for name in sorted(set(r["a"]["scores"]) | set(r["b"]["scores"])):
        va, vb = r["a"]["scores"].get(name, ""), r["b"]["scores"].get(name, "")
        say(f"{name:<{width}}{va!s:>14}{vb!s:>14}")
    say("-- from the profiler CSV")
    for key in ("items", "llm_calls", "llm_calls_per_item", "tokens", "tokens_per_item", "workflow_p90_s"):
        say(f"{key:<{width}}{r['a']['profile'][key]!s:>14}{r['b']['profile'][key]!s:>14}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", type=pathlib.Path, default=DEFAULT)
    a = ap.parse_args()
    if len(a.folders) != 2:
        ap.error("give two output folders (or none for bare vs guarded)")
    for f in a.folders:
        if not f.is_dir():
            raise SystemExit(f"[FAIL] {f} does not exist. Run nat eval first (see m05/README.md).")
    show(*a.folders)


if __name__ == "__main__":
    main()
