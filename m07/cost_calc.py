"""Step 5: turn the measured throughput into a size and a cost, for a price you look up yourself.

    python m07/cost_calc.py --target-rps 5 --slo-p95 10 --price-per-hour 2.50
    python m07/cost_calc.py --target-rps 5 --slo-p95 10 --price-per-hour 2.50 --headroom 2 --utilisation 0.7

The unit of capacity is what load_test.py measured: one model tier (on the laptop, one Ollama;
in a cluster, one GPU serving the model) with the agent replicas in front of it. From
m07/state/load.csv (the latest run for each replica count) it takes, per replica count:

  capacity per unit   the highest throughput among the levels whose p95 latency meets --slo-p95
                      and that had no errors (a level that breaks the SLO doesn't count, however
                      fast it was)
  units for the load  ceil(target / (capacity x utilisation)): --utilisation (default 0.8) keeps
                      each unit below its measured maximum, so a burst doesn't push p95 over the SLO
  N+1                 plus --headroom spare units (default 1) so one unit can fail, or be drained
                      for an upgrade, and the rest still carry the target load
  cost                units x --price-per-hour, per hour and per month (730 h), and per 1,000
                      requests at the target load

--price-per-hour is what one unit costs you per hour, for example the on-demand price of the GPU
instance that would host the model tier. There is no default and the script states no prices:
look the price up on your cloud's pricing page (prices change). The laptop's numbers are a
stand-in; the method is the point: measure a unit, size from the SLO, add headroom, then cost it.
Results go to m07/state/cost.json.
"""
import argparse
import csv
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
STATE = HERE / "state"
LOAD_CSV = STATE / "load.csv"
HOURS_PER_MONTH = 730


def latest_rows(path: pathlib.Path = LOAD_CSV) -> dict[int, list[dict]]:
    """The rows of the most recent run for each replica count."""
    if not path.exists():
        raise SystemExit(f"[ERROR] {path} not found: run m07/load_test.py first")
    rows = list(csv.DictReader(path.open()))
    by_n: dict[int, list[dict]] = {}
    for n in sorted({int(r["replicas"]) for r in rows}):
        mine = [r for r in rows if int(r["replicas"]) == n]
        last = max(r["run_id"] for r in mine)
        by_n[n] = [r for r in mine if r["run_id"] == last]
    return by_n


def num(v: str) -> float | None:
    return float(v) if v not in ("", None, "None") else None


def capacity(rows: list[dict], slo_p95: float) -> dict | None:
    """The fastest level that meets the SLO with no errors, or None."""
    good = [r for r in rows if num(r["p95_s"]) is not None and num(r["p95_s"]) <= slo_p95
            and float(r["error_rate"]) == 0]
    if not good:
        return None
    best = max(good, key=lambda r: float(r["throughput_rps"]))
    return {"concurrency": int(best["concurrency"]), "throughput_rps": float(best["throughput_rps"]),
            "p95_s": num(best["p95_s"]), "run_id": best["run_id"]}


def size(cap_rps: float, target_rps: float, utilisation: float, headroom: int, price: float) -> dict:
    units = math.ceil(target_rps / (cap_rps * utilisation))
    total = units + headroom
    per_hour = total * price
    return {"units_for_load": units, "headroom": headroom, "units_total": total,
            "cost_per_hour": round(per_hour, 4), "cost_per_month": round(per_hour * HOURS_PER_MONTH, 2),
            "cost_per_1000_requests": round(per_hour / (target_rps * 3600) * 1000, 6),
            "load_per_unit_at_target": round(target_rps / units / cap_rps, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-rps", type=float, required=True, help="requests per second to serve")
    ap.add_argument("--slo-p95", type=float, required=True, help="p95 latency target, seconds")
    ap.add_argument("--price-per-hour", type=float, required=True, help="what one unit costs per hour (look it up)")
    ap.add_argument("--headroom", type=int, default=1, help="spare units (N+1 = 1)")
    ap.add_argument("--utilisation", type=float, default=0.8, help="fraction of measured capacity to plan for")
    a = ap.parse_args()
    out = {"target_rps": a.target_rps, "slo_p95_s": a.slo_p95, "price_per_hour": a.price_per_hour,
           "headroom": a.headroom, "utilisation": a.utilisation, "by_replicas": {}}
    print(f"[cost] target {a.target_rps:g} req/s, p95 <= {a.slo_p95:g} s, {a.utilisation:.0%} utilisation, "
          f"N+{a.headroom}, {a.price_per_hour:g} per unit-hour")
    for n, rows in latest_rows().items():
        cap = capacity(rows, a.slo_p95)
        if cap is None:
            fastest = min((num(r["p95_s"]) for r in rows if num(r["p95_s"]) is not None), default=None)
            print(f"  {n} agent replica(s): no measured level meets p95 <= {a.slo_p95:g} s "
                  f"(best p95 {fastest:.2f} s). More replicas won't fix that: the model tier is too slow "
                  f"for this SLO (a faster GPU, a smaller or quantised model, or caching)")
            out["by_replicas"][n] = {"capacity": None, "best_p95_s": fastest}
            continue
        s = size(cap["throughput_rps"], a.target_rps, a.utilisation, a.headroom, a.price_per_hour)
        out["by_replicas"][n] = {"capacity": cap, **s}
        print(f"  {n} agent replica(s) per unit: {cap['throughput_rps']:.3f} req/s per unit "
              f"(concurrency {cap['concurrency']}, p95 {cap['p95_s']:.2f} s)")
        print(f"      units {s['units_for_load']} + {s['headroom']} spare = {s['units_total']}  "
              f"cost {s['cost_per_hour']:.2f}/h  {s['cost_per_month']:.0f}/month  "
              f"{s['cost_per_1000_requests']:.4f} per 1,000 requests")
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "cost.json").write_text(json.dumps(out, indent=1))
    print("[cost] saved to m07/state/cost.json")


if __name__ == "__main__":
    main()
