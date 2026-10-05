"""Step 5: benchmark a release against the one before it: offline on a frozen set, then on live traffic.

    python m08/bench_release.py frozen                  # v7 vs v8 on the frozen set, one replica each
    python m08/bench_release.py live --minutes 3        # canary: balancer started with --canary 10

Start the fleet with a v8 replica first: python m08/fleet_obs.py --canary r3=v8
(v8 is the release candidate: llama3.2:1b instead of llama3.2:3b, see fleet_obs.VERSIONS).

frozen  The M6 test set's `test` split (27 questions), never changed between releases, so scores are
        comparable over time. Each version's results are stored in m08/state/bench/<version>.json;
        a version that already has results is not run again unless --rerun, which is the point:
        every new release is compared with the stored results of the one in production. A replica
        of each version answers every question once (directly, not through the balancer). Scoring is
        M6's: an item passes when keywords_all and refuses_when_unanswerable are both 1. Then:
          pass rate and p50/p95 latency per version, per category
          paired flips (items v7 passed and v8 failed, and the other way) and M6's exact McNemar test
          a release gate: BLOCK if the pass rate drops by more than --max-drop (default 0.05), or if
          McNemar says the regressions are significant; else PASS
        Results: m08/state/bench/compare_<old>_<new>.json. Run it on a schedule (CI, cron) against
        the running version as well, to catch drift that no release caused.
live    Sends mixed questions through the balancer for --minutes (the balancer splits them:
        --canary 10 sends 10% to v8), then compares the versions on what users got: requests,
        error rate and p95 from the balancer's per-version metrics (Prometheus if it is running,
        else the balancer's /metrics), and the share of answers that pass the same scoring.
"""
import argparse
import asyncio
import json
import sys
import time

import httpx

import obs_common as oc

sys.path.insert(0, str(oc.LABS / "m06"))
import compare_configs  # noqa: E402  (m06: McNemar and its verdict)
import evaluators  # noqa: E402  (m06: keywords_all and refuses_when_unanswerable)

BENCH = oc.STATE / "bench"


def score(item: dict, reply: str) -> bool:
    k, _ = evaluators.keywords_score(item.get("keywords"), reply)
    r, _ = evaluators.refusal_score(item["category"], reply)
    return evaluators.item_passes(k, r)


def replica_for(version: str) -> dict:
    fleet = json.loads((oc.STATE / "fleet.json").read_text())
    for r in fleet:
        if r["version"] == version:
            return r
    sys.exit(f"[ERROR] no replica runs {version}: start python m08/fleet_obs.py --canary r3=v8")


def run_frozen(version: str, items: list[dict], users: int) -> dict:
    r = replica_for(version)
    oc.say(f"[bench] {version} ({r['model']}) on {r['name']}: {len(items)} questions")
    by_q = {i["question"]: i for i in items}
    t = time.time()
    recs = asyncio.run(oc.closed_loop(r["url"], list(by_q), users))
    rows = []
    for rec in recs:
        item = by_q[rec["question"]]
        rows.append({"id": item["id"], "category": item["category"], "ok": rec["ok"],
                     "pass": rec["ok"] and score(item, rec.get("reply", "")),
                     "latency_s": rec["latency_s"], "reply": rec.get("reply", "")[:300]})
    out = {"version": version, "model": r["model"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "duration_s": round(time.time() - t, 1), "items": sorted(rows, key=lambda x: x["id"])}
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"{version}.json").write_text(json.dumps(out, indent=1))
    return out


def stats(rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows if r["ok"]]
    return {"n": len(rows), "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3) if rows else None,
            "p50_s": oc.percentile(lat, 50), "p95_s": oc.percentile(lat, 95)}


def frozen(a) -> dict:
    items = [r for r in oc.testset() if r["split"] == "test"][:a.limit or None]
    res = {}
    for v in (a.old, a.new):
        stored = BENCH / f"{v}.json"
        if stored.exists() and not a.rerun:
            res[v] = json.loads(stored.read_text())
            oc.say(f"[bench] {v}: stored results from {res[v]['time']} ({res[v]['model']}), not run again")
        else:
            res[v] = run_frozen(v, items, a.users)
    old, new = ({r["id"]: r for r in res[v]["items"]} for v in (a.old, a.new))
    ids = sorted(set(old) & set(new))
    so, sn = stats([old[i] for i in ids]), stats([new[i] for i in ids])
    oc.say(f"\n{'':10} {'pass rate':>9} {'p50':>7} {'p95':>7}   ({len(ids)} questions)")
    for v, s in ((a.old, so), (a.new, sn)):
        oc.say(f"{v + ' ' + res[v]['model']:22} {s['pass_rate']:>5.0%} {s['p50_s'] or 0:6.1f}s {s['p95_s'] or 0:6.1f}s")
    cats = sorted({old[i]["category"] for i in ids})
    oc.say("\nby category       " + "  ".join(f"{c[:14]:>14}" for c in cats))
    for v, rows in ((a.old, old), (a.new, new)):
        cells = []
        for c in cats:
            rs = [rows[i] for i in ids if rows[i]["category"] == c]
            cells.append(f"{sum(r['pass'] for r in rs)}/{len(rs)}".rjust(14))
        oc.say(f"{v:17} " + "  ".join(cells))
    regress = [i for i in ids if old[i]["pass"] and not new[i]["pass"]]
    improve = [i for i in ids if new[i]["pass"] and not old[i]["pass"]]
    p = compare_configs.mcnemar(len(regress), len(improve))
    oc.say(f"\n[flips] {len(regress)} regressions ({', '.join(regress) or '-'}), "
           f"{len(improve)} improvements ({', '.join(improve) or '-'})")
    oc.say("[mcnemar] " + compare_configs.verdict(len(regress), len(improve), p, len(ids))
           .replace(" A and B", f" {a.old} and {a.new}").replace("B lost", f"{a.new} lost").replace("B gained", f"{a.new} gained"))
    drop = (so["pass_rate"] or 0) - (sn["pass_rate"] or 0)
    reasons = []
    if drop > a.max_drop:
        reasons.append(f"pass rate dropped {drop:.0%} (more than the {a.max_drop:.0%} allowed)")
    if p is not None and p < 0.05 and len(regress) > len(improve):
        reasons.append(f"significant regressions (McNemar p = {p:.3g})")
    gate = "BLOCK" if reasons else "PASS"
    oc.say(f"[gate] {gate}: " + ("; ".join(reasons) if reasons else
                                 f"pass rate change {-drop:+.0%} within -{a.max_drop:.0%}, no significant regressions"))
    out = {"old": a.old, "new": a.new, "questions": len(ids), "stats": {a.old: so, a.new: sn},
           "regressions": regress, "improvements": improve, "mcnemar_p": p, "max_drop": a.max_drop,
           "gate": gate, "reasons": reasons}
    (BENCH / f"compare_{a.old}_{a.new}.json").write_text(json.dumps(out, indent=1))
    oc.say(f"[bench] wrote m08/state/bench/compare_{a.old}_{a.new}.json")
    return out


def per_version_live(url: str, minutes: float = 10) -> dict:
    """requests, error rate and p95 per version, from Prometheus (if running) or the balancer's counters."""
    out = {}
    if oc.prom_up():
        win = f"{int(minutes) + 1}m"
        q = {"requests": f"sum by (version) (increase(lb_requests_total[{win}]))",
             "errors": f'sum by (version) (increase(lb_requests_total{{code!~"2.."}}[{win}]))',
             "p95": f"histogram_quantile(0.95, sum by (le, version) (rate(lb_request_seconds_bucket[{win}])))"}
        for key, expr in q.items():
            for row in oc.prom_query(expr) or []:
                v = row["metric"].get("version")
                out.setdefault(v, {})[key] = float(row["value"][1])
        out["_source"] = f"Prometheus, last {win}"
        return out
    lb = oc.scrape(url + "/metrics")
    for (n, ls), val in lb.items():
        d = dict(ls)
        if n == "lb_requests_total":
            e = out.setdefault(d["version"], {"requests": 0, "errors": 0})
            e["requests"] += val
            if not d["code"].startswith("2"):
                e["errors"] += val
    out["_source"] = "the balancer's /metrics (no Prometheus)"
    return out


def live(a) -> dict:
    status = httpx.get(a.url + "/status", timeout=5).json()
    oc.say(f"[live] balancer canary: {status.get('canary_percent', 0):g}% to v8 for {a.minutes:g} min")
    items = {r["question"]: r for r in oc.testset(("manual_exact", "manual_paraphrase", "order", "mixed"))}
    recs = asyncio.run(oc.closed_loop(a.url, list(items), a.users, duration=a.minutes * 60))
    by_v: dict[str, list] = {}
    for rec in recs:
        by_v.setdefault(rec.get("version") or "-", []).append(rec)
    m = per_version_live(a.url, a.minutes)
    oc.say(f"\n{'version':8} {'requests':>8} {'errors':>7} {'p95':>7} {'answers pass':>13}   (metrics: {m['_source']})")
    out = {"minutes": a.minutes, "versions": {}}
    for v in sorted(by_v):
        rs = by_v[v]
        passed = sum(1 for r in rs if r["ok"] and score(items[r["question"]], r.get("reply", "")))
        mv = m.get(v, {})
        req = mv.get("requests", len(rs))
        err = (mv.get("errors", 0) / req) if req else 0
        p95 = mv.get("p95") or oc.percentile([r["latency_s"] for r in rs if r["ok"]], 95) or 0
        oc.say(f"{v:8} {req:8.0f} {err:7.0%} {p95:6.1f}s {passed:>6}/{len(rs):<6}")
        out["versions"][v] = {"requests": req, "error_rate": round(err, 3), "p95_s": round(p95, 2),
                              "pass": passed, "sent": len(rs)}
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "live.json").write_text(json.dumps(out, indent=1))
    oc.say("[live] wrote m08/state/bench/live.json  (the Grafana dashboard shows the same split, by version)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("frozen", help="both versions on the frozen test split, then the gate")
    f.add_argument("--old", default="v7")
    f.add_argument("--new", default="v8")
    f.add_argument("--rerun", action="store_true", help="run versions that already have stored results again")
    f.add_argument("--max-drop", type=float, default=0.05)
    f.add_argument("--users", type=int, default=1)
    f.add_argument("--limit", type=int, default=0, help="only the first N questions (the check uses 4)")
    l = sub.add_parser("live", help="canary traffic through the balancer, compared by version")
    l.add_argument("--url", default=oc.BALANCER)
    l.add_argument("--minutes", type=float, default=3)
    l.add_argument("--users", type=int, default=2)
    a = ap.parse_args()
    frozen(a) if a.mode == "frozen" else live(a)


if __name__ == "__main__":
    main()
