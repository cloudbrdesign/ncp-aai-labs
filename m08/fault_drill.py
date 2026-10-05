"""Step 4: break the desk on purpose, watch the alerts fire, then find the cause in a trace.

    python m08/fault_drill.py                      # fleet + balancer (+ Prometheus) already running
    python m08/fault_drill.py --slow-s 30 --requests 6

Three phases, each sending desk questions through the balancer (2 users, closed loop):
  1. baseline   no fault: the numbers to compare with
  2. slow tool  m08/state/faults.json makes manual_search (the RAG tool) wait --slow-s seconds.
                Users see slow answers; Prometheus' DeskSlowAnswers alert (p95 > 20 s) fires.
  3. failing    the order_status tool raises an error. Order questions fail; DeskHighErrorRate
                (more than 10% of requests without a 2xx) and DeskToolErrors fire.
Then the diagnosis: the alerts say what users suffer (slow, failing) but not why. For the slowest
request of phase 2 and one failed request of phase 3, the script takes the request ID the
balancer returned, finds the trace ID (x-trace-id, also in m08/state/logstore/turns.jsonl), and
prints that request's span tree from the trace files: every desk node, model call and tool call
with its time, the slow one marked SLOW and the failing one marked ERROR with its message.
The same traces are in Phoenix (http://localhost:6006, project support-desk).

Without Prometheus (the offline check), the alerts are worked out from the balancer's /metrics.
Results: m08/state/drill.json. The fault file is removed at the end, also on Ctrl-C.
"""
import argparse
import asyncio
import json
import sys
import time

import httpx

import obs_common as oc

FAULTS = oc.STATE / "faults.json"


def set_fault(fault: dict | None) -> None:
    if fault:
        FAULTS.write_text(json.dumps(fault))
    else:
        FAULTS.unlink(missing_ok=True)


def phase(name: str, url: str, questions: list[str], fault: dict | None, concurrency: int) -> dict:
    set_fault(fault)
    oc.say(f"\n== {name}: {len(questions)} requests, {concurrency} users"
           + (f", fault {json.dumps(fault)}" if fault else ", no fault"))
    t = time.time()
    recs = asyncio.run(oc.closed_loop(url, questions, concurrency))
    s = oc.summary(recs)
    p95 = f"{s['p95_s']:.1f} s" if s["p95_s"] is not None else "-"
    oc.say(f"[{name}] {s['ok']}/{s['requests']} ok, {s['errors']} failed, p95 {p95}, {time.time() - t:.0f} s")
    return {"name": name, "fault": fault, "summary": s, "requests": recs}


def offline_alerts(url: str, slow_threshold: float) -> list[dict]:
    """No Prometheus: the same conditions as m08/observability/alerts.yml, from the counters so far."""
    lb = oc.scrape(url + "/metrics")
    fleet = json.loads((oc.STATE / "fleet.json").read_text())
    out = []
    reqs = oc.total(lb, "lb_requests_total")
    bad = reqs - sum(v for (n, ls), v in lb.items() if n == "lb_requests_total" and dict(ls)["code"].startswith("2"))
    if reqs and bad / reqs > 0.10:
        out.append({"name": "DeskHighErrorRate", "state": "firing", "summary": f"{bad / reqs:.0%} of requests failed"})
    for r in fleet:
        desk = oc.scrape(r["metrics"])
        for (n, ls), v in desk.items():
            if n == "desk_tool_calls_total" and dict(ls)["status"] == "error" and v > 0:
                out.append({"name": "DeskToolErrors", "state": "firing",
                            "summary": f"Tool {dict(ls)['tool']} failed {v:.0f} times ({r['name']})"})
    return out


def check_alerts(url: str, names: list[str], wait: float, slow_threshold: float) -> list[dict]:
    if oc.prom_up():
        fired = []
        for n in names:
            a = oc.wait_for_alert(n, wait)
            oc.say(f"[alert] {n}: " + (f"FIRING  {a['summary']}" if a else f"not firing after {wait:.0f} s"))
            if a:
                fired.append(a)
        return fired
    fired = [a for a in offline_alerts(url, slow_threshold) if a["name"] in names]
    for a in fired:
        oc.say(f"[alert] {a['name']}: FIRING  {a['summary']}   (no Prometheus: worked out from /metrics)")
    return fired


def diagnose(title: str, rec: dict | None) -> list[dict]:
    oc.say(f"\n== diagnosis: {title}")
    if rec is None:
        oc.say("[trace] no request to look at")
        return []
    trace_id = rec.get("trace_id") or (oc.turn_for(rec.get("request_id", "")) or {}).get("trace_id", "")
    oc.say(f"[request] {rec.get('request_id')} -> replica {rec.get('replica')} ({rec.get('version')}), "
           f"HTTP {rec.get('status')}, {rec['latency_s']:.1f} s: {rec['question'][:60]}")
    oc.say(f"[request] trace ID {trace_id or '?'}")
    time.sleep(1.5)                         # the file exporter writes asynchronously
    return oc.show_trace(trace_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=oc.BALANCER)
    ap.add_argument("--requests", type=int, default=6, help="requests per phase")
    ap.add_argument("--users", type=int, default=2)
    ap.add_argument("--slow-s", type=float, default=30, help="delay added to manual_search in phase 2")
    ap.add_argument("--wait", type=float, default=90, help="seconds to wait for each alert")
    a = ap.parse_args()
    try:
        httpx.get(a.url + "/health", timeout=3)
    except httpx.HTTPError:
        sys.exit(f"[ERROR] no balancer at {a.url}: start m08/fleet_obs.py and m08/balancer_obs.py first")
    manual = [r["question"] for r in oc.testset(("manual_exact",))]
    orders = [r["question"] for r in oc.testset(("order", "mixed"))]
    mixed = [q for pair in zip(manual, orders) for q in pair]
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "prometheus": oc.prom_up(), "phases": []}
    try:
        out["phases"].append(phase("baseline", a.url, mixed[:a.requests], None, a.users))
        slow = phase("slow tool", a.url, manual[:a.requests], {"slow_tool": "manual_search", "slow_s": a.slow_s},
                     a.users)
        slow["alerts"] = check_alerts(a.url, ["DeskSlowAnswers"], a.wait, 30)
        out["phases"].append(slow)
        fail = phase("failing tool", a.url, mixed[:a.requests], {"fail_tool": "order_status"}, a.users)
        fail["alerts"] = check_alerts(a.url, ["DeskHighErrorRate", "DeskToolErrors"], a.wait, 30)
        out["phases"].append(fail)
    finally:
        set_fault(None)
        oc.say("\n[drill] fault file removed: the desk is back to normal")
    slowest = max((r for r in slow["requests"] if r["ok"]), key=lambda r: r["latency_s"], default=None)
    failed = next((r for r in fail["requests"] if not r["ok"]), None)
    out["diagnosis"] = {"slow": diagnose("the slowest answer of phase 2", slowest),
                        "failed": diagnose("a failed request of phase 3", failed)}
    for p in out["phases"]:
        for r in p["requests"]:
            r.pop("reply", None)
    (oc.STATE / "drill.json").write_text(json.dumps(out, indent=1))
    oc.say("[drill] wrote m08/state/drill.json")


if __name__ == "__main__":
    main()
