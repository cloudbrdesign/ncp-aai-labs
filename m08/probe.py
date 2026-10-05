"""Step 7: check the desk from outside, the way a user would, and report uptime against an SLO.

    python m08/probe.py --minutes 10                         # a probe every 10 s through the balancer
    python m08/probe.py --minutes 10 --kill r2 --at 120      # ... and kill replica r2 after 2 minutes
    python m08/probe.py --report                             # the SLO report of the last run again

The health checks of lessons 7.1 and 7.3 ask each replica "are you up?" from inside. A synthetic probe
asks the whole service a real question from outside, through the balancer, every --interval seconds,
and checks the answer: E42 is a known error code, so a good answer names the adapter and the 130 W
one that comes with the dock. A probe is good when the reply is HTTP 200, arrives within
--latency-slo seconds (default 30) and passes that check (M6's keywords_all).

--kill NAME --at SECONDS kills that replica's process group (SIGKILL, as a crash or a lost node) at
that point; m08/fleet_obs.py starts it again 5 seconds later. The balancer's health checks and its
retry should keep users from noticing; the probe shows whether they did.

Every probe line shows the version that answered and the trace ID (the x-version and x-trace-id
headers), so any bad probe leads straight to its trace. Results: m08/state/probe.jsonl and the SLO
report in m08/state/slo_report.json:
  availability   good probes / all probes, against --slo (default 0.99)
  error budget   the bad probes the SLO allows in this window, (1 - SLO) x probes, and how much of it
                 this run used
  latency        p50 / p95 of the probes
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time

import httpx

import obs_common as oc

sys.path.insert(0, str(oc.LABS / "m06"))
import evaluators  # noqa: E402

QUESTION = "What does error E42 mean?"
KEYWORDS = ["adapter", "130"]
PROBES = oc.STATE / "probe.jsonl"
REPORT = oc.STATE / "slo_report.json"


def kill(name: str) -> str:
    fleet = json.loads((oc.STATE / "fleet.json").read_text())
    r = next((x for x in fleet if x["name"] == name), None)
    if not r or not r.get("pid"):
        return f"no PID for {name} in m08/state/fleet.json"
    try:
        os.killpg(r["pid"], signal.SIGKILL)         # the replica runs in its own process group
    except ProcessLookupError:
        return f"{name} (PID {r['pid']}) was not running"
    return f"killed {name} (PID {r['pid']})"


async def probe_once(client: httpx.AsyncClient, url: str, latency_slo: float) -> dict:
    rec = await oc.ask(client, url, QUESTION, headers={"x-request-id": f"probe-{int(time.time())}"})
    k, _ = evaluators.keywords_score(KEYWORDS, rec.get("reply", ""))
    if os.environ.get("M08_FAKE_LLM") == "1":
        k = 1                         # the offline check's scripted model proves plumbing, not answers
    rec["answer_ok"] = rec["ok"] and k == 1
    rec["good"] = rec["answer_ok"] and rec["latency_s"] <= latency_slo
    rec["why"] = ("" if rec["good"] else f"HTTP {rec['status']}" if not rec["ok"] else
                  "wrong answer" if not rec["answer_ok"] else f"slow ({rec['latency_s']:.0f} s)")
    rec.pop("reply", None)
    return rec


async def run(a) -> list[dict]:
    t0, out = time.time(), []
    killed = False
    PROBES.unlink(missing_ok=True)
    async with httpx.AsyncClient(timeout=a.timeout) as client:
        while time.time() - t0 < a.minutes * 60:
            tick = time.time()
            if a.kill and not killed and tick - t0 >= a.at:
                oc.say(f"[probe] +{tick - t0:5.0f} s  {kill(a.kill)}")
                killed = True
            rec = await probe_once(client, a.url, a.latency_slo)
            rec["t"] = round(tick - t0, 1)
            out.append(rec)
            with PROBES.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            oc.say(f"[probe] +{rec['t']:5.0f} s  {'GOOD' if rec['good'] else 'BAD '} {rec['status']:>3} "
                   f"{rec['latency_s']:5.1f} s  {rec.get('replica') or '-':>3} {rec.get('version') or '-':>3}  "
                   f"trace {rec.get('trace_id') or '-'}" + (f"  ({rec['why']})" if rec["why"] else ""))
            await asyncio.sleep(max(0.0, a.interval - (time.time() - tick)))
    return out


def report(recs: list[dict], slo: float, latency_slo: float) -> dict:
    n = len(recs)
    good = sum(r["good"] for r in recs)
    bad = n - good
    allowed = (1 - slo) * n
    lat = [r["latency_s"] for r in recs if r["ok"]]
    rep = {"probes": n, "good": good, "bad": bad, "availability": round(good / n, 4) if n else None,
           "slo": slo, "latency_slo_s": latency_slo, "error_budget_probes": round(allowed, 2),
           "error_budget_used": round(bad / allowed, 2) if allowed else None,
           "p50_s": oc.percentile(lat, 50), "p95_s": oc.percentile(lat, 95),
           "bad_probes": [{"t": r["t"], "why": r["why"], "replica": r.get("replica"), "trace_id": r.get("trace_id")}
                          for r in recs if not r["good"]],
           "replicas": {}}
    for r in recs:
        rep["replicas"][r.get("replica") or "-"] = rep["replicas"].get(r.get("replica") or "-", 0) + 1
    oc.say(f"\n== SLO report: {n} probes, {good} good, {bad} bad")
    oc.say(f"[slo] availability {rep['availability']:.2%} against an SLO of {slo:.1%} "
           f"(good = HTTP 200, the right answer, within {latency_slo:g} s)")
    used = rep["error_budget_used"]
    oc.say(f"[slo] error budget for this window: {allowed:.1f} bad probes allowed; used {bad} "
           f"({used:.0%} of it)" if used is not None else "[slo] error budget: no probes")
    oc.say(f"[slo] latency p50 {rep['p50_s'] or 0:.1f} s, p95 {rep['p95_s'] or 0:.1f} s; answered by "
           + ", ".join(f"{k} {v}" for k, v in sorted(rep["replicas"].items())))
    for b in rep["bad_probes"][:10]:
        oc.say(f"[slo] bad at +{b['t']:.0f} s: {b['why']} (replica {b['replica'] or '-'}, trace {b['trace_id'] or '-'})")
    if len(rep["bad_probes"]) > 10:
        oc.say(f"[slo] ... and {len(rep['bad_probes']) - 10} more in m08/state/slo_report.json")
    REPORT.write_text(json.dumps(rep, indent=1))
    oc.say("[slo] wrote m08/state/slo_report.json")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=oc.BALANCER)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--kill", default="", help="replica to kill during the run, e.g. r2")
    ap.add_argument("--at", type=float, default=120, help="seconds into the run")
    ap.add_argument("--slo", type=float, default=0.99)
    ap.add_argument("--latency-slo", type=float, default=30)
    ap.add_argument("--report", action="store_true", help="only print the report of the last run")
    a = ap.parse_args()
    if a.report:
        recs = [json.loads(x) for x in PROBES.read_text().splitlines()]
    else:
        try:
            httpx.get(a.url + "/health", timeout=3)
        except httpx.HTTPError:
            sys.exit(f"[ERROR] no balancer at {a.url}: start m08/fleet_obs.py and m08/balancer_obs.py first")
        recs = asyncio.run(run(a))
    report(recs, a.slo, a.latency_slo)


if __name__ == "__main__":
    main()
