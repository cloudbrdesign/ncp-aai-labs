"""Step 4: the failover drill. Kill one replica in the middle of a load test and measure what users saw.

    python m07/failover_drill.py                       # 3 replicas, 4 users, 60 s, kill r2 at 15 s, restart at 25 s
    python m07/failover_drill.py --no-retry            # the same drill with the balancer's retry switched off
    python m07/failover_drill.py --no-restart

It starts the fleet and the balancer, runs a closed-loop load (load_test.closed_loop) for
--duration seconds, sends SIGKILL to replica r2 at --kill-at seconds (a crash: no goodbye, the
requests it was answering are cut off), and starts it again at --restart-at seconds.

It reports, from the requests and the balancer's own event log (/status):
  errors before, during and after   "during" = from the kill until the balancer marked r2 down
  time to mark down                 from the kill to the balancer's DOWN event, and whether a
                                    request (passive) or the health checker (active) noticed first
  retried requests                  replies that needed a second try (x-attempts 2)
  time to recover                   from the restart to the balancer's UP event, and r2's share
                                    of the replies after that
Results go to m07/state/failover.json (one entry per run, retry on or off) and the requests to
m07/state/failover_requests.csv.

Run it twice, with and without --no-retry: with the retry the crash should cost no failed
requests (the cut-off ones are sent again to a healthy replica); without it, every request that
was in flight on r2, or sent to it before the balancer noticed, fails.
"""
import argparse
import asyncio
import csv
import json
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_test  # noqa: E402
from serve_fleet import Fleet  # noqa: E402

STATE = HERE / "state"
RESULT = STATE / "failover.json"
REQ_CSV = STATE / "failover_requests.csv"


def drill(replicas: int = 3, users: int = 4, duration: float = 60, kill_at: float = 15,
          restart_at: float | None = 25, retry: bool = True, victim: str = "r2", say=print) -> dict:
    extra = [] if retry else ["--no-retry"]
    with Fleet(replicas, say=say) as fleet:
        bal = load_test.start_balancer(fleet.urls(), extra=extra, log_name="balancer_drill")
        url = f"http://127.0.0.1:{load_test.BALANCER_PORT}"
        try:
            asyncio.run(load_test.warm_up(url, replicas))
            marks: dict = {}

            def chaos(t0: float):
                time.sleep(max(0.0, t0 + kill_at - time.time()))
                marks["kill"] = fleet.kill(victim) - t0
                say(f"[drill] t={marks['kill']:.1f} s  SIGKILL {victim}")
                if restart_at is not None:
                    time.sleep(max(0.0, t0 + restart_at - time.time()))
                    marks["restart"] = fleet.restart(victim) - t0
                    say(f"[drill] t={marks['restart']:.1f} s  restarting {victim}")

            say(f"[drill] {replicas} replicas, {users} users for {duration:g} s, retry {'on' if retry else 'off'}")
            t0 = time.time()
            th = threading.Thread(target=chaos, args=(t0,), daemon=True)
            th.start()
            recs = asyncio.run(load_test.closed_loop(url, users, duration=duration, t0=t0))
            th.join()
            st = load_test.status(url)
        finally:
            bal.terminate()
            bal.wait()
    return report(recs, st, marks, t0, victim, retry, replicas, users, duration)


def report(recs: list[dict], st: dict, marks: dict, t0: float, victim: str, retry: bool,
           replicas: int, users: int, duration: float) -> dict:
    kill = marks["kill"]
    down = next((e for e in st["events"] if e["replica"] == victim and not e["up"]), None)
    up = next((e for e in st["events"] if e["replica"] == victim and e["up"]), None)
    down_at = down["time"] - t0 if down else None
    up_at = up["time"] - t0 if up else None
    edge = down_at if down_at is not None else kill

    def phase(r):
        if r["t_end"] < kill:
            return "before"
        return "during" if r["t_start"] <= edge else "after"

    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "retry": retry, "replicas": replicas, "users": users,
           "duration_s": duration, "victim": victim, "kill_at_s": round(kill, 2),
           "marked_down_at_s": round(down_at, 2) if down_at is not None else None,
           "time_to_mark_down_s": round(down_at - kill, 3) if down_at is not None else None,
           "mark_down_reason": down["reason"] if down else None,
           "restart_at_s": round(marks["restart"], 2) if "restart" in marks else None,
           "marked_up_at_s": round(up_at, 2) if up_at is not None else None,
           "time_to_recover_s": round(up_at - marks["restart"], 2) if up_at is not None and "restart" in marks else None,
           "requests": len(recs), "errors": sum(not r["ok"] for r in recs),
           "retried": sum(r.get("attempts", 1) > 1 for r in recs), "phases": {}}
    for name in ("before", "during", "after"):
        rs = [r for r in recs if phase(r) == name]
        out["phases"][name] = {"requests": len(rs), "errors": sum(not r["ok"] for r in rs),
                               "retried": sum(r.get("attempts", 1) > 1 for r in rs)}
    if up_at is not None:
        later = [r for r in recs if r["ok"] and r["t_start"] > up_at]
        out["after_recovery"] = {"replies": len(later), f"by_{victim}": sum(r["replica"] == victim for r in later)}
    out["errors_by_status"] = {}
    for r in recs:
        if not r["ok"]:
            k = str(r.get("status"))
            out["errors_by_status"][k] = out["errors_by_status"].get(k, 0) + 1
    for r in recs:
        r["phase"], r["retry"] = phase(r), retry
    save(out, recs)
    return out


def save(out: dict, recs: list[dict]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    runs = json.loads(RESULT.read_text()) if RESULT.exists() else []
    runs.append(out)
    RESULT.write_text(json.dumps(runs, indent=1))
    keys = ["retry", "phase", "user", "t_start", "t_end", "latency_s", "status", "ok", "replica", "attempts",
            "error", "question"]
    new = not REQ_CSV.exists()
    with REQ_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(recs)


def show(out: dict, say=print) -> None:
    p = out["phases"]
    say(f"[drill] retry {'on' if out['retry'] else 'off'}: {out['requests']} requests, {out['errors']} failed, "
        f"{out['retried']} retried")
    say(f"        before the kill  {p['before']['requests']:>3} requests  {p['before']['errors']} failed")
    say(f"        during           {p['during']['requests']:>3} requests  {p['during']['errors']} failed  "
        f"({p['during']['retried']} retried)")
    say(f"        after            {p['after']['requests']:>3} requests  {p['after']['errors']} failed")
    if out["time_to_mark_down_s"] is not None:
        say(f"        {out['victim']} marked down {out['time_to_mark_down_s']:.2f} s after the kill "
            f"({out['mark_down_reason']})")
    else:
        say(f"        {out['victim']} was never marked down")
    if out["restart_at_s"] is not None:
        if out["time_to_recover_s"] is not None:
            ar = out.get("after_recovery", {})
            say(f"        back up {out['time_to_recover_s']:.1f} s after the restart; it then served "
                f"{ar.get('by_' + out['victim'], 0)} of {ar.get('replies', 0)} replies")
        else:
            say(f"        not back up before the end of the drill (restart needs longer than --duration allows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--kill-at", type=float, default=15)
    ap.add_argument("--restart-at", type=float, default=25)
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--no-retry", action="store_true")
    a = ap.parse_args()
    out = drill(a.replicas, a.users, a.duration, a.kill_at, None if a.no_restart else a.restart_at, not a.no_retry)
    show(out)
    print(f"[drill] saved to {RESULT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
