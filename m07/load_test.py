"""Step 3: load-test the desk behind the balancer: 1 replica, then 3, at several concurrency levels.

    python m07/load_test.py                                   # replicas 1 and 3, concurrency 1 2 4 8
    python m07/load_test.py --replicas 3 --levels 4 --requests 12
    python m07/load_test.py --url http://127.0.0.1:8100        # a balancer you already started

Closed loop: at concurrency C, C simulated users each send a question, wait for the reply, and
send the next one, until --requests replies (at least 2 per user) have come back. The questions
are the M6 test set's (m06/data/testset.jsonl), in order. Before each run every replica gets one
warm-up request that is not counted.

For each (replicas, concurrency) it appends one row to m07/state/load.csv:
  throughput_rps     replies per second over the level's wall-clock time
  p50_s p95_s p99_s  latency percentiles (nearest rank) of the successful replies
  error_rate         failed requests / all requests
  per_replica        how many replies each replica served (shows the balancer's spread)
and every request to m07/state/load_requests.csv. cost_calc.py reads load.csv.

What to expect on a laptop: the replicas share one model tier (one Ollama). Each desk turn makes
several LLM calls, and that is where the time goes, so a second and third replica barely raise
throughput; latency grows with concurrency because requests queue at the model. More agent
replicas don't make the model faster (lesson 7.3: find the bottleneck before you scale).
"""
import argparse
import asyncio
import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import uuid

import httpx

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = HERE / "state"
LOAD_CSV = STATE / "load.csv"
REQ_CSV = STATE / "load_requests.csv"
BALANCER_PORT = 8100
sys.path.insert(0, str(HERE))
FIELDS = ["run_id", "time", "replicas", "strategy", "concurrency", "requests", "ok", "errors", "error_rate",
          "duration_s", "throughput_rps", "p50_s", "p95_s", "p99_s", "per_replica", "model"]


def questions() -> list[str]:
    rows = [json.loads(x) for x in (LABS / "m06" / "data" / "testset.jsonl").read_text().splitlines() if x.strip()]
    return [r["question"] for r in rows]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of the values at or below it."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def body(q: str) -> dict:
    return {"messages": [{"role": "user", "content": q}]}


async def one(client: httpx.AsyncClient, url: str, q: str, user: int, t0: float) -> dict:
    start = time.time()
    rec = {"user": user, "t_start": round(start - t0, 3), "question": q[:80]}
    try:
        r = await client.post(url + "/v1/chat", json=body(q))
        rec.update(status=r.status_code, replica=r.headers.get("x-replica", "?"),
                   attempts=int(r.headers.get("x-attempts", "1") or 1), ok=r.status_code == 200)
        if rec["ok"]:
            reply = r.json()["choices"][0]["message"]["content"]
            rec["ok"] = bool(reply.strip())
    except httpx.HTTPError as e:
        rec.update(status=0, replica="?", attempts=0, ok=False, error=type(e).__name__)
    end = time.time()
    rec.update(t_end=round(end - t0, 3), latency_s=round(end - start, 3))
    return rec


async def closed_loop(url: str, concurrency: int, n_requests: int | None = None, duration: float | None = None,
                      qs: list[str] | None = None, t0: float | None = None, timeout: float = 600) -> list[dict]:
    """C users, each sending the next question as soon as its reply arrives.

    Stops after n_requests requests have been sent, or (with duration) when that many seconds have passed.
    """
    qs = qs or questions()
    t0 = t0 or time.time()
    counter = {"next": 0}
    out: list[dict] = []

    async def user(u: int, client: httpx.AsyncClient):
        while True:
            if n_requests is not None and counter["next"] >= n_requests:
                return
            if duration is not None and time.time() - t0 >= duration:
                return
            i = counter["next"]
            counter["next"] += 1
            out.append(await one(client, url, qs[i % len(qs)], u, t0))

    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        await asyncio.gather(*(user(u, client) for u in range(concurrency)))
    return out


def summarise(recs: list[dict], duration: float) -> dict:
    ok = [r for r in recs if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    per = {}
    for r in ok:
        per[r["replica"]] = per.get(r["replica"], 0) + 1
    return {"requests": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
            "error_rate": round((len(recs) - len(ok)) / len(recs), 3) if recs else 0.0,
            "duration_s": round(duration, 2), "throughput_rps": round(len(ok) / duration, 4) if duration else 0.0,
            "p50_s": percentile(lat, 50), "p95_s": percentile(lat, 95), "p99_s": percentile(lat, 99),
            "per_replica": json.dumps(dict(sorted(per.items())))}


def start_balancer(urls: list[str], port: int = BALANCER_PORT, extra: list[str] | None = None,
                   log_name: str = "balancer") -> subprocess.Popen:
    STATE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    log = (STATE / "logs" / f"{log_name}.log").open("a")
    proc = subprocess.Popen([sys.executable, str(HERE / "balancer.py"), "--replicas", ",".join(urls),
                             "--port", str(port)] + (extra or []), cwd=LABS, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(url + "/health", timeout=0.5).status_code in (200, 503):
                return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit(f"[ERROR] the balancer did not start; see m07/state/logs/{log_name}.log")


def status(url: str) -> dict:
    return httpx.get(url + "/status", timeout=5).json()


def model_label() -> str:
    if os.environ.get("M07_FAKE_LLM") == "1":
        return "fake (M07_FAKE_LLM=1)"
    return os.environ.get("NIM_MODEL") if os.environ.get("LLM_PROVIDER") == "nim" else os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def append_rows(rows: list[dict], recs: list[dict]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    new = not LOAD_CSV.exists()
    with LOAD_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    keys = ["run_id", "replicas", "concurrency", "user", "t_start", "t_end", "latency_s", "status", "ok", "replica",
            "attempts", "error", "question"]
    new = not REQ_CSV.exists()
    with REQ_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(recs)


def fmt(v) -> str:
    return "-" if v is None else f"{v:.2f}"


async def run_levels(url: str, n_replicas: int, levels: list[int], n_requests: int, strategy: str,
                     run_id: str, say=print) -> list[dict]:
    rows, all_recs = [], []
    for c in levels:
        n = max(n_requests, 2 * c)
        t0 = time.time()
        recs = await closed_loop(url, c, n_requests=n, t0=t0)
        s = summarise(recs, time.time() - t0)
        row = {"run_id": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "replicas": n_replicas,
               "strategy": strategy, "concurrency": c, "model": model_label(), **s}
        rows.append(row)
        for r in recs:
            r.update(run_id=run_id, replicas=n_replicas, concurrency=c)
        all_recs += recs
        say(f"  replicas {n_replicas}  conc {c:>2}  {s['ok']:>3}/{s['requests']:<3} ok  "
            f"{s['throughput_rps']:.3f} req/s  p50 {fmt(s['p50_s'])} s  p95 {fmt(s['p95_s'])} s  "
            f"p99 {fmt(s['p99_s'])} s  served {s['per_replica']}")
    append_rows(rows, all_recs)
    return rows


async def warm_up(url: str, n: int) -> None:
    await closed_loop(url, n, n_requests=n, qs=["Where is order A1003?"])


def run(replica_counts: list[int], levels: list[int], n_requests: int, strategy: str = "round_robin",
        say=print) -> list[dict]:
    """Start max(replica_counts) replicas once; for each count, a balancer in front of the first N."""
    from serve_fleet import Fleet
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    rows = []
    with Fleet(max(replica_counts), say=say) as fleet:
        for n in replica_counts:
            bal = start_balancer(fleet.urls(n), extra=["--strategy", strategy])
            try:
                url = f"http://127.0.0.1:{BALANCER_PORT}"
                asyncio.run(warm_up(url, n))
                say(f"[load] {n} replica(s) behind the balancer ({strategy}), run {run_id}")
                rows += asyncio.run(run_levels(url, n, levels, n_requests, strategy, run_id, say))
            finally:
                bal.terminate()
                bal.wait()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--requests", type=int, default=8, help="requests per level (at least 2 per user)")
    ap.add_argument("--strategy", choices=["round_robin", "least_conn"], default="round_robin")
    ap.add_argument("--url", help="use a balancer that is already running (its replica count is read from /status)")
    a = ap.parse_args()
    if a.url:
        n = len(status(a.url)["replicas"])
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        asyncio.run(run_levels(a.url.rstrip("/"), n, a.levels, a.requests, status(a.url)["strategy"], run_id))
    else:
        run(a.replicas, a.levels, a.requests, a.strategy)
    print(f"[load] rows appended to {LOAD_CSV.relative_to(LABS)}")


if __name__ == "__main__":
    main()
