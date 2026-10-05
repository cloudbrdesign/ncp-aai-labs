"""Step 2: M7's load balancer with metrics, request IDs and canary routing.

    python m08/balancer_obs.py                       # the replicas in m08/state/fleet.json, port 8100
    python m08/balancer_obs.py --canary 10           # 10% of requests to the v8 replica(s), 90% to v7
    curl -s localhost:8100/metrics | grep lb_        # Prometheus scrapes this
    curl -si -X POST localhost:8100/v1/chat -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Where is order A1003?"}]}' | grep -i '^x-'

It is m07/balancer.py's Pool (health checks, failover, one retry) with three additions:

  request IDs   every request gets an x-request-id (the client's, or a new one) that goes to the
                replica, which writes it next to its trace ID in m08/state/logstore/turns.jsonl.
                Every reply carries x-request-id, x-replica, x-version and x-trace-id, so any answer
                a user complains about leads to its trace in Phoenix (lesson 8.5).
  metrics       lb_requests_total{replica,version,code}, lb_request_seconds{version} (a histogram)
                and lb_replica_up{replica,version}, on GET /metrics. This is what users see,
                including the requests no replica could answer (replica "-").
  canary        --canary P sends P% of requests to replicas running v8 and the rest to the others,
                so the dashboard can compare the two versions on live traffic (lesson 8.3).
"""
import argparse
import json
import pathlib
import random
import sys
import time
import uuid

import httpx
import uvicorn
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "m07"))
import balancer  # noqa: E402  (m07)

STATE = HERE / "state"
TURNS = STATE / "logstore" / "turns.jsonl"
SECONDS = (0.5, 1, 2, 4, 8, 15, 20, 30, 40, 60, 90, 120)
LB_REQUESTS = Counter("lb_requests", "Requests through the balancer", ["replica", "version", "code"])
LB_SECONDS = Histogram("lb_request_seconds", "Time from request to reply, as the client sees it", ["version"],
                       buckets=SECONDS)
LB_UP = Gauge("lb_replica_up", "1 when the balancer considers the replica up", ["replica", "version"])
CODES = ("200", "422", "500", "502", "504")       # what a reply through the balancer can be
LB_RETRIES = Counter("lb_retries", "Requests retried on another replica")


def trace_id_for(request_id: str) -> str:
    """The replica wrote request_id and trace_id to turns.jsonl just before it replied."""
    try:
        with TURNS.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 262144))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return ""
    for line in reversed(lines):
        if request_id in line:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("request_id") == request_id:
                return rec.get("trace_id", "")
    return ""


class ObsPool(balancer.Pool):
    def __init__(self, replicas: list[dict], canary: float = 0.0, **kw):
        super().__init__([r["url"] for r in replicas], **kw)
        for r, spec in zip(self.replicas, replicas):
            r["name"], r["version"] = spec["name"], spec["version"]
            LB_UP.labels(r["name"], r["version"]).set(1)
            for code in CODES:      # start every series at 0: a series that appears already at 2 would
                LB_REQUESTS.labels(r["name"], r["version"], code)   # hide its first increase from rate()
        for code in ("502", "503"):
            LB_REQUESTS.labels("-", "-", code)
        self.canary = canary

    def mark(self, r: dict, up: bool, reason: str) -> None:
        super().mark(r, up, reason)
        LB_UP.labels(r["name"], r["version"]).set(1 if r["up"] else 0)

    def pick(self, exclude: set) -> dict | None:
        healthy = [r for r in self.replicas if r["up"] and r["name"] not in exclude]
        if self.canary and healthy:
            new = [r for r in healthy if r["version"] == "v8"]
            old = [r for r in healthy if r["version"] != "v8"]
            group = new if (new and (not old or random.random() * 100 < self.canary)) else old
            if self.strategy == "least_conn":
                return min(group, key=lambda r: (r["inflight"], r["served"]))
            return min(group, key=lambda r: r["served"])          # round robin within the group
        return super().pick(exclude)

    async def forward(self, request: Request) -> Response:
        body = await request.body()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        headers = {"content-type": request.headers.get("content-type", "application/json"), "x-request-id": rid}
        tried: set = set()
        last_error = "no healthy replica"
        t = time.perf_counter()
        for attempt in range(1, 3 if self.retry else 2):
            r = self.pick(tried)
            if r is None:
                break
            if attempt > 1:
                LB_RETRIES.inc()
            tried.add(r["name"])
            r["inflight"] += 1
            try:
                resp = await self.client.post(r["url"] + request.url.path, content=body, headers=headers,
                                              timeout=self.timeout)
            except balancer.RETRYABLE as e:
                r["failed"] += 1
                last_error = f"{r['name']}: {type(e).__name__}"
                self.mark(r, False, f"passive: {type(e).__name__} on a request")
                continue
            except httpx.TimeoutException as e:
                r["failed"] += 1
                return self.reply(JSONResponse({"error": f"{r['name']}: {type(e).__name__}"}, status_code=504),
                                  r, attempt, rid, t)
            finally:
                r["inflight"] -= 1
            r["served"] += 1
            out = Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
            return self.reply(out, r, attempt, rid, t)
        out = JSONResponse({"error": last_error}, status_code=503 if not tried else 502)
        return self.reply(out, None, len(tried), rid, t)

    def reply(self, out: Response, r: dict | None, attempts: int, rid: str, t: float) -> Response:
        name, version = (r["name"], r["version"]) if r else ("-", "-")
        LB_REQUESTS.labels(name, version, str(out.status_code)).inc()
        LB_SECONDS.labels(version).observe(time.perf_counter() - t)
        out.headers.update({"x-replica": name, "x-version": version, "x-attempts": str(attempts),
                            "x-request-id": rid, "x-trace-id": trace_id_for(rid) if r else ""})
        return out

    def status(self) -> dict:
        s = super().status()
        for row, r in zip(s["replicas"], self.replicas):
            row["version"] = r["version"]
        s["canary_percent"] = self.canary
        return s


def build_app(pool: ObsPool):
    app = balancer.build_app(pool)

    async def metrics(_: Request):
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.router.routes.insert(0, Route("/metrics", metrics))
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleet", default=str(STATE / "fleet.json"), help="written by fleet_obs.py")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--canary", type=float, default=0.0, help="percent of requests for the v8 replica(s)")
    ap.add_argument("--strategy", choices=["round_robin", "least_conn"], default="round_robin")
    a = ap.parse_args()
    replicas = json.loads(pathlib.Path(a.fleet).read_text())
    pool = ObsPool(replicas, a.canary, strategy=a.strategy)
    print(f"[balancer] :{a.port} -> " + ", ".join(f"{r['name']} ({r['version']})" for r in replicas)
          + (f"; canary: {a.canary:g}% to v8" if a.canary else "") + "; metrics on /metrics", flush=True)
    uvicorn.run(build_app(pool), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
