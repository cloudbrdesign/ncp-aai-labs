"""Step 2: a small load balancer in front of the desk replicas, with health checks and failover.

    python m07/balancer.py --replicas http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103
    curl -s -X POST localhost:8100/v1/chat -H 'content-type: application/json' \
         -d '{"messages":[{"role":"user","content":"Where is order A1003?"}]}' -i | grep -i x-replica
    curl -s localhost:8100/status                       # each replica: up/down, in flight, served, events

It does what a cloud load balancer or a Kubernetes Service does for you, in about 200 lines you
can read:

  routing          round_robin (the next healthy replica in turn) or least_conn (the healthy
                   replica with the fewest requests in flight). Every reply says which replica
                   answered (x-replica) and how many tries it took (x-attempts).
  active checks    every --interval seconds (default 2) GET /health on every replica, 1 s timeout.
                   --fall failed checks in a row (default 2) mark a replica down; --rise good
                   checks in a row (default 2) mark it up again.
  passive checks   a request that cannot reach a replica (connection refused, connection dropped)
                   marks it down at once, without waiting for the next active check.
  retry            such a request is tried once more on another healthy replica (--no-retry turns
                   this off). Only failures to reach or hear back from a replica are retried: a
                   slow reply (timeout) or an error the desk returned is passed on, because the
                   desk may already have done the work (a turn also writes to its memory).
  no replica up    503, so the client knows to back off.

POST requests to any path are forwarded as they are (/v1/chat, /generate, ...). GET /health says
whether at least one replica is up; GET /status shows the table and the up/down events.
"""
import argparse
import asyncio
import contextlib
import itertools
import json
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError,
             httpx.WriteError)


class Pool:
    def __init__(self, urls: list[str], strategy: str = "round_robin", retry: bool = True,
                 interval: float = 2.0, fall: int = 2, rise: int = 2, timeout: float = 300.0):
        self.replicas = [{"url": u.rstrip("/"), "name": f"r{i + 1}", "up": True, "inflight": 0, "served": 0,
                          "failed": 0, "fails": 0, "oks": 0} for i, u in enumerate(urls)]
        self.strategy, self.retry = strategy, retry
        self.interval, self.fall, self.rise, self.timeout = interval, fall, rise, timeout
        self.events: list[dict] = []
        self._rr = itertools.cycle(range(len(self.replicas)))
        self.client: httpx.AsyncClient | None = None
        self.started = time.time()

    def mark(self, r: dict, up: bool, reason: str) -> None:
        if r["up"] != up:
            r["up"] = up
            r["fails"] = r["oks"] = 0
            self.events.append({"time": time.time(), "replica": r["name"], "up": up, "reason": reason})
            print(f"[balancer] {r['name']} {'UP' if up else 'DOWN'} ({reason})", flush=True)

    def pick(self, exclude: set) -> dict | None:
        healthy = [r for r in self.replicas if r["up"] and r["name"] not in exclude]
        if not healthy:
            return None
        if self.strategy == "least_conn":
            return min(healthy, key=lambda r: (r["inflight"], r["served"]))
        for _ in range(len(self.replicas)):            # round robin over the healthy ones
            r = self.replicas[next(self._rr)]
            if r in healthy:
                return r
        return healthy[0]

    async def check(self, r: dict) -> None:
        try:
            resp = await self.client.get(r["url"] + "/health", timeout=1.0)
            ok = resp.status_code == 200
        except httpx.HTTPError:
            ok = False
        if ok:
            r["oks"], r["fails"] = r["oks"] + 1, 0
            if not r["up"] and r["oks"] >= self.rise:
                self.mark(r, True, f"active: {self.rise} good health checks")
        else:
            r["fails"], r["oks"] = r["fails"] + 1, 0
            if r["up"] and r["fails"] >= self.fall:
                self.mark(r, False, f"active: {self.fall} failed health checks")

    async def health_loop(self) -> None:
        while True:
            await asyncio.gather(*(self.check(r) for r in self.replicas))
            await asyncio.sleep(self.interval)

    async def forward(self, request: Request) -> Response:
        body = await request.body()
        headers = {"content-type": request.headers.get("content-type", "application/json")}
        tried: set = set()
        last_error = "no healthy replica"
        for attempt in range(1, 3 if self.retry else 2):
            r = self.pick(tried)
            if r is None:
                break
            tried.add(r["name"])
            r["inflight"] += 1
            try:
                resp = await self.client.post(r["url"] + request.url.path, content=body, headers=headers,
                                              timeout=self.timeout)
            except RETRYABLE as e:              # never reached the replica, or it died mid-request
                r["failed"] += 1
                last_error = f"{r['name']}: {type(e).__name__}"
                self.mark(r, False, f"passive: {type(e).__name__} on a request")
                continue
            except httpx.TimeoutException as e:  # slow, not dead: don't send the same work twice
                r["failed"] += 1
                return JSONResponse({"error": f"{r['name']}: {type(e).__name__}"}, status_code=504,
                                    headers={"x-replica": r["name"], "x-attempts": str(attempt)})
            finally:
                r["inflight"] -= 1
            r["served"] += 1
            return Response(resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"),
                            headers={"x-replica": r["name"], "x-attempts": str(attempt)})
        return JSONResponse({"error": last_error}, status_code=503 if not tried else 502,
                            headers={"x-replica": "-", "x-attempts": str(len(tried))})

    def status(self) -> dict:
        return {"strategy": self.strategy, "retry": self.retry, "interval_s": self.interval, "fall": self.fall,
                "rise": self.rise, "uptime_s": round(time.time() - self.started, 1),
                "replicas": [{k: r[k] for k in ("name", "url", "up", "inflight", "served", "failed")}
                             for r in self.replicas],
                "events": self.events}


def build_app(pool: Pool) -> Starlette:
    async def health(_: Request):
        up = sum(r["up"] for r in pool.replicas)
        return JSONResponse({"status": "healthy" if up else "unhealthy", "replicas_up": up},
                            status_code=200 if up else 503)

    async def status(_: Request):
        return JSONResponse(pool.status())

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        pool.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=200))
        task = asyncio.create_task(pool.health_loop())
        yield
        task.cancel()
        await pool.client.aclose()

    return Starlette(routes=[Route("/health", health), Route("/status", status),
                             Route("/{path:path}", pool.forward, methods=["POST"])], lifespan=lifespan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", required=True, help="comma-separated replica URLs")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--strategy", choices=["round_robin", "least_conn"], default="round_robin")
    ap.add_argument("--no-retry", action="store_true", help="don't retry a failed request on another replica")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between active health checks")
    ap.add_argument("--fall", type=int, default=2, help="failed checks in a row that mark a replica down")
    ap.add_argument("--rise", type=int, default=2, help="good checks in a row that mark it up again")
    a = ap.parse_args()
    pool = Pool(a.replicas.split(","), a.strategy, not a.no_retry, a.interval, a.fall, a.rise)
    print(f"[balancer] :{a.port} -> {len(pool.replicas)} replica(s), {a.strategy}, "
          f"retry {'on' if pool.retry else 'off'}, health checks every {a.interval:g} s", flush=True)
    print(json.dumps([r["url"] for r in pool.replicas]), flush=True)
    uvicorn.run(build_app(pool), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
