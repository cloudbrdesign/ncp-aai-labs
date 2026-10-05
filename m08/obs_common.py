"""Helpers the M8 scripts share: sending questions, reading Prometheus, and reading a trace.

  ask() / closed_loop()   send desk questions through the balancer and keep the reply headers
                          (x-request-id, x-trace-id, x-replica, x-version) with each result
  prom_query() / alerts() Prometheus' HTTP API (/api/v1/query, /api/v1/alerts); None when no
                          Prometheus is running (the offline check has none)
  scrape()                read a /metrics page directly (prometheus-client's text parser)
  trace() / show_trace()  one request's span tree from the file exporter's trace files
                          (m08/state/traces/rN.jsonl): name, duration, and the error if a span failed
"""
import asyncio
import json
import math
import pathlib
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = HERE / "state"
BALANCER = "http://127.0.0.1:8100"
PROMETHEUS = "http://127.0.0.1:9090"
NODES = {"prepare", "recall", "plan", "execute", "draft", "critique", "remember"}   # the M4 desk graph's nodes
NOISE = ("Runnable", "PydanticOutputParser", "LangGraph", "after_")      # framework plumbing, hidden in show_trace


def say(*a):
    print(*a, flush=True)


def testset(categories: tuple[str, ...] | None = None) -> list[dict]:
    rows = [json.loads(x) for x in (LABS / "m06" / "data" / "testset.jsonl").read_text().splitlines() if x.strip()]
    return [r for r in rows if not categories or r["category"] in categories]


def percentile(values: list[float], p: float) -> float | None:
    """Nearest rank, as in m07/load_test.py."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


async def ask(client: httpx.AsyncClient, url: str, question: str, headers: dict | None = None) -> dict:
    start = time.time()
    rec = {"question": question, "t_start": start}
    try:
        r = await client.post(url + "/v1/chat", json={"messages": [{"role": "user", "content": question}]},
                              headers=headers or {})
        rec.update(status=r.status_code, ok=r.status_code == 200, **{
            k: r.headers.get(f"x-{k.replace('_', '-')}", "") for k in ("request_id", "trace_id", "replica", "version")})
        if rec["ok"]:
            rec["reply"] = r.json()["choices"][0]["message"]["content"]
            rec["ok"] = bool(rec["reply"].strip())
        else:
            rec["error"] = r.text[:200]
    except httpx.HTTPError as e:
        rec.update(status=0, ok=False, error=type(e).__name__, request_id="", trace_id="", replica="?", version="?")
    rec["latency_s"] = round(time.time() - start, 3)
    return rec


async def closed_loop(url: str, questions: list[str], concurrency: int = 2, n: int | None = None,
                      duration: float | None = None, timeout: float = 600) -> list[dict]:
    """C users, each sending the next question when its reply arrives (as m07/load_test.py)."""
    n = len(questions) if n is None and duration is None else n
    t0, nxt, out = time.time(), {"i": 0}, []

    async def user(client):
        while (n is None or nxt["i"] < n) and (duration is None or time.time() - t0 < duration):
            q = questions[nxt["i"] % len(questions)]
            nxt["i"] += 1
            out.append(await ask(client, url, q))

    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        await asyncio.gather(*(user(client) for _ in range(concurrency)))
    return out


def summary(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    return {"requests": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
            "error_rate": round((len(recs) - len(ok)) / len(recs), 3) if recs else 0.0,
            "p50_s": percentile(lat, 50), "p95_s": percentile(lat, 95)}


# ---- Prometheus --------------------------------------------------------------------------------

def prom_up(url: str = PROMETHEUS) -> bool:
    try:
        return httpx.get(url + "/-/ready", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def prom_query(expr: str, url: str = PROMETHEUS) -> list[dict] | None:
    try:
        r = httpx.get(url + "/api/v1/query", params={"query": expr}, timeout=5)
        return r.json()["data"]["result"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def alerts(url: str = PROMETHEUS) -> list[dict] | None:
    """Alerts Prometheus is evaluating: [{"name", "state" (pending|firing), "labels", "summary"}]."""
    try:
        data = httpx.get(url + "/api/v1/alerts", timeout=5).json()["data"]["alerts"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    return [{"name": a["labels"].get("alertname"), "state": a["state"],
             "labels": {k: v for k, v in a["labels"].items() if k not in ("alertname", "severity")},
             "summary": a.get("annotations", {}).get("summary", "")} for a in data]


def wait_for_alert(name: str, timeout: float = 90, url: str = PROMETHEUS) -> dict | None:
    """Poll until the named alert fires (None when no Prometheus is running or it never fired)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = alerts(url)
        if found is None:
            return None
        for a in found:
            if a["name"] == name and a["state"] == "firing":
                return a
        time.sleep(3)
    return None


def scrape(url: str) -> dict:
    """{(metric sample name, frozenset(labels)): value} from one /metrics page."""
    text = httpx.get(url, timeout=5).text
    return {(s.name, frozenset(s.labels.items())): s.value
            for fam in text_string_to_metric_families(text) for s in fam.samples}


def total(samples: dict, name: str, **labels) -> float:
    want = set(labels.items())
    return sum(v for (n, ls), v in samples.items() if n == name and want <= set(ls))


# ---- traces ------------------------------------------------------------------------------------

def trace(trace_id: str, folder: pathlib.Path = STATE / "traces") -> list[dict]:
    """The spans of one request: [{"id", "parent", "name", "kind", "start", "end", "output"}].

    Read from the file exporter's files. Requests that ran at the same time share a file, so the
    request's spans are the ones whose parent chain leads to its WORKFLOW_START event."""
    for path in sorted(folder.glob("*.jsonl")):
        text = path.read_text()
        if trace_id not in text:
            continue
        spans: dict[str, dict] = {}
        root = None
        for line in text.splitlines():
            ev = json.loads(line)
            p = ev["payload"]
            kind, uid = p["event_type"], p["UUID"]
            if kind.endswith("_START"):
                spans[uid] = {"id": uid, "parent": ev["parent_id"], "name": p["name"], "kind": kind[:-6],
                              "start": p["event_timestamp"], "end": None, "output": None}
                meta = (p.get("metadata") or {}).get("provided_metadata") or {}
                if kind == "WORKFLOW_START" and meta.get("workflow_trace_id") == trace_id:
                    root = uid
            elif kind.endswith("_END") and uid in spans:
                spans[uid].update(end=p["event_timestamp"], output=(p.get("data") or {}).get("output"))
        if root is None:
            continue
        keep, changed = {root}, True
        while changed:
            changed = False
            for uid, sp in spans.items():
                if uid not in keep and sp["parent"] in keep:
                    keep.add(uid)
                    changed = True
        return [spans[u] for u in keep]
    return []


def show_trace(trace_id: str, say=say, slow_s: float = 3.0) -> list[dict]:
    """Print the span tree; mark the spans that failed (ERROR) or took longer than slow_s (SLOW)."""
    spans = trace(trace_id)
    if not spans:
        say(f"[trace] no spans for {trace_id or '(no trace ID)'} in m08/state/traces/")
        return []
    by_parent: dict[str, list[dict]] = {}
    for s in spans:
        by_parent.setdefault(s["parent"], []).append(s)
    ids = {s["id"] for s in spans}
    roots = [s for s in spans if s["parent"] not in ids]
    t0 = min(s["start"] for s in spans)
    rows = []

    def walk(s, depth):
        hidden = (s["name"].startswith(NOISE) or s["kind"] != "FUNCTION" and s["kind"] != "WORKFLOW"
                  or s["kind"] == "FUNCTION" and s["name"] == "langgraph_wrapper")
        if not hidden:
            dur = (s["end"] - s["start"]) if s["end"] else None
            out = s["output"]
            err = isinstance(out, str) and out.startswith("error:")
            mark = "ERROR " if err or s["end"] is None and s["kind"] != "WORKFLOW" else (
                "SLOW  " if dur is not None and dur >= slow_s else "      ")
            label = s["name"] if s["kind"] in ("FUNCTION", "WORKFLOW") else f"{s['kind'].lower()} {s['name']}"
            took = f"{dur:6.2f} s" if dur is not None else "  (no end)"
            say(f"[trace] {mark}{'  ' * depth}{label:<{max(1, 34 - 2 * depth)}} +{s['start'] - t0:5.2f} s {took}"
                + (f"  {out[7:110]}" if err else ""))
            rows.append({"name": s["name"], "depth": depth, "start_s": round(s["start"] - t0, 3),
                         "duration_s": round(dur, 3) if dur is not None else None, "error": out if err else ""})
        # LangGraph nests the next node under the edge function that chose it (after_execute ->
        # draft); one level up puts it next to the node it follows, as the graph runs them
        child_depth = depth - 1 if s["name"].startswith("after_") else depth if hidden else depth + 1
        for c in sorted(by_parent.get(s["id"], []), key=lambda c: c["start"]):
            # NAT records the next graph node as a child of the one before it; show them side by side
            sibling = s["name"] in NODES and c["name"] in NODES
            walk(c, depth if sibling else max(0, child_depth))

    for r in sorted(roots, key=lambda r: r["start"]):
        walk(r, 0)
    return rows


def turn_for(request_id: str) -> dict | None:
    path = STATE / "logstore" / "turns.jsonl"
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        rec = json.loads(line)
        if rec.get("request_id") == request_id:
            return rec
    return None
