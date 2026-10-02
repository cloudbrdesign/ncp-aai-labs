"""Check that the M07 lab works end to end (free mode: everything local, model tier on Ollama).

    python m07/check.py        # Ollama running with llama3.2:3b and embeddinggemma; ports 8100-8103 free

1.  Ollama is up and has llama3.2:3b and embeddinggemma (the shared model tier).
2.  serve_fleet starts 3 `nat serve` replicas of the M6 desk; /health says healthy on each (7.1).
3.  One desk turn straight to a replica: /v1/chat returns an OpenAI-style chat completion with a reply.
4.  Each replica has its own state folder (M04_STATE_DIR) with its own copy of the index (7.1).
5.  The balancer, round_robin: 6 requests are spread 2-2-2, and every reply names its replica (7.2).
6.  The balancer, least_conn: 3 requests sent at once go to 3 different replicas.
7.  Active health check: a replica killed with no traffic is marked DOWN by the health checker within
    interval x fall + 3 s, and the next requests avoid it (7.2, 7.5).
8.  It is marked UP again after a restart (rise good checks).
9.  With no replica reachable the balancer answers 503, not a hang.
10. load_test with 1 and 3 replicas writes one row per level to state/load.csv with no errors,
    p50 <= p95 <= p99, and per-replica counts that add up (7.3).
11. The bottleneck: 3 agent replicas vs 1 at the top concurrency; the throughput ratio is printed
    (the replicas share one model tier, so expect well under 3x).
12. Failover drill with retry: a replica killed mid-load is marked down, no request fails, and it
    serves again after its restart (7.5).
13. The same drill without retry: the failures it costs are counted (at least as many as with retry).
14. cost_calc: the sizing arithmetic on planted numbers (N+1, utilisation, cost per 1,000 requests),
    then on the measured load.csv (7.5).
15. The Kubernetes manifests parse; the Service, HPA and PDB point at the Deployment; probes use
    /health on the container's port; no `latest` tags; the NIMService pins its image and asks for a GPU (7.2).
16. The CI workflow exists and runs the module self-tests and kubeconform (7.4).

It writes m07/state/ (replicas/, logs/, load.csv, load_requests.csv, failover.json, cost.json, check.log).
About 7 minutes on a 16 GB Mac (measured 2026-10-02).
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time

import httpx
import yaml

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
sys.path.insert(0, str(HERE))
import cost_calc  # noqa: E402
import failover_drill  # noqa: E402
import load_test  # noqa: E402
import serve_fleet  # noqa: E402
from serve_fleet import Fleet  # noqa: E402

STATE = HERE / "state"
LOG = STATE / "check.log"
BAL = f"http://127.0.0.1:{load_test.BALANCER_PORT}"
FAKE = serve_fleet.fake_mode()
results = []
quiet = lambda *a, **k: None  # noqa: E731


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)
    with LOG.open("a") as f:
        f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail else "") + "\n")


def info(text):
    print(f"       [INFO] {text}", flush=True)
    with LOG.open("a") as f:
        f.write(f"       [INFO] {text}\n")


def ask(url: str, q: str = "Where is order A1003?", timeout: float = 300) -> httpx.Response:
    return httpx.post(url + "/v1/chat", json=load_test.body(q), timeout=timeout)


def wait_event(replica: str, up: bool, timeout: float) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ev = [e for e in load_test.status(BAL)["events"] if e["replica"] == replica and e["up"] == up]
        if ev:
            return ev[-1]
        time.sleep(0.2)
    return None


def ollama_ready() -> bool:
    if FAKE:
        info("offline self-test: M6's scripted model server stands in for Ollama (M07_FAKE_LLM=1)")
        return True
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else "http://" + host
    try:
        names = {m["name"] for m in httpx.get(host.rstrip("/") + "/api/tags", timeout=5).json()["models"]}
    except Exception as e:
        check("Ollama answers", False, f"{type(e).__name__}: {e}. Start Ollama (ollama serve).")
        return False
    need = {"llama3.2:3b", "embeddinggemma:latest"}
    have = {n if ":" in n else n + ":latest" for n in names}
    check(f"Ollama at {host} has llama3.2:3b and embeddinggemma", need <= have,
          f"missing {sorted(need - have)}: ollama pull llama3.2:3b && ollama pull embeddinggemma")
    return need <= have


def fleet_checks() -> None:
    with Fleet(3, say=quiet) as fleet:
        t = {r["name"]: round(time.time() - r["started"], 1) for r in fleet.replicas}
        check(f"3 replicas of the desk under `nat serve` are healthy on :8101-8103 (up within {max(t.values())} s)",
              all(serve_fleet.healthy(u) for u in fleet.urls()))

        r = ask(fleet.urls()[0])
        reply = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
        check(f"/v1/chat on r1: {r.status_code}, {r.json().get('object') if r.status_code == 200 else '-'}, "
              f"reply {reply[:60]!r}", r.status_code == 200 and reply.strip(), r.text[:300])

        dirs = [rep["state_dir"] for rep in fleet.replicas]
        own = all((d / "manuals.db").is_dir() for d in dirs) and len({str(d) for d in dirs}) == 3
        check("each replica has its own state folder with its own copy of the index: "
              + ", ".join(str(d.relative_to(LABS)) for d in dirs), own)

        # 5. round robin
        bal = load_test.start_balancer(fleet.urls(), extra=["--interval", "1"], log_name="balancer_check")
        try:
            got = [ask(BAL).headers.get("x-replica") for _ in range(6)]
            counts = {n: got.count(n) for n in ("r1", "r2", "r3")}
            check(f"round_robin: 6 requests served {counts} (order {' '.join(map(str, got))})",
                  counts == {"r1": 2, "r2": 2, "r3": 2}, got)
        finally:
            bal.terminate()
            bal.wait()

        # 6. least connections
        bal = load_test.start_balancer(fleet.urls(), extra=["--strategy", "least_conn", "--interval", "1"],
                                       log_name="balancer_check")
        try:
            async def three():
                async with httpx.AsyncClient(timeout=300) as c:
                    rs = await asyncio.gather(*(c.post(BAL + "/v1/chat", json=load_test.body(q)) for q in
                                                ["Where is order A1003?", "Where is order A1001?",
                                                 "My D300 dock shows E42. What does it mean?"]))
                return [x.headers.get("x-replica") for x in rs]
            got = asyncio.run(three())
            check(f"least_conn: 3 requests at once went to {sorted(got)}", sorted(got) == ["r1", "r2", "r3"], got)
        finally:
            bal.terminate()
            bal.wait()

        # 7-9. health checks: kill r3 with no traffic, restart it; then nothing up
        bal = load_test.start_balancer(fleet.urls(), extra=["--interval", "1", "--fall", "2", "--rise", "2"],
                                       log_name="balancer_check")
        try:
            time.sleep(2)
            killed = fleet.kill("r3")
            ev = wait_event("r3", False, 1 * 2 + 3)
            after = [ask(BAL).headers.get("x-replica") for _ in range(4)]
            took = ev["time"] - killed if ev else None
            check(f"active health check: r3 killed, marked DOWN after {took:.1f} s ({ev['reason']}); "
                  f"next 4 requests went to {after}" if ev else "active health check: r3 killed but never marked down",
                  ev is not None and ev["reason"].startswith("active") and "r3" not in after, ev)
            restarted = fleet.restart("r3")
            ev_up = wait_event("r3", True, 240)
            check(f"r3 restarted and marked UP {ev_up['time'] - restarted:.1f} s later ({ev_up['reason']})"
                  if ev_up else "r3 restarted but never marked up", ev_up is not None, ev_up)
        finally:
            bal.terminate()
            bal.wait()
        bal = load_test.start_balancer(["http://127.0.0.1:8199"], extra=["--interval", "1"], log_name="balancer_check")
        try:
            time.sleep(0.5)
            r = ask(BAL, timeout=10)
            check(f"no reachable replica: the balancer answers {r.status_code} {r.json().get('error')!r}",
                  r.status_code in (502, 503), r.text)
        finally:
            bal.terminate()
            bal.wait()


def load_checks() -> None:
    levels = [1, 4]
    n = 6 if not FAKE else 8
    rows = load_test.run([1, 3], levels, n, say=quiet)
    for r in rows:
        info(f"replicas {r['replicas']} conc {r['concurrency']}: {r['throughput_rps']:.3f} req/s, "
             f"p50 {load_test.fmt(r['p50_s'])} s, p95 {load_test.fmt(r['p95_s'])} s, served {r['per_replica']}")
    ok = (len(rows) == 4 and all(r["errors"] == 0 and r["p50_s"] <= r["p95_s"] <= r["p99_s"] for r in rows)
          and all(sum(json.loads(r["per_replica"]).values()) == r["ok"] for r in rows)
          and all(len(json.loads(r["per_replica"])) == r["replicas"] for r in rows if r["concurrency"] > 1))
    check(f"load_test: {len(rows)} rows (replicas 1 and 3 x concurrency {levels}) in state/load.csv, no errors, "
          "p50 <= p95 <= p99, per-replica counts add up", ok, rows)
    one = next(r for r in rows if r["replicas"] == 1 and r["concurrency"] == levels[-1])
    three = next(r for r in rows if r["replicas"] == 3 and r["concurrency"] == levels[-1])
    ratio = three["throughput_rps"] / one["throughput_rps"] if one["throughput_rps"] else 0
    check(f"bottleneck: at concurrency {levels[-1]}, 3 replicas give {three['throughput_rps']:.3f} req/s vs "
          f"{one['throughput_rps']:.3f} with 1 ({ratio:.2f}x; one shared model tier)",
          one["throughput_rps"] > 0 and three["throughput_rps"] > 0 and (ratio < 2.0 if FAKE else True))
    if not FAKE and ratio >= 2.0:
        info("more than 2x: your Ollama serves several requests at once (OLLAMA_NUM_PARALLEL); see the README")


def drill_checks() -> None:
    kw = dict(duration=30, kill_at=8, restart_at=12) if FAKE else dict(duration=75, kill_at=15, restart_at=25)
    on = failover_drill.drill(retry=True, say=quiet, **kw)
    failover_drill.show(on, say=info)
    check(f"failover with retry: r2 killed at {on['kill_at_s']} s, marked down after {on['time_to_mark_down_s']} s, "
          f"{on['errors']} of {on['requests']} requests failed, {on['retried']} retried, back up "
          f"{on['time_to_recover_s']} s after the restart",
          on["errors"] == 0 and on["time_to_mark_down_s"] is not None and on["time_to_recover_s"] is not None
          and on.get("after_recovery", {}).get("by_r2", 0) > 0, on)
    off = failover_drill.drill(retry=False, say=quiet, **kw)
    failover_drill.show(off, say=info)
    check(f"failover without retry: {off['errors']} of {off['requests']} requests failed "
          f"(with retry: {on['errors']})", off["errors"] >= on["errors"] and off["time_to_mark_down_s"] is not None, off)


def cost_checks() -> None:
    s = cost_calc.size(cap_rps=0.5, target_rps=3, utilisation=0.8, headroom=1, price=2.0)
    # 3 / (0.5 x 0.8) = 7.5 -> 8 units, +1 = 9; 9 x 2.0 = 18.0/h; 18 / (3 x 3600) x 1000 = 1.666667
    planted = (s["units_for_load"] == 8 and s["units_total"] == 9 and s["cost_per_hour"] == 18.0
               and s["cost_per_month"] == 13140.0 and abs(s["cost_per_1000_requests"] - 1.666667) < 1e-6)
    rows = cost_calc.latest_rows()
    slo = max(float(r["p95_s"]) for rs in rows.values() for r in rs) + 1
    caps = {n: cost_calc.capacity(rs, slo) for n, rs in rows.items()}
    p = subprocess.run([sys.executable, str(HERE / "cost_calc.py"), "--target-rps", "2", "--slo-p95", f"{slo:.1f}",
                        "--price-per-hour", "1"], cwd=LABS, capture_output=True, text=True)
    saved = json.loads((STATE / "cost.json").read_text()) if p.returncode == 0 else {}
    check(f"cost_calc: planted 3 req/s on 0.5 req/s units at 80% -> {s['units_for_load']}+1 = {s['units_total']} units, "
          f"{s['cost_per_hour']}/h, {s['cost_per_1000_requests']:.4f} per 1,000; measured capacity per unit "
          + ", ".join(f"{n} replica(s) {c['throughput_rps']:.3f} req/s" for n, c in caps.items() if c),
          planted and p.returncode == 0 and all(caps.values()) and saved.get("by_replicas"), (s, p.stderr[-300:]))


def manifest_checks() -> None:
    docs = {}
    for f in sorted((HERE / "k8s").glob("*.yaml")):
        for d in yaml.safe_load_all(f.read_text()):
            if d:
                docs[d["kind"]] = d
    dep, svc, hpa, pdb, nim = (docs.get(k) for k in ("Deployment", "Service", "HorizontalPodAutoscaler",
                                                      "PodDisruptionBudget", "NIMService"))
    problems = []
    if not all((dep, svc, hpa, pdb, nim)):
        problems.append(f"kinds found: {sorted(docs)}")
    else:
        labels = dep["spec"]["template"]["metadata"]["labels"]
        sel = dep["spec"]["selector"]["matchLabels"]
        c = dep["spec"]["template"]["spec"]["containers"][0]
        ports = {p["name"]: p["containerPort"] for p in c["ports"]}
        if not all(labels.get(k) == v for k, v in sel.items()):
            problems.append("Deployment selector does not match its pod labels")
        for k, v in svc["spec"]["selector"].items():
            if labels.get(k) != v:
                problems.append("Service selector does not match the pods")
        if svc["spec"]["ports"][0]["targetPort"] not in ports:
            problems.append("Service targetPort is not a named container port")
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            pr = c.get(probe, {}).get("httpGet", {})
            if pr.get("path") != "/health" or pr.get("port") not in ports:
                problems.append(f"{probe} is not GET /health on the container port")
        if hpa["spec"]["scaleTargetRef"]["name"] != dep["metadata"]["name"] or hpa["spec"]["minReplicas"] < 2:
            problems.append("HPA does not target the Deployment with minReplicas >= 2")
        if not all(labels.get(k) == v for k, v in pdb["spec"]["selector"]["matchLabels"].items()):
            problems.append("PDB selector does not match the pods")
        if ":" not in c["image"] or c["image"].endswith(":latest"):
            problems.append("the desk image has no pinned tag")
        img = nim["spec"]["image"]
        if not img.get("tag") or img["tag"] == "latest":
            problems.append("NIMService image tag is not pinned")
        if nim["spec"]["resources"]["limits"].get("nvidia.com/gpu") != 1 or nim["spec"]["expose"]["service"]["port"] != 8000:
            problems.append("NIMService does not ask for one GPU or expose port 8000")
        if "replicas" in nim["spec"] and nim["spec"].get("scale", {}).get("enabled"):
            problems.append("NIMService sets both spec.replicas and spec.scale")
    check(f"k8s manifests: {', '.join(sorted(docs))}; selectors, probes, HPA/PDB targets and pinned images agree",
          not problems, problems)


def ci_check() -> None:
    path = LABS / ".github" / "workflows" / "ci.yml"
    text = path.read_text() if path.exists() else ""
    wf = yaml.safe_load(text) if text else {}
    jobs = (wf or {}).get("jobs", {})
    steps = json.dumps(jobs)
    ok = bool(jobs) and "kubeconform" in steps and "M07_FAKE_LLM=1" in steps and "m06" in steps
    check(f"CI workflow .github/workflows/ci.yml: jobs {sorted(jobs)}; runs the module self-tests and kubeconform", ok)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"m07 check {time.strftime('%Y-%m-%dT%H:%M:%S')}{' (fake model)' if FAKE else ''}\n")
    for port in (8100, 8101, 8102, 8103):
        if not serve_fleet.port_free(port):
            check(f"port {port} is free", False, "stop the replicas or balancer you started by hand (Ctrl-C)")
            return finish()
    if not ollama_ready():
        return finish()
    t = time.time()
    fleet_checks()
    load_checks()
    drill_checks()
    cost_checks()
    manifest_checks()
    ci_check()
    info(f"total time {time.time() - t:.0f} s")
    return finish()


def finish():
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
