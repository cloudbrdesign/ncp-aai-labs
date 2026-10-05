"""Check that the M08 lab works end to end (free mode: everything local, model tier on Ollama).

    python m08/check.py      # Ollama with llama3.2:3b, llama3.2:1b, embeddinggemma and qwen3:4b; ports 8100-8103, 9101-9103 free

1.  Ollama is up and has the four models (the desk, the v8 candidate, embeddings, the judge).
2.  The packages: prometheus-client, NeMo Agent Toolkit's Phoenix exporter (nvidia-nat-phoenix).
3.  The configs: both NAT configs name a file exporter (and desk_obs.yml Phoenix); prometheus.yml
    scrapes the balancer and three replicas and loads alerts.yml; alerts.yml has the four rules;
    `promtool check` passes when promtool is installed (brew install prometheus).
4.  The Grafana dashboard: valid JSON, every panel queries the lab's Prometheus data source, and every
    metric it uses is one the balancer or the desk exports.
5.  fleet_obs starts r1, r2 (v7) and r3 (v8); each replica's /metrics shows desk_info with its version
    and model (8.1).
6.  balancer_obs: a reply carries x-request-id, x-replica, x-version and x-trace-id (8.2, 8.5).
7.  The replica logged the turn: request ID next to trace ID in logstore/turns.jsonl (8.2).
8.  The trace file has that request's spans: the desk nodes, llm.plan and a tool.* span (8.2).
9.  The request log is in the Data Flywheel Blueprint's shape (timestamp, workload_id, client_id,
    request with messages, response with choices and usage) with workloads plan, draft, critique (8.4).
10. Metrics: the balancer counted the requests by replica, version and code; the replicas counted
    turns, model calls by workload, tool calls, and tokens (8.1).
11. Faults: a slow manual_search makes a turn take at least that long and its trace marks the tool span
    SLOW; a failing order_status returns an error, counts desk_tool_calls_total{status="error"}, and the
    trace marks the tool span ERROR with the injected message; the alert conditions are met (8.2).
12. Canary: with --canary 50 the balancer sends requests to both versions (8.3).
13. bench_release frozen (4 questions): results stored per version, a gate verdict; a second run
    reuses the stored v7 results instead of asking again (8.3).
14. bench_release live (20 s of canary traffic): a row per version (8.3).
15. flywheel: a run on the logged v7 traffic writes the report and a registry entry awaiting review;
    `decide` records a person's decision for every workload and the run becomes "reviewed" (8.4).
16. probe (60 s, a probe every 5 s) with r2 killed after 15 s: fleet_obs restarts it (new PID), the
    report has availability, the error budget and the bad probes (8.5).
17. Phoenix (only when it runs at localhost:6006): the support-desk project has spans of this run.

It writes m08/state/ (replicas/, logs/, traces/, logstore/, bench/, flywheel/, registry.json,
probe.jsonl, slo_report.json, check.log).
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import httpx
import yaml

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
sys.path.insert(0, str(HERE))
import fleet_obs  # noqa: E402  (sets M07_FAKE_LLM when M08_FAKE_LLM=1)
import obs_common as oc  # noqa: E402

STATE = HERE / "state"
LOG = STATE / "check.log"
FAKE = os.environ.get("M08_FAKE_LLM") == "1"
OBS = HERE / "observability"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)
    with LOG.open("a") as f:
        f.write(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail else "") + "\n")


def info(text):
    print(f"       [INFO] {text}", flush=True)
    with LOG.open("a") as f:
        f.write(f"       [INFO] {text}\n")


def run(args: list[str], timeout: float = 900, env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable] + args, cwd=LABS, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **(env or {})})
    with (STATE / "logs" / "check_scripts.log").open("a") as f:
        f.write(f"$ python {' '.join(args)}\n{p.stdout}\n{p.stderr[-3000:]}\n")
    return p


def ollama_ready() -> bool:
    if FAKE:
        info("offline self-test: M6's scripted model server stands in for Ollama (M08_FAKE_LLM=1)")
        return True
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = host if host.startswith("http") else "http://" + host
    try:
        names = {m["name"] for m in httpx.get(host.rstrip("/") + "/api/tags", timeout=5).json()["models"]}
    except Exception as e:
        check("Ollama answers", False, f"{type(e).__name__}: {e}. Start Ollama (ollama serve).")
        return False
    need = {"llama3.2:3b", "llama3.2:1b", "embeddinggemma:latest", "qwen3:4b"}
    have = {n if ":" in n else n + ":latest" for n in names}
    check("Ollama has llama3.2:3b, llama3.2:1b, embeddinggemma and qwen3:4b", need <= have,
          f"missing {sorted(need - have)}: ollama pull <name>")
    return need <= have


def static_checks() -> None:
    from importlib.metadata import PackageNotFoundError, version
    vers = {}
    for pkg in ("prometheus-client", "nvidia-nat", "nvidia-nat-phoenix"):
        try:
            vers[pkg] = version(pkg)
        except PackageNotFoundError:
            vers[pkg] = None
    check("packages: " + ", ".join(f"{k} {v or 'MISSING'}" for k, v in vers.items()), all(vers.values()),
          "pip install -r m08/requirements.txt")
    cfg = {n: yaml.safe_load((HERE / "configs" / n).read_text()) for n in ("desk_obs.yml", "desk_obs_file.yml")}
    tr = {n: c["general"]["telemetry"]["tracing"] for n, c in cfg.items()}
    prom = yaml.safe_load((OBS / "prometheus.yml").read_text())
    targets = sorted(t for j in prom["scrape_configs"] for sc in j["static_configs"] for t in sc["targets"])
    rules = [r["alert"] for g in yaml.safe_load((OBS / "alerts.yml").read_text())["groups"] for r in g["rules"]]
    ok = (tr["desk_obs.yml"].get("phoenix", {}).get("_type") == "phoenix" and all(t["file"]["_type"] == "file" for t in tr.values())
          and targets == ["127.0.0.1:8100", "127.0.0.1:9101", "127.0.0.1:9102", "127.0.0.1:9103"]
          and prom["rule_files"] == ["alerts.yml"]
          and set(rules) == {"DeskReplicaDown", "DeskHighErrorRate", "DeskSlowAnswers", "DeskToolErrors"})
    promtool = shutil.which("promtool")
    if promtool:
        p1 = subprocess.run([promtool, "check", "config", str(OBS / "prometheus.yml")], capture_output=True, text=True)
        ok = ok and p1.returncode == 0
        info("promtool check config: " + ("SUCCESS" if p1.returncode == 0 else p1.stdout + p1.stderr))
    else:
        info("promtool not installed (brew install prometheus): config checked as YAML only")
    check(f"configs: NAT exporters {sorted(tr['desk_obs.yml'])}; Prometheus scrapes {len(targets)} targets; "
          f"{len(rules)} alert rules", ok)
    dash = json.loads((OBS / "grafana" / "support_desk.json").read_text())
    exported = {"lb_requests_total", "lb_request_seconds_bucket", "lb_replica_up", "desk_request_seconds_sum",
                "desk_requests_total", "desk_llm_tokens_total", "desk_llm_calls_total", "desk_tool_calls_total",
                "desk_tool_seconds_bucket", "desk_llm_seconds_bucket", "ALERTS"}
    import re
    used = {m for p in dash["panels"] for t in p["targets"] for m in re.findall(r"\b([a-zA-Z_]+_(?:total|bucket|sum|up)|ALERTS)\b", t["expr"])}
    ds_ok = all(p["datasource"]["uid"] == "m08-prometheus" for p in dash["panels"])
    check(f"Grafana dashboard: {len(dash['panels'])} panels on the m08-prometheus data source, metrics {sorted(used)}",
          ds_ok and used <= exported, f"unknown metrics: {sorted(used - exported)}")


def first_spans(trace_id: str) -> list[str]:
    for _ in range(10):
        names = [s["name"] for s in oc.trace(trace_id)]
        if names:
            return names
        time.sleep(1)
    return []


def live_checks(fleet) -> None:
    for r in fleet.replicas:
        m = oc.scrape(r["metrics"])
        info_rows = [dict(ls) for (n, ls), v in m.items() if n == "desk_info" and v == 1]
        r["info"] = info_rows[0] if info_rows else {}
    check("fleet_obs: " + ", ".join(f"{r['name']} {r['info'].get('version')} {r['info'].get('model')}" for r in fleet.replicas),
          [r["info"].get("version") for r in fleet.replicas] == ["v7", "v7", "v8"]
          and fleet.replicas[2]["info"].get("model") == "llama3.2:1b")
    bal = subprocess.Popen([sys.executable, str(HERE / "balancer_obs.py")], cwd=LABS,
                           stdout=(STATE / "logs" / "balancer.log").open("a"), stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                if httpx.get(oc.BALANCER + "/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        r = httpx.post(oc.BALANCER + "/v1/chat", json={"messages": [{"role": "user", "content": "My D300 dock shows E42. What does it mean?"}]},
                       headers={"x-request-id": "check-1"}, timeout=300)
        h = {k: r.headers.get(f"x-{k}", "") for k in ("request-id", "replica", "version", "trace-id")}
        check(f"balancer reply: HTTP {r.status_code}, x-request-id {h['request-id']}, x-replica {h['replica']}, "
              f"x-version {h['version']}, x-trace-id {h['trace-id'][:12]}...",
              r.status_code == 200 and h["request-id"] == "check-1" and len(h["trace-id"]) == 32 and h["version"] in ("v7", "v8"))
        turn = oc.turn_for("check-1") or {}
        check(f"turns.jsonl: request check-1 -> trace {turn.get('trace_id', '?')[:12]}..., {turn.get('workload')}, {turn.get('status')}",
              turn.get("trace_id") == h["trace-id"] and turn.get("status") == "ok")
        names = first_spans(h["trace-id"])
        need = {"plan", "execute", "draft", "llm.plan"}
        check(f"trace file: {len(names)} spans for the request, including {sorted(need & set(names))} and "
              f"{sorted(n for n in set(names) if n.startswith('tool.'))}",
              need <= set(names) and any(n.startswith("tool.") for n in names))
        logs = [json.loads(x) for x in (STATE / "logstore" / "requests.jsonl").read_text().splitlines()]
        last = [x for x in logs if x.get("request_id") == "check-1"]
        shape = all({"timestamp", "workload_id", "client_id", "request", "response"} <= set(x) and x["request"]["messages"]
                    and x["response"]["choices"][0]["message"]["content"] is not None and "usage" in x["response"] for x in last)
        check(f"request log: {len(last)} model calls for check-1, workloads {sorted({x['workload_id'] for x in last})}, "
              "blueprint fields present", shape and {"plan", "draft"} <= {x["workload_id"] for x in last})
        for q in ("Where is order A1003?", "Can I still return order A1002?"):
            httpx.post(oc.BALANCER + "/v1/chat", json={"messages": [{"role": "user", "content": q}]}, timeout=300)
        lb = oc.scrape(oc.BALANCER + "/metrics")
        desk = {}
        for rr in fleet.replicas:
            for k, v in oc.scrape(rr["metrics"]).items():
                desk[k] = desk.get(k, 0) + v
        n_lb = oc.total(lb, "lb_requests_total", code="200")
        check(f"metrics: balancer {n_lb:.0f} x 200; replicas {oc.total(desk, 'desk_requests_total'):.0f} turns, "
              f"{oc.total(desk, 'desk_llm_calls_total'):.0f} model calls, {oc.total(desk, 'desk_tool_calls_total'):.0f} tool calls, "
              f"{oc.total(desk, 'desk_llm_tokens_total'):.0f} tokens",
              n_lb >= 3 and oc.total(desk, "desk_requests_total") >= 3 and oc.total(desk, "desk_llm_calls_total") >= 3
              and oc.total(desk, "desk_tool_calls_total") >= 3)
        fault_checks(fleet)
        canary_and_bench(bal)
    finally:
        bal.terminate()
        bal.wait(timeout=10)


def fault_checks(fleet) -> None:
    import fault_drill
    slow_s = 4.0
    fault_drill.set_fault({"slow_tool": "manual_search", "slow_s": slow_s})
    try:
        r = httpx.post(fleet.replicas[0]["url"] + "/v1/chat", json={"messages": [{"role": "user", "content": "What does error E45 mean?"}]},
                       headers={"x-request-id": "check-slow"}, timeout=300)
        fault_drill.set_fault({"fail_tool": "order_status"})
        f = httpx.post(fleet.replicas[0]["url"] + "/v1/chat", json={"messages": [{"role": "user", "content": "Where is order A1003?"}]},
                       headers={"x-request-id": "check-fail"}, timeout=300)
    finally:
        fault_drill.set_fault(None)
    time.sleep(2)
    slow_turn, fail_turn = oc.turn_for("check-slow") or {}, oc.turn_for("check-fail") or {}
    slow_rows = oc.show_trace(slow_turn.get("trace_id", ""), say=lambda *a: None, slow_s=slow_s - 0.5)
    fail_rows = oc.show_trace(fail_turn.get("trace_id", ""), say=lambda *a: None)
    tool_slow = next((x for x in slow_rows if x["name"] == "tool.manual_search"), {})
    tool_err = next((x for x in fail_rows if x["name"] == "tool.order_status"), {})
    errors = oc.total(oc.scrape(fleet.replicas[0]["metrics"]), "desk_tool_calls_total", status="error", tool="order_status")
    check(f"faults: slow manual_search turn {slow_turn.get('latency_s')} s (tool span {tool_slow.get('duration_s')} s); "
          f"failing order_status -> HTTP {f.status_code}, {errors:.0f} tool error(s), trace span error",
          r.status_code == 200 and (tool_slow.get("duration_s") or 0) >= slow_s and f.status_code >= 400
          and errors >= 1 and "injected fault" in (tool_err.get("error") or ""))


def canary_and_bench(bal) -> None:
    bal.terminate()
    bal.wait(timeout=10)
    can = subprocess.Popen([sys.executable, str(HERE / "balancer_obs.py"), "--canary", "50"], cwd=LABS,
                           stdout=(STATE / "logs" / "balancer.log").open("a"), stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                if httpx.get(oc.BALANCER + "/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        versions = []
        for q in [x["question"] for x in oc.testset(("manual_exact",))][:8]:
            r = httpx.post(oc.BALANCER + "/v1/chat", json={"messages": [{"role": "user", "content": q}]}, timeout=300)
            versions.append(r.headers.get("x-version"))
        check(f"canary 50%: 8 requests went to {sorted(set(versions))} ({versions.count('v8')} to v8)",
              {"v7", "v8"} <= set(versions))
        shutil.rmtree(STATE / "bench", ignore_errors=True)
        p = run([str(HERE / "bench_release.py"), "frozen", "--limit", "4"])
        cmp_path = STATE / "bench" / "compare_v7_v8.json"
        cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else {}
        p2 = run([str(HERE / "bench_release.py"), "frozen", "--limit", "4"])
        reused = "stored results" in p2.stdout and "v7" in p2.stdout
        check(f"bench_release frozen: {cmp.get('questions')} questions, gate {cmp.get('gate')}, "
              f"pass rates {({v: s['pass_rate'] for v, s in cmp.get('stats', {}).items()})}; second run reuses the stored results",
              p.returncode == 0 and cmp.get("gate") in ("PASS", "BLOCK") and reused, p.stderr[-500:])
        p = run([str(HERE / "bench_release.py"), "live", "--minutes", "0.34"])
        lv = json.loads((STATE / "bench" / "live.json").read_text()) if (STATE / "bench" / "live.json").exists() else {}
        check(f"bench_release live: versions {sorted(lv.get('versions', {}))}",
              p.returncode == 0 and lv.get("versions"), p.stderr[-500:])
        flywheel_and_probe()
    finally:
        can.terminate()
        can.wait(timeout=10)


def flywheel_and_probe() -> None:
    (STATE / "registry.json").unlink(missing_ok=True)
    p = run([str(HERE / "flywheel.py"), "run", "--eval-size", "2", "--icl", "1"])
    reg = json.loads((STATE / "registry.json").read_text()) if (STATE / "registry.json").exists() else {}
    runs = reg.get("flywheel_runs", [])
    workloads = sorted(runs[-1]["results"]) if runs else []
    ok = p.returncode == 0 and runs and runs[-1]["status"] == "awaiting human review" and (STATE / "flywheel" / "report.md").exists()
    for w in workloads:
        run([str(HERE / "flywheel.py"), "decide", "--workload", w, "--decision", "reject", "--by", "check.py",
             "--note", "self-test"])
    reg = json.loads((STATE / "registry.json").read_text()) if (STATE / "registry.json").exists() else {}
    check(f"flywheel: workloads {workloads}, release {[(r['version'], r['model']) for r in reg.get('releases', [])]}, "
          f"{len(reg.get('decisions', []))} decisions, run {reg.get('flywheel_runs', [{}])[-1].get('status')}",
          ok and reg.get("flywheel_runs", [{}])[-1].get("status") == "reviewed", p.stderr[-500:])


def probe_check(fleet) -> None:
    before = fleet.get("r2")["proc"].pid
    bal = subprocess.Popen([sys.executable, str(HERE / "balancer_obs.py")], cwd=LABS,
                           stdout=(STATE / "logs" / "balancer.log").open("a"), stderr=subprocess.STDOUT)
    import threading
    stop = threading.Event()

    def supervise():                       # what fleet_obs.main does: restart a replica that died
        while not stop.is_set():
            r2 = fleet.get("r2")
            if r2["proc"].poll() is not None:
                time.sleep(5)
                fleet.restart("r2")
                fleet.wait_healthy(["r2"])
                fleet.write_json()
            time.sleep(1)
    t = threading.Thread(target=supervise, daemon=True)
    t.start()
    try:
        time.sleep(2)
        p = run([str(HERE / "probe.py"), "--minutes", "1", "--interval", "5", "--kill", "r2", "--at", "15",
                 "--latency-slo", "60"], timeout=600)
        rep = json.loads((STATE / "slo_report.json").read_text()) if (STATE / "slo_report.json").exists() else {}
        for _ in range(60):
            if fleet.get("r2")["proc"].pid != before and fleet.get("r2")["proc"].poll() is None:
                break
            time.sleep(1)
        after = fleet.get("r2")["proc"].pid
        check(f"probe: {rep.get('probes')} probes, availability {rep.get('availability')}, error budget used "
              f"{rep.get('error_budget_used')}, p95 {rep.get('p95_s')} s; r2 killed and restarted (PID {before} -> {after})",
              p.returncode == 0 and rep.get("probes", 0) >= 8 and after != before and "killed r2" in p.stdout, p.stderr[-500:])
    finally:
        stop.set()
        bal.terminate()
        bal.wait(timeout=10)


def phoenix_check() -> None:
    try:
        spans = httpx.get("http://localhost:6006/v1/projects/support-desk/spans", params={"limit": 50}, timeout=5).json()
    except Exception:
        info("Phoenix is not running at localhost:6006: no Phoenix check (the file traces were checked above)")
        return
    names = {s["name"] for s in spans.get("data", [])}
    check(f"Phoenix: project support-desk has spans ({len(names)} names, e.g. {sorted(n for n in names if n.startswith(('tool.', 'llm.')))[:4]})",
          any(n.startswith("tool.") for n in names))


def main():
    shutil.rmtree(STATE, ignore_errors=True)
    (STATE / "logs").mkdir(parents=True)
    LOG.write_text(f"m08 check {time.strftime('%Y-%m-%dT%H:%M:%S')}{' (fake model)' if FAKE else ''}\n")
    for port in (8100, 8101, 8102, 8103, 9101, 9102, 9103):
        if not fleet_obs.serve_fleet.port_free(port):
            check(f"port {port} is free", False, "stop the fleet, balancer or stack you started by hand (Ctrl-C)")
            return finish()
    if not ollama_ready():
        return finish()
    t = time.time()
    static_checks()
    phoenix = None
    try:
        phoenix = fleet_obs.PHOENIX if httpx.get("http://localhost:6006", timeout=2).status_code == 200 else None
    except httpx.HTTPError:
        pass
    info("Phoenix at localhost:6006: " + ("yes, traces go there too" if phoenix else "no, traces to files only"))
    fleet = fleet_obs.ObsFleet(3, {"r3": "v8"}, phoenix)
    try:
        fleet.start()
        live_checks(fleet)
        probe_check(fleet)
        if phoenix:
            phoenix_check()
    finally:
        fleet.stop()
    info(f"total time {time.time() - t:.0f} s")
    return finish()


def finish():
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
