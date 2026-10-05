# Module 8: Run, monitor and maintain

Version 8 of the course project. The M7 desk fleet becomes observable and maintainable:
Prometheus metrics on every replica and on the load balancer, a Grafana dashboard, a trace of
every request (NeMo Agent Toolkit's Phoenix and file exporters), alert rules, a fault drill
diagnosed from the traces, a benchmark of each release against the one before it (offline and as a
canary on live traffic), a small data flywheel that tests a cheaper model on the logged traffic and
leaves the decision to a person, and a synthetic probe that reports uptime against an SLO.

Free mode only, on a laptop: no GPU, no Docker, no cloud account. Prometheus and Grafana come from
Homebrew, Phoenix from pip; the model tier is your Ollama.

## Install

From the repo root, with the Module 0 venv active and Modules 4-7 installed:

```bash
pip install -r m08/requirements.txt          # prometheus-client, NAT's Phoenix exporter
ollama pull llama3.2:1b                      # the v8 candidate (also used in M6)
ollama pull qwen3:4b                         # the judge (M6)
brew install prometheus grafana              # the metrics store and the dashboards
python3 -m venv ~/phoenix-venv && ~/phoenix-venv/bin/pip install arize-phoenix==20.19.0   # the trace viewer
```

Phoenix gets its own venv because it pins its own versions of the OpenTelemetry packages. NeMo
Agent Toolkit's guide starts Phoenix with Docker; `phoenix serve` from pip runs the same server
without Docker.

Ports: 8100 (balancer), 8101-8103 (replicas), 9101-9103 (replica metrics), 9090 (Prometheus),
3000 (Grafana), 6006 (Phoenix).

| File | What it is |
|---|---|
| `desk_obs.py` | the M6 desk with metrics, spans for every model and tool call, a request log, and fault switches |
| `configs/desk_obs.yml` | NAT workflow for `nat serve`, traces to Phoenix and to a file (`desk_obs_file.yml`: file only) |
| `fleet_obs.py` | step 1: three replicas (M7's Fleet) with metrics ports; `--canary r3=v8`; restarts a replica that dies |
| `balancer_obs.py` | step 2: M7's balancer with request IDs, `/metrics`, and `--canary P` |
| `observability/` | step 3: `prometheus.yml`, `alerts.yml`, the Grafana dashboard and `run_stack.sh` |
| `fault_drill.py` | step 4: a slow tool, then a failing tool; the alerts; the cause found in a trace |
| `bench_release.py` | step 5: v7 vs v8 on a frozen test set with a release gate; then live, as a canary |
| `flywheel.py` | step 6: logged traffic -> datasets -> the candidate model -> a report -> a person decides |
| `probe.py` | step 7: a known question every 10 s from outside; uptime and error budget against an SLO |
| `obs_common.py` | helpers: send questions, read Prometheus, read a trace |
| `check.py` | the lab check (17 checks) |

## Step 1: the fleet, with telemetry

Four terminals, all from the repo root:

```bash
~/phoenix-venv/bin/phoenix serve                                # 1. Phoenix, http://localhost:6006
python m08/fleet_obs.py --canary r3=v8                          # 2. r1, r2 run v7; r3 runs v8
python m08/balancer_obs.py                                      # 3. the balancer, :8100
bash m08/observability/run_stack.sh                             # 4. Prometheus :9090, Grafana :3000
```

What `desk_obs.py` adds to the M6 desk, without changing it:

- **metrics** (`curl -s localhost:9101/metrics | grep desk_`): turns by version, workload and status;
  turn latency (a histogram); model calls by desk node (plan, draft, critique, sql), their latency
  and tokens; tool calls by tool and status. Every series carries a `version` label.
- **spans**: NAT traces the desk's graph nodes; `desk_obs.py` adds `llm.<node>` for every model call
  and `tool.<name>` for every tool call. The exporters in the config send the tree to Phoenix and
  to `m08/state/traces/rN.jsonl`.
- **a request log** (`m08/state/logstore/requests.jsonl`): one line per model call in the Data
  Flywheel Blueprint's shape (`timestamp`, `workload_id`, `client_id`, `request`, `response`), plus
  the request ID and trace ID. `turns.jsonl` has one line per turn.
- **fault switches** in `m08/state/faults.json`, used by the drill.

Versions: v7 is the M7 desk on `llama3.2:3b`; v8 is the release candidate on `llama3.2:1b`
(`fleet_obs.VERSIONS`). `--canary r3=v8` makes r3 a v8 replica.

## Step 2: the balancer, with request IDs and metrics

```bash
curl -si -X POST localhost:8100/v1/chat -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"What does error E42 mean?"}]}' | grep -i '^x-'
```

Every reply carries `x-request-id` (yours, or a new one), `x-replica`, `x-version` and `x-trace-id`.
The trace ID opens the request in Phoenix. `--canary 25` sends a quarter of the requests to the v8
replica and the rest to v7.

## Step 3: dashboards and alerts

`run_stack.sh` starts Prometheus with `observability/prometheus.yml` (it scrapes the balancer and
the three replicas every 5 s and evaluates `alerts.yml`) and Grafana with the "Support desk (M8)"
dashboard already loaded, no login (anonymous admin, for a laptop only). The dashboard has the four
signals from the balancer (traffic, errors, latency, saturation) by version, then what happens
inside the replicas: tokens and model calls per request, tool calls by result, p95 per tool and per
desk node.

The alert rules (`alerts.yml`): `DeskReplicaDown`, `DeskHighErrorRate` (more than 10% of requests
without a 2xx in the last minute), `DeskSlowAnswers` (p95 above 20 s over two minutes) and
`DeskToolErrors` (a hint at the cause). The windows are short so they fire within a minute or two;
see them at http://127.0.0.1:9090/alerts.

## Step 4: the fault drill

```bash
python m08/fault_drill.py
```

Baseline, then a slow `manual_search` (30 s added), then a failing `order_status`. After each fault
the script waits for the alerts, then prints the trace of the slowest request and of a failed one,
with the slow span marked SLOW and the failing one marked ERROR. Results: `m08/state/drill.json`.

## Step 5: benchmark the release against the last one

```bash
python m08/bench_release.py frozen                  # v7 and v8 on the 27 test questions; the gate
python m08/balancer_obs.py --canary 25              # (restart the balancer with a canary split)
python m08/bench_release.py live --minutes 3        # what users got from each version
```

The frozen set is the M6 test split. Results per version are stored in `m08/state/bench/`, so the
next release is compared with the stored results of this one. The gate blocks a release whose pass
rate drops by more than 5 points or whose regressions are significant (M6's exact McNemar test).

## Step 6: the data flywheel

```bash
python m08/flywheel.py run                          # on the v7 traffic logged so far
cat m08/state/flywheel/report.md
python m08/flywheel.py decide --workload draft --decision reject --by "your name" --note "..."
python m08/flywheel.py registry
```

## Step 7: uptime from outside

```bash
python m08/probe.py --minutes 10 --kill r2 --at 120
```

A good probe is HTTP 200, the right answer to "What does error E42 mean?", within 30 s. The SLO report
gives availability against 99%, the error budget of this window and how much the run used. Ten
minutes is about 60 probes, so one bad probe is already more than the budget: an error budget is
meant for a longer window (a month), and short runs only show the mechanism.

## Check

```bash
python m08/check.py                       # with Ollama; Phoenix optional (checked when it runs)
M08_FAKE_LLM=1 python m08/check.py        # offline self-test (CI): scripted model, no Ollama
```

## Tested

| Run | Where | Result |
|---|---|---|
| `M08_FAKE_LLM=1 python m08/check.py` | Linux, Python 3.11, nvidia-nat 1.9.0, nvidia-nat-phoenix 1.9.0, prometheus-client 0.26.0, arize-phoenix 20.19.0 (Phoenix running), promtool 3.6.0 | 16 passed, 0 failed (check 1 is for Ollama only) |

## Runs

On a 16 GB Mac, 5 October 2026, steps 1 to 7 in order (30 min end to end). Versions: Prometheus 3.15.0, Grafana 13.2.3,
arize-phoenix 20.19.0, nvidia-nat 1.9.0, nvidia-nat-phoenix 1.9.0, prometheus-client 0.26.0.
`python m08/check.py`: 17 passed, 0 failed.

| Step | Result |
|---|---|
| 1-3 traffic | 50 requests in 3 min through the balancer, all answered; p50 6.9 s, p95 12.3 s |
| 4 fault drill | baseline p95 7.9 s; slow `manual_search`: p95 39.6 s, `DeskSlowAnswers` fired; failing `order_status`: 4 of 6 answered, `DeskHighErrorRate` (24.5%) and `DeskToolErrors` fired. The slowest trace showed `tool.manual_search` at 30.1 s of 39.5 s; the failed request (HTTP 422) showed the `order_status` error |
| 5 frozen | v7 (llama3.2:3b) 59%, p95 6.1 s; v8 (llama3.2:1b) 48%, p95 5.3 s; 5 regressions, 2 improvements, McNemar p = 0.453 (inconclusive); gate BLOCK (pass rate dropped 11%) |
| 5 live, canary 25% | v7: 47 requests, 0% errors, p95 14.6 s, 37 of 47 answers pass; v8: 16 requests, 0% errors, p95 13.7 s, 11 of 16 pass |
| 6 flywheel | 165 distinct logged requests (plan 38, draft 66, critique 61); the 1B against the 3B's answers: plan 0.00 base / 0.38 with examples, draft 0.76 / 0.64, critique 0.62 / 0.25; suggestion for every workload: keep the production model, awaiting human review |
| 7 probe, 10 min, r2 killed at 120 s | 60 probes, 41 good: availability 68% against 99%, error budget used 3167%; p50 3.5 s, p95 3.8 s. Killing r2 cost no probe (the balancer took it out of rotation and the fleet restarted it). All 19 bad probes were wrong answers from r3, the v8 replica the gate had blocked: the dashboards stayed green (HTTP 200, no alerts), only the probe, which checks the answer, saw it |

The last row is the reason for both checks: metrics count failed requests, not wrong answers, and a
release the gate blocks has to be taken out of the fleet as well.
