# Module 7: Deployment and scaling

Version 7 of the course project. The M6 support desk becomes a small production-style service:
several replicas of the desk served as APIs with `nat serve`, a load balancer with health checks
and failover, a load test at several concurrency levels with 1 and 3 replicas, a failover drill
that kills a replica under load, a sizing and cost calculator fed by the measured numbers,
Kubernetes manifests for the same service plus a `NIMService` for the model tier, and a GitHub
Actions workflow that runs every module's offline tests and validates the manifests.

Free mode only, on a laptop: no GPU, no Docker, no cloud account. The replicas are local
processes and the model tier is your Ollama. Kubernetes, EKS, the GPU Operator and the NIM
Operator are taught in the videos from the manifests in `k8s/`, which CI validates; this lab does
not apply them to a cluster.

## Install

From the repo root, with the Module 0 venv active and Modules 4-6 installed:

```bash
pip install -r m07/requirements.txt       # the M6 pins (nothing new to download)
# llama3.2:3b (Module 0) and embeddinggemma (Module 4) are already in Ollama
```

Ports 8100 (the balancer) and 8101-8103 (the replicas) must be free. Each replica uses a few
hundred MB of memory on top of Ollama; three fit on a 16 GB Mac.

| File | What it is |
|---|---|
| `configs/desk_serve.yml` | NAT workflow: the M6 desk (`m06/desk_app.py:agent`) for `nat serve` |
| `serve_fleet.py` | step 1: N replicas on ports 8101.., each with its own state folder |
| `balancer.py` | step 2: round robin or least connections, active and passive health checks, one retry |
| `load_test.py` | step 3: closed-loop load at several concurrency levels, 1 and 3 replicas; `state/load.csv` |
| `failover_drill.py` | step 4: kill a replica under load, count what users saw, restart it |
| `cost_calc.py` | step 5: units for a target load and p95 SLO, N+1, cost at a price you type in |
| `k8s/` | step 6: Deployment, Service, HPA, PodDisruptionBudget for the desk; a `NIMService` for the model |
| `../.github/workflows/ci.yml` | step 7: compile, offline self-tests for M3-M7, kubeconform |
| `check.py` | the lab check (16 checks) |

## Step 1: serve the desk as an API, three times

```bash
python m07/serve_fleet.py --replicas 3
```

Each replica is `nat serve --config_file m07/configs/desk_serve.yml --port 810N`: NAT's FastAPI
front end around the M6 desk. In a second terminal:

```bash
curl -s localhost:8101/health                     # {"status":"healthy"}
curl -s -X POST localhost:8101/v1/chat -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Where is order A1003?"}]}'
```

`/v1/chat` takes OpenAI-style messages and returns a `chat.completion`. NAT also serves
`/generate`, `/chat`, the streaming variants and `/v1/chat/completions`; `curl -s
localhost:8101/docs` is the full list.

What the replicas share is the model tier: every replica sends its LLM calls to the same
Ollama, as agent pods share a NIM in a cluster. What they don't share is state: each replica
gets its own folder, `m07/state/replicas/rN` (`M04_STATE_DIR`), with its own copy of the M4
index (Milvus Lite locks its file to one process) and its own `memory.db` and `threads.db`. A
fact one replica remembers is unknown to the others and lost when that replica dies. That is
lesson 7.1's point: keep conversation state in a shared store, or pin each user to a replica.

## Step 2: the load balancer

```bash
python m07/balancer.py --replicas http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103
```

Then send a few requests to port 8100 and watch the `x-replica` header change:

```bash
for i in 1 2 3; do curl -s -o /dev/null -D - -X POST localhost:8100/v1/chat \
  -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Where is order A1003?"}]}' \
  | grep -i x-replica; done
curl -s localhost:8100/status
```

| Feature | What it does |
|---|---|
| `--strategy round_robin` | the next healthy replica in turn (the default) |
| `--strategy least_conn` | the healthy replica with the fewest requests in flight |
| active checks | `GET /health` on every replica every `--interval` s (2); `--fall` failures in a row (2) mark it down, `--rise` successes (2) mark it up |
| passive checks | a request that can't reach a replica, or whose connection drops, marks it down at once |
| retry | that request goes once more to another healthy replica (`--no-retry` turns it off) |
| no replica | 503 (none up) or 502 (every try failed) |

Only failures to reach or hear back from a replica are retried. A slow reply (timeout) or an
error the desk itself returned is passed on: the desk may already have done the work, and a
desk turn writes to its memory, so sending it twice is not free.

Stop the balancer and the fleet with Ctrl-C before step 3; the next scripts start their own.

## Step 3: load test, 1 replica vs 3

```bash
python m07/load_test.py                  # replicas 1 and 3, concurrency 1 2 4 8, 8 requests per level
```

It starts three replicas once; for the 1-replica run the balancer only knows r1. At each
concurrency C, C simulated users send a question, wait for the reply and send the next (a closed
loop), using the M6 test set's questions. One row per level goes to `m07/state/load.csv`
(throughput, p50/p95/p99, error rate, replies per replica) and every request to
`load_requests.csv`. `--url http://127.0.0.1:8100` load-tests a balancer you started yourself.

What to look for: throughput with 3 replicas is about the same as with 1, and latency rises
with concurrency. Each desk turn makes several LLM calls and they all queue at one Ollama. The
agent replicas are not the bottleneck; the model tier is. Adding replicas in front of a
saturated model only adds queueing (lesson 7.3). If 3 replicas do help on your machine, your
Ollama is serving several requests at once (its `OLLAMA_NUM_PARALLEL` setting); the model tier
still caps the total.

## Step 4: the failover drill

```bash
python m07/failover_drill.py               # 3 replicas, 4 users, 60 s; SIGKILL r2 at 15 s, restart at 25 s
python m07/failover_drill.py --no-retry    # the same, with the balancer's retry off
```

It reports the failed requests before the kill, during it (until the balancer marked r2 down)
and after it; how long the balancer took to notice and how (a request, passively, or the health
checker); how many requests were retried; and how long r2 took to come back and take traffic
again. Results go to `m07/state/failover.json`.

With the retry on, the crash should cost no failed requests: the ones r2 was answering are cut
off and sent again to a healthy replica. Without it, every request that was in flight on r2
fails. Detection is near-instant here because a request notices the dead connection; with no
traffic, the health checker needs `--fall` x `--interval` seconds (the check shows both).

## Step 5: size and cost it

```bash
python m07/cost_calc.py --target-rps 2 --slo-p95 30 --price-per-hour 1.00
```

It takes the latest load test for each replica count, finds the highest throughput whose p95
meets your SLO with no errors (one unit's capacity), and computes units = ceil(target /
(capacity x utilisation)), plus `--headroom` spare units (N+1 by default), then the cost per
hour, per month and per 1,000 requests. `--price-per-hour` is what one unit costs you, for
example the hourly on-demand price of the GPU instance that would host the model tier: look it up
on your cloud's pricing page (prices change; the lab states none). The laptop's numbers stand in
for a real unit; the method is the point. If no level meets the SLO, the script says so: more
replicas won't help, a faster model tier will (a GPU, a smaller or quantised model, caching).

## Step 6: the same service on Kubernetes (read, don't apply)

| File | What it is |
|---|---|
| `k8s/desk-deployment.yaml` | the desk pods: rolling update (max 1 extra, none unavailable), startup, readiness and liveness probes on `/health`, spread across zones, `LLM_PROVIDER=nim` pointing at the NIM's Service |
| `k8s/desk-service.yaml` | one stable address that sends traffic only to ready pods (the balancer's job) |
| `k8s/desk-hpa.yaml` | 2 to 6 desk pods on CPU, with a 5-minute scale-down window; its comment says why CPU is a weak signal here |
| `k8s/desk-pdb.yaml` | at most one desk pod down during a node drain |
| `k8s/nimservice.yaml` | the NIM Operator's LLM sample (`llama-3.1-8b-instruct`, image tag `1.3.3`, one GPU, port 8000) with comments |

CI validates the built-in kinds against the Kubernetes 1.33 schemas with kubeconform;
`NIMService` is a custom resource, so check.py checks its fields instead. The desk image tag is
a placeholder: building the image is not part of this lab.

## Step 7: continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to the repo:

- `compile`: every Python file byte-compiles (M1 and M2's checks need live services).
- `selftest`: each module's offline self-test, M3 to M7, against a scripted stand-in for the
  model (`MNN_FAKE_LLM=1`). No Ollama, GPU or key. It proves the plumbing and the APIs, not
  answer quality.
- `manifests`: kubeconform v0.7.0 (checksum pinned) on `m07/k8s`.

Open the repo's Actions tab on GitHub to see the runs; a red cross names the failed step.

## Check

```bash
python m07/check.py
```

16 checks: Ollama has the two models; 3 `nat serve` replicas are healthy; a direct `/v1/chat`
turn; separate state folders; round robin spreads 2-2-2; least connections sends 3 concurrent
requests to 3 replicas; a replica killed with no traffic is marked down by the active health
check and up again after a restart; 502/503 when nothing is reachable; the load test's rows for
1 and 3 replicas at concurrency 1 and 4; the 3-vs-1 throughput ratio; the failover drill with
and without retry; the cost arithmetic on planted numbers and on your load.csv; the manifests
agree with each other; the CI workflow is there. It writes `m07/state/` and logs to
`m07/state/check.log` (replica and balancer logs in `m07/state/logs/`).

Small local models are slow; if a check times out, run it again and read the logs.

Offline self-test (course maintainers only): `M07_FAKE_LLM=1 python m07/check.py` starts M6's
scripted model server (`m06/tests/fake_oai.py`) with one slot and 0.3 s per call, shared by all
replicas, so the model tier is a bottleneck as on a laptop. You never need it.

## Notes (API details found while building the lab)

- NAT 1.9.0's `/health` always answers `{"status": "healthy"}` while the server process runs; it
  does not call the model. It is a liveness signal. A dead model tier shows up as failed
  requests, not as an unhealthy replica, which is why the balancer also checks passively. A
  readiness check that matters would call the model (NIM's own probe is `/v1/health/ready`).
- `/health` stayed fast (about 2 ms) while a replica was busy with 4 desk turns: the M4 graph's
  synchronous steps run in LangGraph's thread pool, not on the server's event loop.
- `/v1/chat` and `/chat` take `{"messages": [...]}`; `/v1/workflow` and `/generate` take the
  `langgraph_wrapper`'s own input schema (`curl localhost:8101/openapi.json`).
- Milvus Lite locks its `.db` folder to one process, so replicas cannot share `m04/state`; each
  gets a copy through `M04_STATE_DIR` (read by `m04/vector_store.py`, which `desk_graph.py` also
  uses for `memory.db` and `threads.db`).
- A replica killed with SIGKILL while answering shows up at the balancer as
  `httpx.RemoteProtocolError` (the connection closed without a response); a stopped one as
  `ConnectError`. Both are retried.

## What this lab does not show

This lab does not show, and the videos don't say, that:

- The manifests were applied to a cluster, or that the desk image exists. They are validated
  against the Kubernetes schemas only.
- Laptop throughput, latency or the 1-vs-3 result predict GPU or NIM serving numbers.
- Any price. The calculator multiplies the price you give it.
- The balancer is production-grade. It shows what cloud load balancers and Kubernetes Services
  do; use those.
- Conversation state survives a replica restart or moves between replicas (it doesn't, here).

## Tested

| Mode | Where | Result |
|---|---|---|
| Offline self-test (`M07_FAKE_LLM=1`) | Linux, Python 3.11.15, 2026-10-02 | 15 passed, 0 failed (check 1 is skipped: no Ollama) |

Packages in the self-test venv: nvidia-nat 1.9.0 (`langchain`, `profiler`), fastapi 0.141.1,
starlette 1.7.0, uvicorn 0.54.0, httpx 0.28.1, PyYAML 6.0.3, langgraph 1.2.12, pymilvus 2.6.9,
milvus-lite 3.2.1; kubeconform 0.7.0 (Kubernetes 1.33.0 schemas: 4 valid, 1 skipped).

## Runs

| Run | Where | Result |
|---|---|---|
| `python m07/check.py` (free mode) | (to fill after the Mac run) | |
