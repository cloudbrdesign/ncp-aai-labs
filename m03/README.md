# Module 3: Cognition, planning and memory

Version 3 of the course project. The support desk becomes a LangGraph state graph: it
plans multi-part requests ("Where is A1001, and can I return A1002?") into steps,
remembers the conversation after a restart, remembers each customer's preferences and
past requests across conversations, and checks its own draft before replying. The same
memory then shows up in a NeMo Agent Toolkit agent through a small memory provider.
Runs in free mode (Ollama or the NVIDIA API catalog). No GPU, no AWS.

Why LangGraph for the desk: NAT 1.9.0 has no built-in thread checkpointer or local
memory backend (its memory providers are plugins for external services), so the graph
is LangGraph, and step 8 brings its memory into NAT through NAT's provider interface.

## Install

From the repo root, with the Module 0 venv active:

```bash
pip install -r m03/requirements.txt
pip install -e m01/support_tools -e m02/desk_tools -e m03/desk_memory   # + the desk_memory provider
```

Everything defaults to local Ollama `llama3.2:3b` (Module 0). For NVIDIA's API catalog,
set the variables from the Module 1 README (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`,
`LLM_API_KEY`).

A 3B model is small, so it only gets short jobs with a fixed answer shape: a plan from a
fixed menu, the reply text, and one advisory grade. Order lookups, routing, most checks
and memory writes are plain Python. Every model step has a fallback, and the log says
when it was used (`fallback`, `template`).

The graph keeps its state in `m03/state/` (`threads.db`, `memory.db`; not in Git). To
start from scratch: `rm -rf m03/state`.

| File | What it is |
|---|---|
| `desk_graph.py` | the graph: nodes, edges, checkpointer, store, command line |
| `planner.py`, `critic.py` | the plan schema and its fallback; the draft checks |
| `handlers.py` | the step handlers (M1 orders and policy, M2 ticket API) |
| `long_term.py` | long-term memory: profile, episodes, lessons |
| `llm_calls.py` | the one place that calls the model |
| `prompts/` | the prompts, in files as in M2 |

## 1. Chain-of-thought vs program-aided reasoning

```bash
python m03/reasoning_check.py              # add --verbose to read every reply
```

Five return-window and refund-date questions from the M1 returns policy, each asked three
ways: a direct answer, "think step by step" (chain-of-thought), and program-aided, where
the model only pulls the dates out as JSON and Python counts the days. You see one line
per question and a score per method:

```
Q4 A customer's return arrived at our warehouse on Thursday 2026-09-03. What is the latest date ...
   expected 2026-09-10 | direct 2026-09-08 WRONG | CoT 2026-09-10 ok | PAL 2026-09-10 ok
...
Score: direct 2/5   chain-of-thought 4/5   program-aided 5/5
```

Your scores will differ, and a 3B model's chain-of-thought often reads well and still
miscounts a weekend. Run it twice and compare.

## 2. Planning a multi-part request

```bash
python m03/desk_graph.py --show-plan --input "Where is my order A1001, and can I return A1002?"
```

The `plan` node asks the model for at most 4 steps from a fixed menu (`order_status`,
`return_check`, `open_ticket`, `answer`) as structured output (a Pydantic schema,
temperature 0). The code then checks the plan: every order ID in the request is covered,
none is invented, a damaged item gets `open_ticket`, a return question gets
`return_check`. If the check fails, a regex splitter builds the plan and the log says
`model plan rejected (...); using the regex fallback`.

```
[plan] 2 steps (model): order_status A1001, return_check A1002
{ "steps": [ { "action": "order_status", "order_id": "A1001" }, ... ] }
```

## 3. Executing the plan: a loop and a logic tree

The same run shows one `[execute]` line per step. `execute` runs one step, then a
conditional edge (a routing function) sends the graph back to `execute` until the plan
is done, then on to `draft`. A recursion limit (25 steps) stops a runaway loop. Draw the graph:

```bash
python m03/desk_graph.py --draw          # prints Mermaid and writes m03/graph.mmd
```

Paste `m03/graph.mmd` into [mermaid.live](https://mermaid.live) to see it. Dotted arrows
are conditional edges.

## 4. Short-term memory: threads and checkpoints

```bash
python m03/desk_graph.py --thread t1 --input "Where is order A1001?"
python m03/desk_graph.py --thread t1 --input "And when will it arrive?"   # a new process: still knows A1001
python m03/desk_graph.py --thread t2 --input "And when will it arrive?"   # a new thread starts empty
python m03/desk_graph.py --thread t3 --memory inmemory --input "Where is order A1001?"
python m03/desk_graph.py --thread t3 --memory inmemory --input "And when will it arrive?"   # forgotten
```

The graph is compiled with `SqliteSaver` (`m03/state/threads.db`): after every node the
state is saved under the thread ID, so the second command plans `order_status A1001`
even though the process restarted. A new thread ID starts empty and asks for the order ID.
With `--memory inmemory` (`InMemorySaver`) the checkpoints die with the process.
Before each model call, the conversation is trimmed to the last 6 messages
(`trim_messages`).

## 5. Long-term memory: profile and episodes

```bash
python m03/desk_graph.py --thread t4 --input "Can I return order A1002? Please contact me by email only."
python m03/desk_graph.py --thread t5 --input "Is order A1002 still returnable, and when do I get my refund?"
```

The store (`SqliteStore`, `m03/state/memory.db`) holds, per customer (`--customer`,
default `tom`):

- `("customers", id, "profile")`: semantic memory, one document (contact channel, language)
- `("customers", id, "episodes")`: episodic memory, one document per resolved request
- `("customers", id, "lessons")`: lessons from failed drafts (step 7)

`recall` loads them at the start of every thread; `remember` writes new facts (simple
rules spot "email only") and the episode at the end. In the second command, a new
thread, you should see:

```
[recall] profile: Contact preference: email only, no phone calls.
[recall] most similar past episode (0.11): "Can I return order A1002? Please contact me by email only."
```

The most similar past episode (word overlap, no embedding model) goes into the draft
prompt as a worked example, and the reply follows up by email.

## 6. History, replay, fork, and resume after a failure

```bash
python m03/time_travel.py
```

Part 1 lists the thread's checkpoints (one per step), replays from the one just before
`draft` (only `draft -> critique -> remember` run again), then forks that checkpoint with
`update_state` (the customer now prefers a phone call). The fork is a new branch:

```
[PASS] original checkpoint ...0de7a7fc still has contact=email (update_state branches, it never rolls back)
```

Part 2 starts the M2 ticket API with `--fail-first 1` (on port 8767), so opening the
ticket fails and the run stops in `execute`. Re-invoking the thread with no new input
continues from the last checkpoint: the order lookup and the plan are not repeated.

```
[FAIL] the run stopped in execute: ticket API returned HTTP 503 (simulated outage)
[INFO] last checkpoint: next=execute, plan has 2 steps, 1 done
[INFO] re-invoked the thread: ran execute -> draft -> critique -> remember
[PASS] resumed from the last checkpoint; plan did not run again
```

The same by hand:

```bash
# terminal 1
python m02/ticket_api.py --fail-first 1
# terminal 2
python m03/desk_graph.py --thread t6 --input "My USB-C dock from order A1002 arrived cracked."
python m03/desk_graph.py --thread t6 --resume
```

Every run uses `durability="sync"`: each checkpoint is written before the next step starts.

## 7. Self-critique and bounded lessons

```bash
python m03/desk_graph.py --bad-draft --thread t7 --input "Can I return order A1002?"
```

`critique` checks the draft with rules (every planned order ID is answered, no order ID
outside the evidence, the contact preference is respected) and asks the model for one
advisory groundedness grade, 0 to 2. `--bad-draft` plants order A1009 in the first draft:

```
[critique] groundedness 0/2 (model, advisory): ...
[critique] FAIL unknown_order: A1009 is not in the facts; remove it
[critique] lesson saved (1/3 kept): Only mention order IDs that are in the facts.
[draft] revision 1 (model): ...
[critique] PASS: order IDs answered, none invented, contact preference respected
```

Failed checks go back to `draft` as named feedback, at most 2 times; after that the desk
sends the template reply built from the evidence. Each failure also writes a one-line
lesson to the customer's lessons, which keep only the newest 3.

## 8. The same memory in NeMo Agent Toolkit

```bash
nat run --config_file m03/configs/support_memory.yml \
  --input "Can you give me a call about returning order A1002?"
```

`m03/desk_memory` registers a `desk_memory` memory provider (`register_memory`, a
`MemoryEditor` over `m03/state/memory.db`, keyword search, no embedder).
`configs/support_memory.yml` is the M2 support agent plus the toolkit's `get_memory` and
`add_memory` tools with a fixed `user_id: tom`, so the model never picks whose memory it
reads. Run step 5 first: the agent calls `get_memory`, finds "email only", and offers
email instead of a call.

Optional: the ReWOO agent over the same tools. Compare its `#E1`, `#E2` plan with step 2.
If the 3B model can't write the ReWOO plan format, use NVIDIA mode.

```bash
nat run --config_file m03/configs/support_rewoo.yml \
  --input "Where is my order A1001, and can I return A1002?"
```

## Check

```bash
python m03/check.py
```

It starts its own ticket API for the failure test (port 8767). It checks: the model
answers; the reasoning check scores all three methods; the plan is valid JSON with 1-4
known steps and an invalid plan falls back; the graph draws; a thread survives a restart
and a new thread or `InMemorySaver` starts empty; a new thread recalls the profile and
episodes are written; history has more than one checkpoint and a fork leaves the original
alone; resume after the forced failure doesn't re-run `plan`; `--bad-draft` is caught and
revised; lessons never exceed 3; NAT `get_memory` returns the saved preference.

Small local models don't follow instructions every time. If a check fails, run it again;
if it keeps failing, try NVIDIA mode or a larger local model.

Offline self-test (course maintainers only): `M03_FAKE_LLM=1 python m03/check.py` swaps
the model for the scripted fake in `m03/tests/fake_llm.py`. You never need it.

## Tested

2026-09-29, Python 3.11.15: nvidia-nat 1.9.0, langgraph 1.2.12, langgraph-checkpoint 4.2.0,
langgraph-checkpoint-sqlite 3.1.1, langchain-core 1.6.5, langchain-ollama 1.1.0,
pydantic 2.13.5. Checked with the offline self-test; the run with Ollama `llama3.2:3b`
on a 16 GB Mac comes next.

The orders, customers and tickets are made up for the course.
