# Module 10: Human-AI interaction and oversight

Version 10 of the course project. The v9 desk gets a refund tool that never runs without a person. A refund
request becomes an approval card: the order's facts, the policy check, the amount the 3B proposes, and three
allowed decisions (approve, edit, reject). The card waits in a LangGraph interrupt on a SQLite-checkpointed
thread, so it survives a restart; the reviewer decides on a web page or in the terminal; the refund runs once,
after the decision, and not at all while the stop switch is on. Around it: a reviewer queue sorted by
criticality with dual approval for large refunds, feedback buttons that feed M6's test set and a reviewed FAQ
fix with a measured before/after, and a "why this answer" record for every reply, joined by one request ID.

Free mode, on a 16 GB Mac: the M9 desk (Ollama `llama3.2:3b` + `embeddinggemma`), no key, no new package.
Nothing in `m01`-`m09` is changed: M10 wraps the v9 desk as M8 and M9 wrapped theirs.

## Install

From the repo root, with the venv from Modules 0-9 active and the Module 9 install done (spaCy model, gpt2-large):

```bash
pip install -r m10/requirements.txt          # the M9 packages + the langchain 1.4.2 pin
python -c "import langchain, langgraph; from langchain.agents.middleware import HumanInTheLoopMiddleware; from langgraph.types import interrupt, Command, GraphOutput; from langgraph.checkpoint.sqlite import SqliteSaver; from importlib.metadata import version as v; print('langchain', v('langchain'), 'langgraph', v('langgraph'), 'checkpoint-sqlite', v('langgraph-checkpoint-sqlite'))"
python m02/ticket_api.py                     # second terminal (port 8765): tickets for the stop switch and expiry
python m09/heuristics_server.py              # third terminal (port 1337): only for layer L4 (the default)
```

The desk keeps its own index, memory, audit log and request log in `m10/state/` (built on the first run, about
a minute); Module 9's results in `m09/state/` are not touched. Everything the lab writes goes to `m10/state/`
(not in Git).

| File | What it is |
|---|---|
| `oversight_desk.py` | the v10 desk: the outer graph `desk_turn -> propose_refund -> review -> execute_refund -> reply`, SqliteSaver, CLI and `--serve` (the page and the API) |
| `refunds.py` | eligibility (M4's 30-day rule in M9's identity scope), the amount, the idempotent `issue_refund()`, the ledger, the stop switch |
| `approvals.py` | the reviewer CLI: `pending`, `show`, `approve`, `edit --amount`, `reject --message`, `expire --older-than`, `log` |
| `web/index.html` | the page: chat with feedback buttons and "Why this answer?", the reviewer queue (plain HTML and JavaScript) |
| `hitl_middleware_demo.py` | side scene: the same approval with LangChain's `HumanInTheLoopMiddleware` on `create_agent` |
| `drill.py` | the oversight drill: decisions, restart, double resume, stop, burst; `state/oversight/report.md` |
| `feedback_loop.py` | `ingest`, `report`, `promote`, `review`, `fix`, `eval`, `compare`: the feedback loop with its measurement |
| `decision_record.py` | `show --request-id` (operator or customer view), `coverage`, `reasoning` |
| `bootstrap.py` | shared start-up: environment pins, `m10/state/`, the desk version, the v9 desk, the audit log's two new fields |
| `data/prices.json` | fictional unit prices for H200, D300, M270 (EUR) |
| `data/feedback_script.jsonl` | 12 feedback events in M6's shape (thumb, reason from M6's list, comment, correction) |
| `data/refund_burst.jsonl` | 12 refund requests: eligible, outside the window, not delivered, someone else's order, one above the dual-approval threshold, a partial and a duplicate |
| `check.py` | the lab check (18 checks) |
| `tests/fake_oai.py` | scripted stand-in for the models (maintainers' offline self-test only) |

## The v10 desk

```bash
python m10/oversight_desk.py --customer "Tom B." --input "I want my money back for order A1002."
```

```
[turn]   thread t-3f2a9c1e, request 8c1f..., desk action answered, status pending
[card]   c-5a6b...: refund EUR 189.00 for A1002 (1 x USB-C dock), policy: delivered 3 days ago, inside the 30-day
         window (27 days left); approvers needed: 1; proposed by the model
[next]   python m10/approvals.py show t-3f2a9c1e
Reply: ... I've asked a colleague to approve a refund of EUR 189.00 for order A1002. You'll see their answer here.
```

The outer graph, and why each piece is where it is:

| Node | What it does | Why there |
|---|---|---|
| `desk_turn` | Module 9's `Desk.respond`: rails, identity scope, audit record, escalation, disclosure; the M4 graph runs inside it unchanged | finished and checkpointed before any card exists, so a resume never re-runs it |
| `propose_refund` | only for M4's RETURN pattern + an order ID. Eligibility is a rule (`refunds.py`); ineligible requests end here. Otherwise the 3B fills `RefundProposal` (order, amount, reason) and the card is built | rules don't spend a reviewer's attention; the model only fills a schema |
| `review` | one `interrupt(payload)` and nothing else; the payload has the middleware's shape: `action_requests` [name, args, description] + `review_configs` [action_name, allowed_decisions, args_schema], plus the card's facts | on resume the node runs again from its first line; nothing else is in it |
| `execute_refund` | `refunds.issue_refund()`: reads the stop switch, writes one ledger row with the UNIQUE key thread + order, refuses an amount above what is left | the side effect after the interrupt, in its own node, idempotent |
| `reply` | the customer's reply from the outcome | |

The reviewer's edited arguments go straight to `execute_refund`: the model is not asked again (a design
choice; with the middleware an edit goes back to the model, which can re-plan). The thread ID is the
conversation; one card per thread: while it waits, a new message gets a holding reply.

## Step 1 (10.1): the page and the card

```bash
python m10/oversight_desk.py --serve --port 8110          # open http://127.0.0.1:8110/
```

Left: the chat (pick the customer, send "I want my money back for A1002"), each reply with Helpful / Not
helpful (reason from M6's list, comment, correction) and "Why this answer?". Right: the reviewer queue,
refreshed every 3 s, most critical first: the order's facts, the policy check, the proposal and who proposed
it, any check that failed (an amount that can't be paid), approvers needed, the waiting time. Approve; edit
the amount (validated in the page and again on the server, before anything resumes); reject with a message
(required: the customer reads it). Decided cards stay on screen with the outcome. Reload the page, or restart
the server: the card is still there, because it lives in the checkpoint.

The same from the terminal:

```bash
python m10/approvals.py pending
python m10/approvals.py show t-3f2a9c1e
python m10/approvals.py edit t-3f2a9c1e --amount 94.50 --reviewer "Ana"
```

## Step 2 (10.4): the oversight drill

```bash
python m10/drill.py --drill all              # about 5 minutes at L4; --layer L1 for rules only (recorded)
cat m10/state/oversight/report.md
```

| Drill | What happens | What must hold |
|---|---|---|
| `decisions` | reject (with a message), approve, edit (Marco's two headsets, full -> half) | reject pays nothing; edit pays the edited amount |
| `restart` | the server is killed (SIGKILL) with a card pending, then restarted | the card is still pending; approving it then pays once |
| `double-resume` | two resumes of one card at the same time (the pending check skipped, as if two clicks raced past it) | one ledger row (the second gets `duplicate`); a third decision is refused (nothing pending) |
| `stop` | two cards, then the stop switch on; one is approved, a new request arrives, the rest expire | 0 refunds while stopped; the customer is told a colleague will follow up (a ticket); stale cards become tickets |
| `burst` | 12 requests from five customers | 8 refused by rule, 4 queued; the EUR 298 refund needs Ana and Ben; the duplicate is refused at execution by the ledger |

The reviewer in the drill is scripted (Ana and Ben, `--pace` seconds per decision), so the waiting times
measure the queue, not a person. report.md: decisions by type, override rate (edited + rejected / reviewed by
a person), waiting time p50/p95, double payments, refunds executed while stopped.

```bash
python m10/refunds.py stop --on --reason "payment provider incident" --by "Ana"     # the switch by hand
python m10/approvals.py expire --older-than 30
python m10/refunds.py stop --off
```

Side scene, the built-in middleware:

```bash
python m10/hitl_middleware_demo.py
```

Five runs of a `create_agent` agent with `HumanInTheLoopMiddleware(interrupt_on={"issue_refund": {...},
"order_status": False})` and `InMemorySaver`: approve, edit (half), reject (the message goes back to the model),
a status question (no interrupt), and `when=` a predicate that interrupts only above EUR 200 (the EUR 189
refund runs with no person: a conditional interrupt is a policy decision). It prints the interrupt payload
(`action_requests`, `review_configs`) and resumes with `Command(resume={"decisions": [...]})`. Whether the 3B
calls the tool is printed, not assumed.

## Step 3 (10.2): feedback that changes the desk, measured

```bash
python m10/feedback_loop.py ingest                 # 12 scripted events (+ whatever you clicked on the page)
python m10/feedback_loop.py report                 # two channels, by reason, and where each goes
python m10/feedback_loop.py promote                # M6's promote: thumbs-down + correction -> candidates
python m10/feedback_loop.py review --reviewer "Ana"            # accept each (y/n); or --accept-all
python m10/feedback_loop.py eval --label before --reps 3
python m10/feedback_loop.py fix --reviewer "Ana"   # reviewed corrections -> FAQ passages, re-index, v10.1
python m10/feedback_loop.py eval --label after --reps 3
python m10/feedback_loop.py compare
python m10/feedback_loop.py fix --revert           # back to v10 (to run the loop again)
```

Two channels in one store (`state/feedback/feedback.jsonl`), never added together: the users' thumbs (M6's
record shape + request ID) and the reviewers' decisions (`reviewer_decision`: approve, edit, reject, proposed vs
final arguments, wait). `report` prints where each kind goes: a wrong fact with a correction to the knowledge
and the test set, a missing citation to the prompt, should_refuse to the rails, reviewer edits to the proposal
prompt, reviewer rejections to the eligibility rules.

`promote` is M6's own code on this store: 9 thumbs-down carry a correction, one is a near-duplicate (skipped),
8 candidates. `review` shows each with the index's best section for its correction; accepted items get
`ref_sections` and `needs_review: false`. `fix` writes the accepted corrections (not the refusal item: a passage
can't teach a refusal) answer-first as `state/feedback/feedback_faq.md`, adds them to the cleaned manuals (one
file per product, so M4's product filter finds them), rebuilds the index and sets the desk version to v10.1.
`eval` runs the desk turn (layer L1 by default, no refund cards) on the reviewed items and the 42-item M6 test
set with M6's deterministic checks; `compare` gives pass rates, paired flips and the exact McNemar p-value
(M6's `compare_configs.mcnemar`). Expect a gain on the handful of reviewed items and INCONCLUSIVE on the full
set: with exact McNemar it takes at least 6 flips in one direction and none back for p < 0.05.

## Step 4 (10.3): why this answer

```bash
python m10/decision_record.py show --request-id <a request ID from the page or the drill>
python m10/decision_record.py show --request-id <...> --view customer
python m10/decision_record.py coverage
python m10/decision_record.py reasoning --request-id <...>      # optional: qwen3:4b with thinking on
```

The operator view joins what the desk already writes under one request ID: the audit record (route, rails
and their decisions, tool calls, config version), M8's request log (each model call: plan, draft, critique,
refund, with prompt and reply), retrieved vs cited chunk IDs, the card, the approval records (reviewer,
decision, proposed vs final amount, wait), the ledger row, both feedback channels. The customer view is the
"Why this answer?" panel: sources, steps in plain words, whether a person approved; no prompts, no rail
internals, no staff names. `coverage` checks every turn's record is complete. `reasoning` shows a reasoning
model's thinking next to the record: the thinking is text the model wrote, the record is what the desk did.

## HTTP

| Endpoint | What |
|---|---|
| `GET /` | the page |
| `POST /v1/chat` | as M9 (`x-customer`, `x-request-id`), plus `x-thread-id` in and `x-thread-id`, `x-pending` (card ID), `x-status` out |
| `GET /v1/threads/{thread}` | the thread's status and latest reply (the page polls it while a card waits) |
| `GET /v1/approvals` | the pending cards (from `get_state(...).tasks[*].interrupts`), most critical first, and the stop switch |
| `POST /v1/approvals/{thread}/decision` | `{"decision", "reviewer", "amount"?, "message"?}`; 409 if nothing is pending, 422 if invalid |
| `POST /v1/feedback` | `{"request_id", "value", "reason", "comment", "correction"}`; 422 outside M6's schema |
| `GET /v1/records/{request_id}?view=customer\|operator` | the decision record |
| `GET`/`POST /v1/stop` | the stop switch |

## Check

```bash
python m10/check.py                       # Ollama with llama3.2:3b and embeddinggemma
M10_FAKE_LLM=1 python m10/check.py        # offline self-test (CI): scripted models
```

`check.py` works in `m10/state/check/` and starts its own ticket API. The desk runs at L1 there, so the
checks that must hold whatever the model says (the card, the interrupt payload, restart, idempotency, the
stop switch, scope, validation, dual approval, the records) are asserted, and what depends on the 3B (its
proposed amount, whether the middleware demo's agent calls the tool, the burst amounts) is reported as INFO.

## Expected runtime and memory

About 25-35 minutes on the Mac for the whole lab (estimate): the check (about 10 minutes), the drill at L4
(about 5), the feedback loop's two eval passes (50 items x 3 reps x 2 at L1, about 15), the middleware demo
and the clicks on the page. Memory: as Module 9 (Ollama with the 3B, en_core_web_lg twice, gpt2-large in the
heuristics server for L4). Peak RSS: to be recorded from the Mac run.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| no card after a refund request | the desk's turn was blocked by a rail (`desk action blocked` in the output) or the message has no order ID / return words; `--layer L1` shows whether the rails were the reason |
| the card says `proposed amount not payable` | the 3B proposed more than can be refunded; edit or reject it (approving it is refused at execution and opens a ticket) |
| `ticket API not reachable` in a ticket field | start `python m02/ticket_api.py` (port 8765, or set `TICKET_API_URL`) |
| Milvus Lite "database is locked" or a hang at start | one m10 process at a time on `m10/state/desk/manuals.db` (the server, the drill, the feedback loop) |
| `NotImplementedError: The SqliteSaver does not support async methods` | something ran the desk graph from inside the outer graph's context; `oversight_desk.run_async` submits it from an empty context (see Notes) |
| `... is not in the subpath of ...ncp-aai-labs` | `M10_STATE_DIR` must be inside the repo (M4's clean/ingest print paths relative to it) |
| a script dies with `OMP: Error #15` | macOS libomp: every m10 script sets `KMP_DUPLICATE_LIB_OK=TRUE` through `bootstrap.py` (as in Module 9) |

## Notes (API details found while building the lab)

Checked in the installed packages (langchain 1.4.2, langgraph 1.2.12, langgraph-checkpoint 4.2.0,
langgraph-checkpoint-sqlite 3.1.1, langchain-ollama 1.1.0, fastapi 0.141.1, nvidia-nat 1.9.0):

- `invoke(..., version="v2")` returns `GraphOutput` with `.value` and `.interrupts` (a tuple of `Interrupt`
  with `.value` and `.id`). `get_state(config).tasks[i].interrupts` holds the same payload while it waits.
- `Command(resume=...)` on a thread that has nothing pending returns the current state and runs nothing;
  the double-resume drill therefore races two resumes while the card is still pending.
- A graph invoked inside a node of another graph inherits the outer graph's config (LangGraph keeps it in a
  context variable, and asyncio copies the context into new tasks). The M4/M5 desk graph then ran as a
  subgraph with the outer `SqliteSaver`, which has no async methods. `run_async` submits the desk's
  coroutine from an empty `contextvars.Context`, so it runs as its own graph.
- `HumanInTheLoopMiddleware(interrupt_on, *, description_prefix="Tool execution requires approval",
  edit_notice=...)`; `InterruptOnConfig` keys `allowed_decisions`, `description`, `args_schema`, `when`
  (a predicate on `ToolCallRequest`, whose `.tool_call["args"]` holds the call). One interrupt per model turn
  with every pending call (`HITLRequest`: `action_requests`, `review_configs`); the resume value is
  `{"decisions": [...]}`, one per action request, in order. An edit runs the tool with the edited call and
  prefixes the tool's result with the edit notice; a reject returns the message to the model as the tool's
  result. A decision type not in `allowed_decisions` raises `ValueError`.
- `nat.data_models.interactive` has `HumanPromptText`, `HumanPromptBinary`, `HumanPromptRadio`,
  `HumanPromptCheckbox`, `HumanPromptDropdown`, `HumanPromptNotification` (taught from the docs, not used).
- Module 9's `audit_log.write()` keeps only the fields in `audit_log.FIELDS`; the desk's route was computed but
  dropped. `bootstrap.py` adds `route` and `approval` to the list for the v10 desk.
- M4's `clean.clean()` empties the cleaned-manuals folder, so `fix` writes the FAQ files after cleaning and
  before `ingest`. The FAQ passages lead with the answer: the desk drafts from the top of a passage.

## What this lab does not show

- NeMo Agent Toolkit's interactive workflows (`prompt_user_input`, WebSocket and HTTP interactive execution)
  and its web UI: taught from the docs (decision 2: our own page, no Node).
- A tracing backend (Phoenix, M8): the decision record is built from the audit log and M8's request log.
- Preference tuning (DPO) or any change to model weights: the fix here is knowledge (a reviewed passage).
- Real payments: the ledger is a SQLite table; prices, orders and customers are made up.
- Reviewer authentication: the reviewer types a name. In production the name comes from the login, and the
  approval would be bound to the exact action (a signed receipt), which the lesson discusses.

## Tested

| Run | Where | Result |
|---|---|---|
| `M10_FAKE_LLM=1 python m10/check.py` | Linux, Python 3.11.15, langchain 1.4.2, langgraph 1.2.12, langgraph-checkpoint-sqlite 3.1.1, nemoguardrails 0.24.1, fastapi 0.141.1 (torch and transformers not installed: L4 not used) | 17 passed, 0 failed (check 1 is for Ollama only) |
| `python m10/check.py` | the 16 GB Mac, see Runs | to be run |

## Runs

To be filled from the Mac run (`scripts/media/m10_capture.sh` in the corpus repo): Python, Ollama and package
versions; the check; the drill report (decisions, override rate, waits, double payments, refunds while stopped);
the burst table; the feedback before/after with McNemar; the middleware demo (did the 3B call the tool); the
coverage number; peak RSS.

The manuals, orders, customers, prices, test questions and feedback are made up for the course.
