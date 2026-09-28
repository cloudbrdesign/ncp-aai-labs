# Module 2: Agent development

Version 2 of the course project. The support desk gets its prompt from files, real
tools behind an MCP server, a ticket API that sometimes fails (and error handling that
copes), streaming answers, a photo question for a vision model, and a test that scores
how well the agent picks its tools. Runs in free mode (Ollama or the NVIDIA API catalog).
No GPU, no AWS.

## Install

From the repo root, with the Module 0 venv active:

```bash
pip install -r m02/requirements.txt
pip install -e m01/support_tools -e m02/desk_tools   # lookup_order (M01) + create_ticket, get_ticket
ollama pull qwen3-vl:8b                               # the vision model for step 3 (a large download)
```

On a Mac with 8 GB of memory, pick a smaller vision model from
[ollama.com/library/qwen3-vl](https://ollama.com/library/qwen3-vl) and set `VLM_MODEL`.

Everything else defaults to local Ollama `llama3.2:3b` (Module 0). For NVIDIA's API
catalog, set the variables from the Module 1 README (`LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL`, `LLM_API_KEY`).

Most steps need the ticket API running in its own terminal:

```bash
python m02/ticket_api.py          # http://localhost:8765, tickets kept in memory
```

## 1. Prompts in files, and a prompt chain

```bash
python m02/triage_chain.py --all
```

Step 1 sorts each message into a category with `prompts/triage/classify.txt`; step 2
answers with that category's template. Edit a file in `m02/prompts/triage/` and run it
again: the behaviour changes, the code doesn't. The agent configs load their rules the
same way: `additional_instructions: file://../prompts/support_rules.md`.

## 2. Tools over MCP

```bash
# terminal 2: publish the desk's tools as an MCP server on http://localhost:9901/mcp
nat mcp serve --config_file m02/configs/ticket_tools.yml \
  --tool_names lookup_order --tool_names create_ticket --tool_names get_ticket
# terminal 3: see what it publishes, then let the agent use it
nat mcp client tool list
nat run --config_file m02/configs/support_mcp.yml \
  --input "My USB-C dock from order A1002 arrived cracked. Please help."
```

The agent has no tools of its own: `mcp_client` discovers them from the server, so they
show up as `desk__create_ticket` and so on.

## 3. A photo question

```bash
python m02/photo_question.py
```

The label photo (`m02/data/shipping_label.jpg`, made up for the course) goes to the vision
model in the same message as the question. Our code then looks the order up.

## 4. When the ticket API fails

```bash
# terminal 1: restart the ticket API so its first 2 requests fail
python m02/ticket_api.py --fail-first 2
# terminal 2:
python m02/breaker_demo.py
```

`configs/support_desk.yml` wraps the ticket tools in the toolkit's `timeout` and
`circuit_breaker` middleware. Two failures trip the breaker; the next call is refused
without touching the API; after the cooldown one probe goes through and the breaker
closes again. The model has its own retries (`num_retries` in the `llms` section), and the
agent passes a failed tool call back to the model so it can tell the customer.

## 5. Streaming

```bash
# terminal 2
nat serve --config_file m02/configs/support_desk.yml --port 8000
# terminal 3
python m02/stream_client.py "What is the status of order A1003?"
curl -N -X POST http://localhost:8000/generate/stream \
  -H 'Content-Type: application/json' -d '{"input_message": "Where is order A1001?"}'
```

The agent's steps arrive first (`intermediate_data:` lines), then the answer, piece by piece
(`data:` lines).

## 6. Does it pick the right tool?

```bash
python m02/decision_check.py        # ticket API running; stop nat serve from step 5 first
```

Ten support questions, each with the tool the agent should use. The script scores a bare
prompt (`prompts/support_rules_v1.md`) against the support-desk rules
(`prompts/support_rules.md`) with the same model and tools. Change the rules, run it
again, and see whether the score moves.

## Check

With the ticket API (terminal 1) and the MCP server (terminal 2) running:

```bash
python m02/check.py
```

Small local models don't follow instructions every time. If a check fails, run it again;
if it keeps failing, try NVIDIA mode or a larger local model.

The orders, tickets and label photo are made up for the course.
