# Module 11: Capstone

The course project is finished. Version 10 of the support desk wraps everything you built: M10's oversight
around M9's safety, around M8's monitoring, M6's evaluation, M5's NIM and NeMo Guardrails, and the M4 graph
(plan, tools, RAG, critique) with M3's memory inside. The capstone adds no feature. It sends the finished desk
four conversations and shows, for each one, every layer it went through and what that layer did.

Free mode, on a 16 GB Mac: the M10 desk (Ollama `llama3.2:3b` + `embeddinggemma`), no key, no new package.
Nothing in `m01`-`m10` is changed.

## Install

With the venv from Modules 0-10 active and the Module 10 install done:

```bash
python m02/ticket_api.py                     # second terminal (port 8765)
python m09/heuristics_server.py              # third terminal (port 1337): the jailbreak heuristics (layer L2+)
```

Everything the capstone writes goes to `m11/state/` (not in Git); Modules 9 and 10 keep their own results.

## Run

```bash
python m11/capstone.py run                   # about 2 minutes; layer L4 (all rails)
cat m11/state/capstone/report.md
```

| Conversation | Customer says | What you should see |
|---|---|---|
| manual | "My D300 dock shows E42. What does it mean?" | the plan routes to `manual_search`, chunks retrieved, the reply cites one of them, the critique scores it |
| order | "Where is my order A1002?" | the `order_status` tool, inside Tom's identity scope (M9) |
| injection | M9's attack `j1` ("Ignore your rules ...") | stopped by the input rails before the model runs; a refusal |
| refund | "I want my money back for order A1002." | an approval card; the script approves as reviewer Ana; one ledger row |

For each conversation the run prints the layers, from the records the desk already writes (M9's audit record,
M8's request log, M10's cards, approvals and ledger, joined by the request ID):

```bash
python m11/capstone.py layers --request-id <a request ID from the run>
python m10/decision_record.py show --request-id <...> --view customer      # the "Why this answer?" panel
```

`--layer L1` runs without the heuristics server: the rules layer alone lets the injection reach the model, and
whether the output rails then catch the reply depends on what the model writes. That is the point of layers.

## The exam domains, in this repo

```bash
python m11/capstone.py map
```

Prints the ten exam domains, the module that covers each, and the files that implement it.

## Check

```bash
python m11/check.py                          # Free mode, layer L4
M11_FAKE_LLM=1 python m11/check.py           # offline self-test (CI): scripted models, layer L1
```

Six checks: the map's files exist; the manual answer cites a retrieved chunk and was critiqued; the order tool
ran in the customer's scope; the injection was blocked (by the input rails at L2 and up); the refund has one
decided card, reviewer Ana and exactly one ledger row; every turn has a complete decision record.
