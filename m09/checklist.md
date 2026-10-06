# Licence and risk checklist: support desk v9

The desk's compliance record, filled for this lab. Lab facts (what runs where, which version, what the
lab measured) are filled in. A cell marked `TODO(source needed)` waits for a sourced licence fact: fill it
from the licence text itself, not from memory. `check.py` fails if any cell is empty.

Owner of this checklist: the person who runs the desk (the lab learner). Review it with every release
(the audit log's `config_version` changes when `rails/` changes).

## 1. Models and components

| Component | Role in the desk | Where it runs, version | Licence | Obligations for this lab |
|---|---|---|---|---|
| llama3.2:3b | the desk (plan, draft, critique, SQL) and the self-check rails; local L3 safety model | Ollama on the Mac | TODO(source needed): Llama 3.2 licence name and terms (meta-llama-licenses, Llama 3.2 files after the corpus fetch) | TODO(source needed): attribution ("Built with Llama"?), naming of derived models, the Acceptable Use Policy for a support desk |
| llama3.2:1b | the v8 release candidate in M8; not used by the v9 desk | Ollama on the Mac | TODO(source needed): same licence as llama3.2:3b? (meta-llama-licenses, Llama 3.2) | not in production: none while unused |
| embeddinggemma | embeddings for the manual search (M4 index) | Ollama on the Mac | TODO(source needed): Gemma terms of use (gemma-terms, after the corpus fetch) | TODO(source needed): use restrictions and notices in the Gemma terms |
| qwen3:4b | M6's judge in bias_check.py (offline evaluation only) | Ollama on the Mac | TODO(source needed): Qwen3 licence (qwen3-license, after the corpus fetch) | TODO(source needed): notice requirements, if any |
| nvidia/llama-3.1-nemotron-safety-guard-8b-v3 | L3 content safety with `--hosted` | build.nvidia.com (hosted), called with NVIDIA_API_KEY | TODO(source needed): model licence on its model card (NVIDIA Open Model License or other) | TODO(source needed): attribution, and the clause on bypassing guardrails if it is the NVIDIA Open Model License |
| NemoGuard JailbreakDetect | L2 jailbreak detection with `--hosted` | build.nvidia.com (hosted) | TODO(source needed): model licence on its model card | TODO(source needed): terms of the hosted API trial |
| gpt2-large | L2 perplexity heuristics, in-process (loaded by nemoguardrails' heuristics checks) | the Mac, CPU, downloaded once from Hugging Face (about 3 GB) | TODO(source needed): licence on the gpt2-large model card | TODO(source needed): notices, if any |
| en_core_web_lg 3.8.0 | spaCy model for Presidio (masking rails and the audit masker) | the Mac | MIT (package metadata, checked 2026-10-06) | keep the licence notice with any redistribution |
| nemoguardrails 0.24.1 | the rails runtime (LLMRails, Colang 1.0) | the Mac | Apache-2.0 (package metadata) | keep the licence and NOTICE files with any redistribution |
| presidio-analyzer / presidio-anonymizer 2.2.364 | PII detection and masking | the Mac | MIT (package metadata) | keep the licence notice |
| yara-python 4.5.4 | YARA rules for injection detection | the Mac | Apache 2.0 (package metadata) | keep the licence notice |
| nvidia-nat 1.9.0 (NeMo Agent Toolkit) | M8's telemetry around the desk; red-teaming taught from its docs only | the Mac | Apache-2.0 (package metadata) | keep the licence notice |
| Ollama | serves every local model | the Mac | TODO(source needed): Ollama's licence (ollama repo LICENSE) | TODO(source needed): notices, if any |

## 2. EU AI Act

| Question | Answer for the desk | Evidence in this lab |
|---|---|---|
| Role | the shop deploys the desk; the model makers are providers of the models (GPAI providers where that applies) | checklist section 1 |
| Risk tier | a customer-support chatbot: not a prohibited practice and not in the high-risk list; the transparency obligation for systems that interact with people applies (Art. 50(1)) | lesson 9.5 |
| Art. 50(1) disclosure | the first reply of every session says that the customer is talking to an AI system and can ask for a person | guarded_desk.DISCLOSURE; check.py checks the first and second turn |
| Record-keeping (Art. 12, a high-risk duty) | not required for this tier; the desk keeps an audit log anyway (one masked record per turn) | m09/state/audit/audit.jsonl; audit_log.py query |
| Human oversight (Art. 14, a high-risk duty) | not required for this tier; escalation to a person on threats and repeated attacks | escalate.py; drills/escalation.json |
| Deployer log retention (Art. 26, high-risk) | not required for this tier; the lab purges with `audit_log.py purge --older-than DAYS` (choose DAYS from your own policy) | audit/purge.jsonl |
| GPAI duties | on the model providers, not on the shop | lesson 9.5 |
| Dates that apply | see 9.5 (the dates in force after Regulation (EU) 2026/1744, which moved them from the original text) | lesson 9.5 |

## 3. NIST AI RMF (Govern, Map, Measure, Manage)

| Function | What the desk does | Artefact |
|---|---|---|
| Govern | a named owner, a versioned policy (rails/ hashed into every audit record), retention, this checklist reviewed per release | audit record `config_version`; checklist.md; audit/purge.jsonl |
| Map | the threats and the data: 8 attack categories, who can see which orders, what leaves the Mac | data/attacks.jsonl; identity scope in guarded_desk.py; section 5 |
| Measure | attack success per category and layer, false positives on normal traffic, added latency, model calls; paired-name check; PII threshold sweep | state/redteam/report.md; state/bias/; state/redteam/pii_sweep.json |
| Manage | layered rails L1-L4, fail-closed fallback per rail, escalation and session lock, provider-down drill | rails/layers.py (FAIL_POLICY); escalate.py; state/drills/ |

## 4. GenAI Profile risks

| Risk (NIST GenAI Profile) | Control in the desk | Measured by | Residual risk |
|---|---|---|---|
| Information security (prompt injection, direct and indirect) | L2 heuristics + self check input; L4 YARA + self check output; canary regex | redteam.py categories direct, leak, indirect, output_injection | see report.md from the Mac run (indirect injection is the hardest to stop) |
| Data privacy | identity-scoped database connection; Presidio masking in and out; masked audit log | categories data and pii; check.py greps the audit log for raw PII | see report.md from the Mac run |
| Dangerous, violent or hateful content | L3 content safety (local 3B prompt or Nemotron Safety Guard); escalation of threats | category toxic; drills/escalation.json | see report.md from the Mac run |
| Harmful bias and homogenization | paired prompts that differ only in the name, read by a person | bias_check.py; state/bias/summary.json | ten pairs are a smoke test, not a fairness measurement |
| Intellectual property | model and component licences in section 1 | this checklist | open until the TODO licence cells are filled |
| Confabulation | M4's critique and citations (unchanged); self check output | M6/M8 benchmarks | unchanged from v8 |

## 5. Data leaving the Mac

| Mode | What leaves | Where to | Control |
|---|---|---|---|
| Local (default) | nothing at run time; models and the spaCy model are downloaded once | - | none needed at run time |
| `--hosted` input rails | the customer's message after L1 masking (Presidio runs first) | build.nvidia.com: NemoGuard JailbreakDetect and Nemotron Safety Guard 8B v3 | masking order in rails/config.yml; NVIDIA_API_KEY only in the environment |
| `--hosted` output rails | the masked message and the desk's reply before output masking (masking runs last), so a leaked name would be sent | build.nvidia.com: Nemotron Safety Guard 8B v3 | identity scope keeps other customers' data out of the reply; output regex runs first |
| `--gliner` (optional) | the unmasked message and reply (GLiNER is the masker) | build.nvidia.com: GLiNER-PII | not used in the lab runs |
