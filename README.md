# NCP-AAI course labs

Hands-on labs for the free **NVIDIA-Certified Professional: Agentic AI (NCP-AAI)** exam-prep
course by Cloud Brewery Academy. Videos are free on YouTube; this repo holds the code.

Independent course. Not affiliated with or endorsed by NVIDIA.

## Two ways to run the labs

| Mode | Runs on | What you need |
|---|---|---|
| Free | Your laptop | Python 3.10+, and either Ollama (free, local, no key) or an NVIDIA API key from build.nvidia.com |
| AWS GPU | One `g6e.xlarge` in `us-east-1` | An AWS account, GPU quota, a budget alarm |

Start with **[setup/](setup/README.md)** (Module 0). Each later module lives on its own
branch (`m01-react` ... `m10-hitl`), so you can join at any module.

## When something doesn't work

Run the module's offline self-test first: `M04_FAKE_LLM=1 python m04/check.py` (any module
from 3 on: `MNN_FAKE_LLM=1 python mNN/check.py`). It replaces the model with a scripted
stand-in, so it needs no Ollama, GPU or key. A pass means the code, packages and data are
wired up correctly; it says nothing about how good a real model's answers are.

## Technical review

The labs for Modules 4 to 6 (retrieval, tools, guardrails, evaluation) were reviewed by
[Lekha Priyadarshini Bhan](https://www.linkedin.com/in/lekhapriya/) in October 2026, at commit
`2cba243`. Her findings and what changed:

| Finding | Change |
|---|---|
| The free-SQL validator matched table names with a regex; quoted names and comma joins got past it | SQLite's authorizer enforces the allowed tables; the row cap is structural; `m04/tests/test_sql_guard.py` |
| Module 9's caller-scoped views could be bypassed by reading `main.orders` directly | the authorizer allows the base tables only through the scope's views (same tests) |
| Harmless reads (`replace()`, "update" in a string) were refused; a `LIMIT` in a comment counted as the cap | keywords are matched outside literals and comments; the cap wraps the query |
| M6's pass rate is keyword and refusal checks, and a negated answer can pass it | documented next to the judge's Answer Accuracy (m06 README); "Sorry for the delay" is no longer a refusal |
| The default eval run includes the held-out test split | `compare_configs.py` prints the splits compared and warns when test is included |
| A missing latency counted as 0 s in the Pareto line; identical results weren't a tie | unknown stays unknown; ties are reported |
| precision@k's documented and implemented denominators differed; duplicate IDs undefined | one definition, documented and tested |
| The M4 README said Milvus Lite builds only FLAT | corrected: the pinned versions build HNSW, as `describe_index` shows |
| An unknown order ID led to an unfiltered manual search; a missing grade was logged as "grounded" | the desk asks for the order number; the log says "grounding unverified" |

Errors that remain are ours.

## Cost safety (AWS mode)

- The GPU instance terminates itself 55 minutes after it boots.
- `python setup/aws_lab.py down` deletes everything a lab created. Run it as soon as
  you finish.
- Prices change: check the AWS pricing page for `g6e.xlarge` before you start.
