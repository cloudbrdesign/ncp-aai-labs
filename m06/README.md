# Module 6: Evaluation and tuning

Version 6 of the course project. The M5 support desk gets an evaluation harness: a labelled
test set of 42 questions with dev/test splits and categories; retrieval metrics at several k;
`nat eval` runs with repetitions, two custom evaluators and Ragas' Answer Accuracy scored by a
separate local judge model; Ragas context metrics on the passages the desk really retrieved; a
check of the judge against hand labels (agreement, repeats, swapped positions); configuration
A (`llama3.2:3b`) against B (`llama3.2:1b`) with paired flips and an exact McNemar test; a
feedback log whose thumbs-down answers become new test cases; and a triage report that says
what to fix first.

Free mode only, on a laptop: no GPU, no Docker, no API key. Everything runs on Ollama.

## Install

From the repo root, with the Module 0 venv active:

```bash
pip install -r m06/requirements.txt       # adds nvidia-nat-ragas 1.9.0 (and Ragas 0.4.3) to the M5 pins
pip install --no-deps -e m06              # registers m06/evaluators.py as a NeMo Agent Toolkit plugin
ollama pull llama3.2:1b                   # configuration B
ollama pull qwen3:4b                      # the judge
# llama3.2:3b (Module 0) and embeddinggemma (Module 4) are already there
```

The second `pip` line matters: NAT 1.9.0 finds evaluators only through Python entry points,
so `m06/pyproject.toml` declares `evaluators.py` as one. The editable install maps that single
module to this folder (edits take effect without reinstalling) and adds nothing else to the
path. `nat info components -t evaluator` then lists `keywords_all` and
`refuses_when_unanswerable`.

The judge is `qwen3:4b`, a different model family from the desk. Every judge call asks for
JSON that matches a schema (`response_format: json_schema`), and Ollama constrains the reply to
it. That keeps the reply parseable, and it also keeps qwen3 from thinking out loud: on the Mac
run, asking qwen3:4b for "no thinking" (`reasoning_effort: "none"`) was ignored, and a free-text
reply to "say OK" was 120 tokens of thinking ending in a bare `</think>`, then OK (see Notes).
To use another judge, set `M06_JUDGE_MODEL` for every step. `nemotron-mini` (NVIDIA's 4B model,
4,096 tokens of context) was the fallback, but on the same Mac its free-text reply to "Reply with
the single word OK" was "Sure, that's correct. Is there anything else you need help with?".
`check.py` stops at check 1 unless the desk models reply just "OK" and the judge returns
`{"word": "OK"}`. The desk reads the M4 index (`m04/state/manuals.db`) and
`m04/data/orders.db`; any M6 script builds them if they are missing.

| File | What it is |
|---|---|
| `data/testset.jsonl` | 42 labelled items: M4's 20 retrieval questions, 12 of M5's eval items, 10 new ones |
| `data/judge_labels.jsonl` | 24 replies with a human pass/fail verdict (4 planted cases) and 8 reply pairs |
| `data/feedback_seed.jsonl` | 6 feedback records as if from users (one planted near-duplicate) |
| `testset.py` | step 1: counts per category and split; checks that every label finds its chunks |
| `retrieval_eval.py` | step 2: recall@k, precision@k, hit@k for dense, keyword, hybrid; Ragas ID-based metrics |
| `desk_app.py` | the M5 desk for `nat eval`, plus the passage sidecar |
| `llm_calls.py` | M5's model switch plus `DESK_TEMPERATURE`, `DESK_SEED` and the judge client |
| `evaluators.py`, `pyproject.toml` | two NAT evaluators registered with `register_evaluator` |
| `configs/eval_A.yml` | NAT workflow, dataset with a split filter, profiler, evaluators, judge LLM |
| `ragas_eval.py` | step 4: Ragas metrics on saved runs with the local judge and a disk cache |
| `judge_check.py` | step 5: judge vs hand labels; repeats; pairwise with swapped positions |
| `compare_configs.py` | step 6: A vs B per category, flips, McNemar, Pareto; NeMo Evaluator files |
| `feedback.py` | step 7: `rate`, `report`, `promote` |
| `triage.py` | step 8: failure buckets and "fix first" |
| `check.py` | the lab check (18 checks) |
| `tests/fake_oai.py` | scripted stand-in for Ollama (maintainers' offline self-test only) |

Everything the lab writes goes to `m06/state/` (not in Git), plus `m06/data/testset_feedback.jsonl`
from step 7 (also not in Git until a person has reviewed it).

## 1. The test set

```bash
python m06/testset.py --stats
python m06/testset.py --show --category unanswerable
```

```
category              dev  test   all
order                   1     2     3
manual_exact            4    10    14
manual_paraphrase       4     8    12
mixed                   2     2     4
unanswerable            2     1     3
off_topic               1     2     3
injection               1     2     3
all                    15    27    42
[INFO] sources: m04 20, m05 12, new 10
[INFO] 30 items have reference sections; chunks per label: 1 chunk: 28, 2 chunks: 2
[INFO] every reference section resolves to at least one chunk in the index
```

Each item has an `id`, a `split` (dev: what you tune on and re-run; test: held back for the
final comparison), a `category`, the `question`, a reference `answer`, `keywords` a correct
reply must contain (`"a|b"` accepts either spelling), the `product` whose manual it should
cite, `ref_sections` and a `source`. M5's `e05` is left out: it is the same question as M4's
`q02`. The ten new items: two multi-hop questions (the order says which product's manual),
three the manuals can't answer, two everyday paraphrases, two prompt injections and one more
off-topic question.

The labels name a product and a section, plus a phrase when a section has several chunks
(`{"product": "D300", "section": "Troubleshooting", "contains": "Error E42"}`), never a chunk
ID. Chunk IDs are looked up in the index at run time, so re-chunking with M4's
`ingest.py --chunk-size 600` does not break a label; `--stats` tells you if one stops
matching. Two labels match two chunks, because M4's chunks overlap and the phrase is in both.

## 2. Retrieval on its own

```bash
python m06/retrieval_eval.py
python m06/retrieval_eval.py --split dev --verbose
```

```
mode       k  recall  precision   hit   recall@k per category
                                      manual_exact manual_parap        mixed
dense      1    ...       ...     ...            ...          ...          ...
dense      3    ...
...
hybrid    10    ...
[INFO] Ragas IDBasedContextRecall/Precision differ from this script's numbers on 0 item x mode x k rows
```

No LLM, only the embedding model. For the 30 items with reference chunks and k = 1, 3, 5, 10:
recall@k (reference chunks found / all reference chunks), precision@k (reference chunks in
the top k / k) and hit@k (M4's hit@3 was this at k = 3), per search mode and per category.
Recall can only grow with k and precision usually falls: one or two chunks are right, the rest
of a top 10 is noise, and the desk hands the model k = 3. Ragas' `IDBasedContextRecall` and
`IDBasedContextPrecision` compute the same two numbers from IDs without a judge; the script
runs them as a cross-check. The per-item rows (`state/retrieval_items.csv`) feed step 8.
This search has no product filter; the desk's own search filters by product (M4), which can
only narrow the candidates.

## 3. `nat eval` with repetitions, custom evaluators and a judge

```bash
nat validate --config_file m06/configs/eval_A.yml
nat eval --config_file m06/configs/eval_A.yml --reps 3                         # A -> m06/state/runs/A
OLLAMA_MODEL=llama3.2:1b nat eval --config_file m06/configs/eval_A.yml --reps 3 \
    --override eval.general.output_dir m06/state/runs/B                        # B
```

Configuration B changes one thing: the desk's model. `DESK_TEMPERATURE` (default 0, as in M5)
and `DESK_SEED` set its sampling for either run. `--reps 3` runs every item three times; NAT
renames the items `q01_rep0`, `q01_rep1`, `q01_rep2`, and every output file has one entry per
item and rep. The dataset is `_type: jsonl` with an allowlist filter on `split` (both splits by
default; `check.py` narrows it to dev with
`--override eval.general.dataset.filter.allowlist.field.split dev`).

| Evaluator | Type | Score |
|---|---|---|
| `keywords_all` | `keywords_all` (m06/evaluators.py) | 1 if the reply has every keyword of the item; items without keywords score 1 |
| `refuses_when_unanswerable` | `refuses_when_unanswerable` (m06/evaluators.py) | 1 if the reply declines exactly when the category calls for it (unanswerable, off_topic, injection) |
| `cites_right_manual`, `no_invented_order` | `langsmith_custom` (m05/evals.py) | M5's deterministic checks |
| `answer_accuracy` | `ragas`, `metric: AnswerAccuracy`, `llm_name: judge` | Ragas' NVIDIA metric: reference vs reply, two judge prompts rated 0/2/4, averaged to 0-1 |
| `llm_latency`, `llm_calls`, `tokens_per_call` | `avg_llm_latency`, `avg_num_llm_calls`, `avg_tokens_per_llm_end` | from the profiler's events |

An item-rep passes when `keywords_all` and `refuses_when_unanswerable` are both 1 (M5's
`has_fact`, the share of keywords, is replaced by the stricter `keywords_all`). The custom
evaluators are NAT components: a config class (`EvaluatorBaseConfig, name="keywords_all"`),
a function decorated with `@register_evaluator` that yields an `EvaluatorInfo`, and a
`BaseEvaluator` subclass whose `evaluate_item` turns an `EvalInputItem` (question, reference,
reply, trajectory, the whole dataset row) into an `EvalOutputItem` (score, reasoning).

The judge is an `llms:` entry of `_type: openai` pointing at Ollama's `/v1` (base URL and
model from `M06_JUDGE_BASE_URL` and `M06_JUDGE_MODEL`, default `qwen3:4b`). NAT's `ragas`
evaluator is used for Answer Accuracy only, because it compares the reply with the reference
and needs no context. For context metrics NAT builds Ragas' `retrieved_contexts` from the
outputs of the workflow's `TOOL_END`, `LLM_END` and `CUSTOM_END` steps. The wrapped desk shows
NAT only its LLM calls, so the "contexts" are the desk's own plan, draft and grade (open
`answer_accuracy_output.json` and look at `retrieved_contexts`: `**Step 0** {"steps": ...}`),
not the retrieved passages. That is why `desk_app.py` writes the passages to a sidecar file
and step 4 scores context metrics outside NAT.

The sidecar, `<output_dir>/passages.jsonl`, has one record per item and rep: the test-set
item, the reply, the route (`manual_search D300 (rag)`), the retrieved passages (IDs and
text), the latency and the model. `desk_app.py` finds the output folder the way `nat` does:
from `--override eval.general.output_dir` if given, otherwise from the config file.

Re-score without running the desk (for a new or changed evaluator):

```bash
nat eval --config_file m06/configs/eval_A.yml --skip_workflow \
    --dataset m06/state/runs/A/workflow_output.json --override eval.general.output_dir m06/state/runs/A_rescored
```

`--skip_completed_entries` does the same for items that already have a generated answer,
which resumes an interrupted run. The profiler files are as in M5
(`standardized_data_all.csv`, `workflow_profiling_report.txt`, ...).

## 4. Ragas metrics on the saved runs

```bash
python m06/ragas_eval.py                      # rep 0 of runs/A and runs/B
python m06/ragas_eval.py                      # again: every judge reply now comes from the cache
python m06/ragas_eval.py m06/state/runs/A --all-reps --no-cache
```

```
[INFO] m06/state/runs/A: 42 items (rep 0); judge qwen3:4b at http://localhost:11434/v1; cache on
metric                   scored    mean  NaN  >1 or <0
answer_accuracy             ...     ...  ...       ...
context_relevance           ...
response_groundedness       ...
faithfulness                ...                                (dev split only)
context_recall              ...                                (dev split only)
[INFO] judge calls ..., tokens ... in / ... out, ... min
```

Ragas 0.4.3's `ragas.metrics.collections` API, with the judge from
`llm_factory(model, provider="openai", client=AsyncOpenAI(base_url=Ollama /v1))`. Answer
Accuracy, Context Relevance and Response Groundedness (NVIDIA's metrics, two short judge
prompts each) run on every item that has what they need; Faithfulness and Context Recall
break the text into claims first and cost more calls, so they run on the dev split only. The
script counts judge calls and tokens with an HTTP hook on the client and writes one Ragas
experiment CSV per run (`state/ragas/experiments/ragas_A.csv`) plus a summary JSON.

`DiskCacheBackend` stores each judge reply on disk, so the second run makes no judge calls
and returns the same scores. The cache key is the prompt and the reply schema, not the model,
so each judge model gets its own cache folder (`state/ragas_cache/<model>/`). A cached score
is a repeated score, not a more correct one: without the cache the same judge may answer
differently. Ragas returns NaN when a judge reply can't be parsed and does not clamp scores
to [0, 1]; both are counted.

## 5. How far can the judge be trusted?

```bash
python m06/judge_check.py
python m06/judge_check.py --temperature 0.8        # verdict stability when the judge samples
```

```
Single replies (24 labelled: 12 pass, 12 fail; 3 reps each, temperature 0)
judge                  parseable   agree  false pass  false fail   same verdict x3
qwen3:4b                   .../24    .../..        ...         ...             .../24
llama3.2:3b (self)         .../24    ...

Pairs (8, each judged A/B and B/A)
judge                  parseable  position-consistent  wins (both orders)  first-slot picks
...
Planted cases (human -> the judge's majority verdict; for the pair, the reply it picked in A/B, B/A order)
  qwen3:4b             right_but_terse pass->...; wrong_but_long_polite fail->...; wrong_number fail->...; ...
```

Each labelled reply is judged three times with a pass/fail JSON verdict, by the separate
judge (`qwen3:4b`) and by the desk's own model (`llama3.2:3b`, the self-judge). The majority
is compared with the human verdict: agreement, false passes (too lenient) and false fails.
The planted cases: a terse but right reply, a long and polite but wrong one, one wrong number,
a citation to the wrong manual. Each of the 8 pairs is judged twice with the replies swapped;
a reply wins only if the judge picks it in both orders, and "first-slot picks" shows how often
it simply chose whichever came first. One pair is a long, repetitive, wrong reply against a
short right one (verbosity). The replies are hand-written, so this does not measure
self-enhancement bias. No threshold is applied; 24 labels for one desk are a small sample.
At temperature 0 the repeats are usually identical; `--temperature 0.8` shows how stable the
verdicts are when the judge samples.

## 6. A vs B: per category, paired, with a significance test

```bash
python m06/compare_configs.py
python m06/compare_configs.py m06/state/runs/A m06/state/runs/B --show-flips
```

```
                           pass rate   answer acc.      grounded         p50 s         p95 s     LLM calls        tokens
category             n      A      B      A      B      A      B      A      B      A      B      A      B      A      B
order                3    ...
...
ALL                 42    ...
Judge calls: A: NAT Answer Accuracy at least ... (2 per item and rep), ragas_eval ...; B: ...
Rep-to-rep spread (items whose verdict changed between reps): A: ...; B: ...
Paired flips (A -> B): ... regressions, ... improvements, ... pass in both, ... fail in both
McNemar exact (binomial on the discordant pairs): p = ...
Pareto (pass rate vs p50 latency): ...
VERDICT: INCONCLUSIVE: ...
```

An item passes when it passed in more than half of its reps. The table puts pass rate,
Answer Accuracy, groundedness (step 4), p50/p95 latency (from the sidecar), LLM calls and
tokens per item (the profiler's traces) side by side per category. The rep-to-rep spread
lists the items whose verdict changed between reps of the same configuration. Paired flips
count the items that passed in A and failed in B (regressions) and the reverse
(improvements); only these discordant pairs carry information, and the exact McNemar test is
a two-sided binomial test of the improvements out of all flips
(`scipy.stats.binomtest(improvements, flips, 0.5)`). The verdict says INCONCLUSIVE unless the
flips are lopsided enough for p < 0.05, and even then it asks you to confirm on the held-out
split: with about 40 items few items flip, and a small sample only detects large differences.
The Pareto line says whether one configuration is at least as accurate and at least as fast
as the other, or whether the choice depends on your priority.

The script also writes NeMo Evaluator's format for each run: `state/nel/<run>/results.jsonl`
(one record per item and rep: `problem_idx`, `repeat`, `reward` 1.0/0.0, `metadata.category`,
the reply and the reference) and a minimal `eval-desk.json` bundle (`benchmark.name`,
`benchmark.scores.mean_reward.value`, per-category means).

### Optional: `nel compare` in a second venv

NeMo Evaluator needs Python 3.12 or 3.13 (`>=3.12,<3.14`), so it gets its own venv; the lab
venv stays on Python 3.11. Install it from the Git repository at the release tag `v0.3.0`,
with the `stats` extra (scipy, for the tests), so the code is the release the course docs
describe:

```bash
python3.12 -m venv ~/venvs/nel
~/venvs/nel/bin/pip install "nemo-evaluator[stats] @ git+https://github.com/NVIDIA-NeMo/Evaluator.git@v0.3.0"
~/venvs/nel/bin/nel --version
~/venvs/nel/bin/nel compare m06/state/nel/A m06/state/nel/B --correct-above 0.5 --show-flips --no-strict
~/venvs/nel/bin/nel compare m06/state/nel/A m06/state/nel/B --correct-above 0.5 --no-strict --verbose
```

`nel compare` pairs the same items, prints per-category deltas and flips and gives its own
verdict (PASS, WARN, BLOCK or INCONCLUSIVE) with the number of discordant pairs it would
need. It averages the repeats per item first; when all rewards are then 0 or 1 it runs
McNemar, otherwise a sign or permutation test. Its McNemar is one-sided (does the candidate
regress?), so its p-value differs from the two-sided one above on the same flips. It writes
a Markdown report next to the candidate bundle (`--no-report` to skip). `nel gate` (policy
tiers) is Module 8's CI step.

## 7. Feedback into the test set

```bash
python m06/feedback.py rate --question "How many devices can the H200 remember?"
python m06/feedback.py rate --question "Where is order A1003?" --answer down --reason wrong_fact \
    --comment "It said shipped" --correction "Order A1003 is still processing, awaiting stock."
python m06/feedback.py report
python m06/feedback.py promote
nat eval --config_file m06/configs/eval_A.yml \
    --override eval.general.dataset.file_path m06/data/testset_feedback.jsonl \
    --override eval.general.output_dir m06/state/runs/feedback
```

```
[promote] f01 <- fb-seed-02 (wrong_fact): How many devices can the H200 remember?
[promote] f02 <- fb-seed-03 (should_refuse): Which customers bought a D300 dock last week?
[skip]    fb-seed-04: no correction
[skip]    fb-seed-05: near-duplicate of q01 (Jaccard 0.83)
[skip]    fb-...: near-duplicate of e01 (Jaccard 1.00)
```

`rate` runs the desk, shows the reply and records one structured rating: the key
`user_rating` with a score (1 up, 0 down) and a value, a reason from a fixed list
(`wrong_fact`, `wrong_product`, `missing_citation`, `should_refuse`, `should_answer`,
`other`; anything else is refused), an optional comment and an optional correct answer, plus
the trace ID, model, route and retrieved chunk IDs of that run. Without `--answer` it asks
you. Records go to `state/feedback.jsonl`; `data/feedback_seed.jsonl` holds six more so the
loop runs without typing. `report` counts them by reason. `promote` turns every thumbs-down
record with a correction into a test-set candidate (`source: feedback`, `split: dev`,
`needs_review: true`) and skips questions that share at least 60% of their words with a
test-set question (one seed record is a planted near-duplicate of `q01`). A person then
checks the reference, category and keywords, adds `ref_sections` and moves the item into
`data/testset.jsonl`. The `nat eval` above scores the candidates with the same evaluators.
This grows the test set; it does not retrain anything. Module 10 wires buttons to the same
file.

## 8. Triage: what to fix first

```bash
python m06/triage.py                       # configuration A
python m06/triage.py m06/state/runs/B
```

```
bucket           manual_exac manual_para   mixed  unanswerabl  off_topic  injection   total
wrong_refusal              .           .       .          ...        ...        ...     ...
...
Where the time goes (profiler traces, mean per item): ... s per turn, of which ... s in ... LLM calls (...%) ...
Fix first: ...
```

Every failed item goes into the first bucket that fits: `infra` (error or empty reply),
`routing` (the plan used the wrong source), `wrong_refusal`, `retrieval_miss` (no reference
chunk among the passages the desk retrieved), `citation`, `judge_disagrees` (the keyword
check failed it but Answer Accuracy says it matches the reference: read it by hand) and
`not_grounded` (the facts were available but the reply doesn't carry them). The report shows
bucket x category counts, two examples per bucket, where the time goes (LLM calls vs
everything else, from the profiler's traces; NAT sees the wrapped graph as one function, so
it can't name the M4 node) and the buckets by size. The bare M5 desk has no refusal path
(M5's rails did the refusing), so expect the refusal categories in `wrong_refusal`.

## Check

```bash
python m06/check.py
```

18 checks on the dev split with 2 reps (the full run above uses every item and 3 reps): the
four models answer; the test set loads and its labels resolve; `retrieval_eval` writes 3 x 4
rows, recall never falls as k grows and Ragas' ID-based metrics agree; `nat validate` passes
and the custom evaluators are registered; `nat eval --reps 2` writes items x reps outputs,
`LLM_END` rows and every evaluator file; the sidecar has one record per item and rep with
real chunk IDs; NAT's Answer Accuracy scored every item-rep in [0, 1]; `ragas_eval` scores the
NVIDIA metrics and a second run comes from the cache; `--skip_workflow` re-scores without the
desk; `judge_check` gets parseable verdicts for at least 90% of the labels per judge and
prints the pairwise position consistency; B's run and the comparison table; flips, McNemar
and the NeMo Evaluator files; temperature 0 with a seed gives the same text twice; a scripted
rating is saved and counted; `promote` keeps the two seeded candidates and skips the planted
duplicate; `nat eval` scores the promoted items; triage puts every failed item in one bucket.
The check makes several hundred calls to the local models; it rebuilds `m04/state` first.

Small local models don't answer the same way every time; if a check fails, run it again,
and read `m06/state/check.log` for the details.

Offline self-test (course maintainers only): `M06_FAKE_LLM=1 python m06/check.py` starts
`m06/tests/fake_oai.py`, M5's scripted server plus rules that return a valid instance of
whatever JSON schema Ragas or NAT ask for. It proves plumbing, not quality. You never need it.

## Notes (API details found while building the lab)

- NAT 1.9.0 registers plugins only from Python entry points (`nat.components`); a module on
  `sys.path` is not enough, hence `m06/pyproject.toml` and `pip install --no-deps -e m06`.
- The NAT docs mention `eval.general.dataset.pass_full_entry`; NAT 1.9.0 has no such key (the
  dataset config forbids unknown keys) and always passes the whole row to evaluators as
  `EvalInputItem.full_dataset_entry`.
- `nat eval --dataset FILE` reads FILE as JSON (`pandas.read_json`), so a `.jsonl` file fails
  with "Trailing data". Point the configured `jsonl` dataset elsewhere with
  `--override eval.general.dataset.file_path FILE`. `--dataset` works for `--skip_workflow`,
  whose input, `workflow_output.json`, is JSON.
- A list value in `--override` is written comma-separated (`...split dev,test`).
- NAT's `ragas` evaluator turns a NaN from Ragas into 0 (`nan_to_zero`), and any evaluator
  exception into a score of 0 with an `error` in its reasoning; an unparseable judge reply therefore
  looks like a wrong answer in `answer_accuracy_output.json`. `ragas_eval.py` keeps the NaN.
- NAT's judge (`_type: openai`) is LangChain's `ChatOpenAI`; Ragas calls it through
  `with_structured_output`, which sends `response_format: json_schema` (strict). Extra keys in
  the `llms:` entry go to `ChatOpenAI` (the config allows extra fields), which is how
  `reasoning_effort` reaches Ollama. `ChatOpenAI` 1.6.6 sends `max_tokens` as
  `max_completion_tokens`, which Ollama's `/v1` field list doesn't include.
- NAT's own `thinking` switch only applies to Nemotron model names, so it can't switch qwen3's
  thinking off. Ollama's OpenAI-compatible API takes `reasoning_effort`; for a model with an
  on/off switch, `"none"` requests no thinking. The native API has `think: false` (Python
  client `ollama.chat(..., think=False)`, `ChatOllama(reasoning=False)`). The M6 judge calls
  send `reasoning_effort: "none"` to qwen3 models only. On the Mac run this was ignored: the
  current `qwen3:4b` kept thinking, and Ollama returned the thinking inside the answer (ending in
  a bare `</think>`). Asking for schema-constrained JSON (`json_schema_format` in `llm_calls.py`)
  is what keeps the judge's replies short and parseable. Note that `setup/llm.py`
  switches reasoning on for qwen3 when it is the desk's model.
- Ragas' `llm_factory` with an OpenAI client uses Instructor's JSON mode
  (`response_format: json_object`, the schema in the prompt) with temperature 0.01, top_p 0.1
  and max_tokens 1024 unless you pass others; `ragas_eval.py` passes temperature 0. In JSON mode a
  small local judge often returned JSON without the metric's field (`rating` missing), and
  Instructor retried each call three times. `ragas_eval.py` therefore re-patches the client in
  Instructor's `JSON_SCHEMA` mode (`response_format: json_schema` with the metric's schema, which
  Ollama enforces), allows one retry (`M06_JUDGE_RETRIES`) and 120 s per call (`M06_JUDGE_TIMEOUT`);
  a reply that still misses the schema is scored NaN and counted.
- Ragas' disk cache key leaves out the model name (see step 4).
- Ragas 0.4.3 warns that importing `IDBasedContextRecall` from `ragas.metrics` is deprecated
  and points to `ragas.metrics.collections`, which doesn't contain it yet.
- A Ragas `@experiment` runs all rows at once and drops a row whose function raised;
  `ragas_eval.py` limits the concurrency and records metric errors in the row instead.
- M5's graph inside another graph's node takes that graph's store, so `desk_app.agent` is
  compiled with an `InMemoryStore` (as M5's `guarded_agent`).
- NeMo Evaluator 0.3.0's `nel compare` needs an `eval-*.json` bundle next to `results.jsonl`;
  it reads `problem_idx`, `repeat`, `reward`, `metadata.category`, `model_response` and
  `expected_answer`.
- The planning probe found NAT's `langsmith_judge` failing with these LLM entries; this lab
  does not use it (not re-tested here).

## What this lab does not show

This lab does not show, and the videos don't say, that:

- A 3B or 4B local judge agrees with humans like the GPT-4 judges in the LLM-as-a-judge paper
  (over 80%). Judge agreement here is measured on 24 hand labels for this desk only.
- Configuration A and B differ significantly. About 40 items give few discordant pairs, and
  NeMo Evaluator's own power table says small sets detect only large effects.
- NAT's `ragas` context metrics on the wrapped desk measure the retrieved passages (they see
  the desk's LLM outputs), or that `langsmith_judge` works in NAT 1.9.0 with these LLM entries.
- Ragas scores are deterministic, or that caching makes them more correct (it only repeats them).
- Mac latency, tokens or the 1B-vs-3B differences say anything about GPU or NIM serving; or
  anything about prices.
- The feedback loop retrains anything (it grows the test set), or that NVIDIA's Data Flywheel
  Blueprint uses thumbs feedback (it doesn't).
- NeMo Evaluator's built-in benchmarks (MMLU, GSM8K, ...) were run, or that its version is
  the one printed in its docs.
- Self-enhancement bias was measured: the labelled replies are hand-written, not the desk's.

## Tested

| Mode | Where | Result |
|---|---|---|
| Offline self-test (`M06_FAKE_LLM=1`) | Linux, Python 3.11.15, 2026-09-30 | 18 passed, 0 failed |

Packages in the self-test venv (`uv pip check` clean): nvidia-nat 1.9.0 (with `langchain`,
`profiler`), nvidia-nat-ragas 1.9.0, ragas 0.4.3, instructor 1.17.0, datasets 5.0.1, scipy
1.17.1, diskcache 5.6.3, openai 2.54.0, httpx 0.28.1, langchain-core 1.6.5, langchain-openai
1.6.6, langchain-ollama 1.1.0, langgraph 1.2.12, ollama 0.6.2 (Python client), pymilvus 2.6.9,
milvus-lite 3.2.1, nemoguardrails 0.24.1, pandas 2.3.3. The optional NeMo Evaluator venv:
Python 3.12.3, nemo-evaluator 0.3.0 from the `v0.3.0` tag (`nel, version 0.3.0`), scipy
1.18.1; `nel compare` ran on the self-test's `results.jsonl` and `eval-desk.json`.

## Runs

To be filled after the first free-mode run on a Mac (Ollama `llama3.2:3b`, `llama3.2:1b`,
`qwen3:4b`, `embeddinggemma`).

| Run | Where | Result |
|---|---|---|
| | | |

The manuals, orders, customers, test questions, labels and feedback are made up for the course.
