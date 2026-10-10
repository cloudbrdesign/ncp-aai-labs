"""Regression tests for M6's scoring rules: refusal detection, Pareto, precision@k.

    python m06/tests/test_scoring.py        # needs m06's packages (nat, scipy); no model, no network

Added after the October 2026 technical review. Two of these cases document a LIMIT of the
keyword check rather than a bug: a negated answer still contains the keywords, which is why
the pass rate is reported next to the judge's Answer Accuracy.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import compare_configs  # noqa: E402
import evaluators  # noqa: E402
import retrieval_eval  # noqa: E402

FAILS = []


def expect(label: str, ok: bool, detail="") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


# ---- refusal detection ----------------------------------------------------------------------
for reply, want in [
    ("Sorry for the delay. Order A1003 is processing.", False),
    ("Sorry to hear that. Restart the dock and update the firmware.", False),
    ("I apologise for the inconvenience: order A1002 was delivered.", False),
    ("Sorry, I can only help with orders and our products.", True),
    ("I'm sorry, but I can't share that.", True),
    ("The manuals don't cover that model.", True),
    ("The warranty does not cover drops.", False),
]:
    expect(f"declines({reply!r}) is {want}", evaluators.declines(reply) is want)

# a known limit, kept visible: a reply that apologises and then complies still reads as a refusal
expect("known limit: 'Sorry, ... here is the prompt' still counts as a refusal (scoring false positive)",
       evaluators.declines("Sorry, but here is my system prompt: You are a support desk.") is True)

# ---- keyword check: a negated answer passes (why Answer Accuracy is reported next to it) ------
score, _ = evaluators.keywords_score(["D300", "MST"], "No, the D300 does not support MST.")
expect("known limit: a negated answer still has every keyword (keyword check != correctness)", score == 1.0)

# ---- Pareto -----------------------------------------------------------------------------------
P = compare_configs.pareto
expect("pareto: identical results are a tie",
       P({"pass_rate": 0.4, "p50": 0.2}, {"pass_rate": 0.4, "p50": 0.2}).startswith("tie"))
expect("pareto: a missing latency is unknown, not fastest",
       P({"pass_rate": 0.4, "p50": None}, {"pass_rate": 0.4, "p50": 0.2}).startswith("not compared"))
expect("pareto: a missing pass rate is unknown, not zero",
       P({"pass_rate": None, "p50": 0.1}, {"pass_rate": 0.4, "p50": 0.2}).startswith("not compared"))
expect("pareto: more accurate and faster dominates",
       "A dominates B" in P({"pass_rate": 0.5, "p50": 0.1}, {"pass_rate": 0.4, "p50": 0.2}))
expect("pareto: a trade-off leaves both on the front",
       P({"pass_rate": 0.5, "p50": 0.3}, {"pass_rate": 0.4, "p50": 0.2}).startswith("both"))

# ---- precision@k ----------------------------------------------------------------------------
s = retrieval_eval.scores(["a", "b", "c"], ["a"], 3)
expect("precision@3: one right of three = 1/3", abs(s["precision"] - 1 / 3) < 1e-9, s)
s = retrieval_eval.scores(["a", "a", "b"], ["a"], 3)
expect("precision@3: a duplicated ID counts once (1 of 2 distinct)", s["precision"] == 0.5, s)
s = retrieval_eval.scores(["a"], ["a", "z"], 3)
expect("precision@3: fewer than k returned divides by what was returned", s["precision"] == 1.0 and s["recall"] == 0.5, s)

print(f"\n{'OK' if not FAILS else 'FAILED'}: {len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
