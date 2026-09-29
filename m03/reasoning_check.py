"""Three ways to reason about dates: direct answer, chain-of-thought, program-aided.

    python m03/reasoning_check.py
    python m03/reasoning_check.py --verbose     # also print every model reply

Five return-window and refund-date questions from the M1 returns policy (30 days from
delivery; refund within 5 business days of the return arriving). Each is asked three ways:

  direct         "reply with only the answer"
  chain-of-thought  "think step by step, then give the answer"
  program-aided  the model only pulls the dates out as JSON; Python does the date maths

Python knows the right answers, so each method gets a score. Small models often write a
fluent chain of thought that still miscounts days; letting code do the arithmetic is
the program-aided (PAL) idea.
"""
import argparse
import datetime as dt
import pathlib
import re
from typing import Literal

from pydantic import BaseModel, Field

from llm_calls import chat, describe

POLICY = (pathlib.Path(__file__).resolve().parents[1] / "m01" / "data" / "returns_policy.md").read_text()
WINDOW_DAYS, REFUND_BUSINESS_DAYS = 30, 5


def day(s: str) -> str:
    return f"{dt.date.fromisoformat(s):%A} {s}"


# (kind, first date, today) -- today is only used by return-window questions
CASES = [
    ("return_window", "2026-08-03", "2026-08-30"),   # 27 days: yes
    ("return_window", "2026-07-28", "2026-08-28"),   # 31 days: no
    ("return_window", "2026-08-01", "2026-08-31"),   # exactly 30 days: yes
    ("refund_date", "2026-09-03", ""),                # Thursday: the 5 business days cross a weekend
    ("refund_date", "2026-09-28", ""),                # Monday: crosses a weekend and a month end
]


def question(kind: str, first: str, today: str) -> str:
    if kind == "return_window":
        return f"An order was delivered on {day(first)}. Today is {day(today)}. Can the customer still return it?"
    return (f"A customer's return arrived at our warehouse on {day(first)}. "
            "What is the latest date the refund should arrive?")


# ---- the program in program-aided reasoning ---------------------------------------

def add_business_days(start: dt.date, n: int) -> dt.date:
    d = start
    while n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:   # Monday..Friday
            n -= 1
    return d


def solve(kind: str, first: str, today: str) -> str:
    start = dt.date.fromisoformat(first)
    if kind == "return_window":
        return "yes" if (dt.date.fromisoformat(today) - start).days <= WINDOW_DAYS else "no"
    return add_business_days(start, REFUND_BUSINESS_DAYS).isoformat()


class DateFacts(BaseModel):
    question_type: Literal["return_window", "refund_date"]
    start_date: str = Field(description="the delivery date or the date the return arrived, as YYYY-MM-DD")
    today: str = Field(default="", description="today's date as YYYY-MM-DD, or empty if not given")


# ---- the three methods --------------------------------------------------------------

DIRECT = (f"You answer questions about this returns policy:\n{POLICY}\n"
          "Reply with only the final answer: yes or no, or a date as YYYY-MM-DD.")
COT = (f"You answer questions about this returns policy:\n{POLICY}\n"
       "Think step by step: write out the dates and count the days. "
       "Then write the last line as 'Answer: yes', 'Answer: no' or 'Answer: YYYY-MM-DD'.")
EXTRACT = ("Read the question and fill in the JSON: the question type (return_window or refund_date), "
           "the start date (delivery date, or the date the return arrived) and today's date if it is given.")


def parse(reply: str) -> str:
    """Take the answer after the last 'Answer:', else the last date or yes/no in the reply."""
    tail = reply.rsplit("Answer:", 1)[-1].lower()
    found = re.findall(r"\d{4}-\d\d-\d\d|\byes\b|\bno\b", tail)
    return found[-1] if found else "?"


def ask(system: str, n: int, q: str) -> tuple[str, str]:
    try:
        reply = chat([("system", system), ("user", f"Question {n}: {q}")])
        return parse(reply), reply
    except Exception as e:
        return "?", f"(model failed: {type(e).__name__})"


def program_aided(n: int, q: str) -> tuple[str, str]:
    try:
        facts = chat([("system", EXTRACT), ("user", f"Question {n}: {q}")], schema=DateFacts)
        return solve(facts.question_type, facts.start_date, facts.today), facts.model_dump_json()
    except Exception as e:   # bad JSON or a date Python can't read
        return "?", f"(extraction failed: {type(e).__name__})"


def run(verbose=False, show=print) -> dict:
    scores = {"direct": 0, "chain-of-thought": 0, "program-aided": 0}
    for n, (kind, first, today) in enumerate(CASES, 1):
        q, expected = question(kind, first, today), solve(kind, first, today)
        answers = {"direct": ask(DIRECT, n, q), "chain-of-thought": ask(COT, n, q),
                   "program-aided": program_aided(n, q)}
        marks = [f"{label} {a} {'ok' if a == expected else 'WRONG'}"
                 for label, (a, _) in zip(("direct", "CoT", "PAL"), answers.values())]
        show(f"Q{n} {q}\n   expected {expected} | " + " | ".join(marks))
        for method, (answer, reply) in answers.items():
            scores[method] += answer == expected
            if verbose:
                show(f"   --- {method} said:\n   " + reply.replace("\n", "\n   "))
    show("\nScore: " + "   ".join(f"{m} {s}/{len(CASES)}" for m, s in scores.items()))
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="print every model reply")
    args = ap.parse_args()
    print(f"[INFO] model: {describe()}\n")
    run(args.verbose)
