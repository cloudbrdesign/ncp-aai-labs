"""A scripted stand-in for the model, for the offline self-test only (M03_FAKE_LLM=1).

Learners never need this. It lets `M03_FAKE_LLM=1 python m03/check.py` exercise the
whole lab on a machine without Ollama. It returns plausible outputs for each job:

  Plan       the regex planner's plan (M03_FAKE_BAD_PLAN=1: an invalid plan, to exercise the fallback)
  Grade      2, or 0 when the reply mentions an order ID that is not in the facts
  DateFacts  the dates from the question (program-aided reasoning)
  draft      a reply built from the facts in the prompt
  reasoning  direct and chain-of-thought answers with a few typical small-model mistakes
"""
import json
import os
import re

import handlers
import planner

# reasoning_check.py: (method, question number) -> the fake's answer. Direct slips on the
# 30-day edge and on weekends; CoT fixes most of it but still miscounts one weekend.
REASONING = {
    "direct": {1: "yes", 2: "no", 3: "no", 4: "2026-09-08", 5: "2026-10-03"},
    "cot": {1: "yes", 2: "no", 3: "yes", 4: "2026-09-10", 5: "2026-10-03"},
}


def _text(messages, role):
    return "\n".join(t for r, t in messages if r == role)


def chat(messages, schema=None):
    system, user = _text(messages, "system"), _text(messages, "user")
    name = getattr(schema, "__name__", None)
    if name == "Plan":
        return _plan(user)
    if name == "Grade":
        return _grade(user, schema)
    if name == "DateFacts":
        dates = re.findall(r"\d{4}-\d\d-\d\d", user)
        kind = "refund_date" if "refund" in user.lower() and "arrived" in user.lower() else "return_window"
        return schema(question_type=kind, start_date=dates[0], today=dates[1] if len(dates) > 1 else "")
    if "Write the reply to the customer" in system:
        return _draft(system)
    if "Question" not in user:
        return "OK"
    q = int(re.search(r"Question (\d+)", user).group(1))
    if "step by step" in system:
        return f"Let me work through the dates one at a time.\nAnswer: {REASONING['cot'][q]}"
    return REASONING["direct"][q]


def _plan(user):
    if os.environ.get("M03_FAKE_BAD_PLAN") == "1":   # an unknown step type and too many steps
        raw = {"steps": [{"action": "refund_now", "order_id": "A1002"}] * 5}
        return planner.Plan.model_validate(raw)       # raises ValidationError, like a bad model output
    request = user.split("Request:")[-1].strip()
    carried = planner.order_ids(user.split("Request:")[0])
    return planner.regex_plan(request, carried)


def _grade(user, schema):
    facts, reply = user.split("Reply:", 1)
    extra = [o for o in planner.order_ids(reply) if o not in planner.order_ids(facts)]
    if extra:
        return schema(score=0, reason=f"The reply mentions {extra[0]}, which is not in the facts.")
    return schema(score=2, reason="Every claim in the reply matches the facts.")


def _draft(system):
    evidence = [json.loads(line[2:]) for line in system.splitlines() if line.startswith("- {")]
    profile = {"contact": "email"} if "email only" in system else {}
    reply = "Thanks for reaching out. " + handlers.template_reply(evidence, profile)
    return reply + (" We'll give you a call to follow up." if "prefers a phone call" in system else "")
