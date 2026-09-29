"""Self-critique: check the draft before it goes to the customer.

Two kinds of evaluator, as in Reflexion and the RAG Blueprint's self-reflection:

  deterministic checks (they decide)   every planned order ID is answered; no order ID
                                       that isn't in the evidence; the stored contact
                                       preference is respected
  model grade                          groundedness 0-2: is every claim supported by the facts?
                                       0 (the reply contradicts the facts) sends it back too,
                                       like the RAG Blueprint's groundedness threshold

Each failed check comes back as named feedback ("unknown_order: ...") for the next
draft, and as a one-line lesson for the customer's lesson memory.
"""
import pathlib
import re

from pydantic import BaseModel, Field

from llm_calls import chat
from planner import order_ids

PROMPT = (pathlib.Path(__file__).resolve().parent / "prompts" / "grade.txt").read_text()
OTHER_REF = re.compile(r"#\s?\d{3,}")   # made-up references such as "order #1234"
PHONE = re.compile(r"\b(call you|give you a call|phone you|ring you|text you|sms|over the phone|phone call)\b", re.I)


class Grade(BaseModel):
    score: int = Field(ge=0, le=2, description="2 = every claim is in the facts, 1 = partly, 0 = not supported")
    reason: str = Field(description="one short sentence")


def checks(draft: str, planned_ids: list[str], evidence_ids: list[str], profile: dict) -> list[str]:
    """The deterministic checks. Returns named feedback; an empty list means the draft passes."""
    in_draft = order_ids(draft)
    feedback = [f"missing_order: the reply does not answer order {o}" for o in planned_ids if o not in in_draft]
    feedback += [f"unknown_order: {o} is not in the facts; remove it" for o in in_draft if o not in evidence_ids]
    feedback += [f"unknown_order: {o} is not in the facts; remove it" for o in OTHER_REF.findall(draft)]
    if profile.get("contact") == "email" and PHONE.search(draft):
        feedback.append("contact_preference: the customer wants email only; do not offer a call")
    return feedback


LESSONS = {
    "missing_order": "Answer every order ID the customer asked about.",
    "unknown_order": "Only mention order IDs that are in the facts.",
    "contact_preference": "This customer wants email only: never offer a phone call.",
    "grounding": "Say only what the facts say; don't add details.",
}


def lesson_for(feedback: str) -> str:
    return LESSONS[feedback.split(":")[0]]


def grade(facts: str, draft: str, log=print) -> int | None:
    """One groundedness grade from the model (0-2). None if the model fails."""
    try:
        g = chat([("system", PROMPT), ("user", f"Facts:\n{facts}\n\nReply:\n{draft}")], schema=Grade)
        log(f"[critique] groundedness {g.score}/2 (model): {g.reason}")
        return g.score
    except Exception as e:
        log(f"[critique] groundedness grade unavailable ({type(e).__name__}); the rule checks still decide")
        return None
