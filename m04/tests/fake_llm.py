"""A scripted stand-in for the models, for the offline self-test only (M04_FAKE_LLM=1).

Learners never need this. It lets `M04_FAKE_LLM=1 python m04/check.py` exercise the
whole lab on a machine without Ollama:

  embed()   a hashed bag of words: each word adds 1 to one of DIM slots (MD5 of the word,
            so it is the same on every machine). Texts that share words get similar
            vectors, which is enough to exercise Milvus dense and hybrid search. It knows
            no synonyms, so it behaves like keyword search, not like a real embedding model.
  Plan      the router's keyword fallback (M04_FAKE_BAD_PLAN=1: an invalid plan)
  Grade     2, or 0 when the reply cites a chunk ID that is not in the passages
  SqlQuery  a fixed SELECT for the demo question
  draft     a reply built from the facts and the first passage, with its chunk ID
"""
import hashlib
import os
import re

DIM = 256
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "to", "of", "is", "it", "my", "i", "do", "what", "how", "on", "in",
        "for", "with", "does", "can", "you", "your", "this", "that", "be", "are", "at", "from", "by", "up"}


def embed(texts):
    vectors = []
    for t in texts:
        v = [0.0] * DIM
        for w in WORD.findall(t.lower()):
            if w not in STOP:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        v[0] += 0.01   # never an all-zero vector
        vectors.append(v)
    return vectors


def _text(messages, role):
    return "\n".join(t for r, t in messages if r == role)


def chat(messages, schema=None):
    system, user = _text(messages, "system"), _text(messages, "user")
    name = getattr(schema, "__name__", None)
    if name == "Plan":
        import router
        if os.environ.get("M04_FAKE_BAD_PLAN") == "1":      # an unknown step type and too many steps
            return router.Plan.model_validate({"steps": [{"action": "refund_now"}] * 5})
        request = user.split("Request:")[-1].strip()
        carried = router.order_ids(user.split("Request:")[0])
        return router.keyword_plan(request, carried)
    if name == "Grade":
        facts, reply = user.split("Reply:", 1)
        given = set(re.findall(r"\[([A-Z0-9]+-[a-z0-9-]+)\]", facts))
        cited = set(re.findall(r"\[([A-Z0-9]+-[a-z0-9-]+)\]", reply))
        if cited - given:
            return schema(score=0, reason=f"The reply cites {sorted(cited - given)[0]}, which is not in the passages.")
        return schema(score=2, reason="Every claim in the reply matches the facts and passages.")
    if name == "SqlQuery":
        return schema(sql="SELECT order_id, customer, note FROM orders WHERE status = 'processing' LIMIT 20")
    if "Write the reply to the customer" in system:
        facts_part = system.split("Facts from our systems:")[1].split("Manual passages:")[0]
        facts = [l[2:] for l in facts_part.splitlines() if l.startswith("- ")]
        passage = re.search(r"^\[([A-Z0-9]+-[a-z0-9-]+)\] (.+)$", system, re.M)
        parts = ["Thanks for reaching out."] + facts
        if passage:
            first = re.split(r"(?<=[.!?])\s", passage.group(2))[0]
            parts.append(f"{first} [{passage.group(1)}]")
        return " ".join(parts)
    return "OK"
