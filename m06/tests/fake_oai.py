"""A scripted stand-in for Ollama, for the M6 offline self-test only (M06_FAKE_LLM=1).

Learners never need this. It is M5's fake server (m05/tests/fake_oai.py) with one more rule
set, so `M06_FAKE_LLM=1 python m06/check.py` runs the whole lab without Ollama. It is not a
model: every reply comes from rules, so it proves plumbing and APIs, never quality.

    python m06/tests/fake_oai.py --port 11999

What it adds to M5's rules:
  JSON schemas   Ragas (through Instructor's JSON mode: the schema is in the prompt text) and
                 NAT's `ragas` evaluator (LangChain `with_structured_output`: response_format
                 json_schema) ask for structured replies. The fake finds the schema wherever the
                 request carries it and returns a valid instance of it: the minimal one, except
                 for the rules below.
  ratings        the NVIDIA metrics' {"rating": n} replies: the top rating when the two texts
                 being compared share most of their words, the middle one when they share some,
                 0 otherwise (AnswerAccuracy 0/2/4; ContextRelevance, ResponseGroundedness 0/1/2)
  judge_check    {"verdict": "pass"|"fail"}: pass when every number, error code and chunk ID in
                 the reply appears in the passages and the reply shares words with them;
                 {"better": "A"|"B"}: the reply with more words in common with the passages
                 (ties go to A, so position bias shows up in the pairwise test)
  models         llama3.2:1b, qwen3:4b and nemotron-mini are listed next to M5's models
  llama3.2:1b    the smaller model drafts differently, so configuration B differs from A and the
                 comparison has flips to count: a vague reply for about a third of the questions
                 (fixed by a hash; the desk's critique usually sends it back), and a refusal when
                 the draft has no manual passages
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("m05_fake_oai", LABS / "m05" / "tests" / "fake_oai.py")
m05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m05)

MODELS = ["llama3.2:3b", "llama3.2:1b", "qwen3:4b", "nemotron-mini", "embeddinggemma"]
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "to", "of", "is", "it", "my", "i", "do", "what", "how", "on", "in",
        "for", "with", "does", "can", "you", "your", "this", "that", "be", "are", "at", "from", "by", "up"}
NUMBER = re.compile(r"\b(?:[A-Z]\d{2,4}|\d+(?:[.,]\d+)?)\b")
CHUNK = re.compile(r"\b((?:H200|D300|M270|FAQ)-[a-z0-9-]+?-\d+)\b")
TOP = {"AnswerAccuracyOutput": (4, 2), "ContextRelevanceOutput": (2, 1), "ResponseGroundednessOutput": (2, 1)}
FIELDS = {"AnswerAccuracyOutput": ("user_answer", "reference_answer"),
          "ContextRelevanceOutput": ("user_input", "context"),
          "ResponseGroundednessOutput": ("response", "context")}


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(str(text).lower()) if w not in STOP}


def overlap(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(len(wa), 1)


# ---- finding the schema ----------------------------------------------------------------

def schema_in(req: dict, text: str) -> dict | None:
    """The JSON schema of a structured-output request, from any of the places clients put it."""
    rf = req.get("response_format") or {}
    if rf.get("type") == "json_schema":
        return rf.get("json_schema", {}).get("schema")
    if isinstance(req.get("format"), dict):                  # Ollama /api/chat
        return req["format"]
    for marker in ("json_schema:", "JSON Schema:"):          # Instructor JSON mode, Ragas prompts
        at = text.rfind(marker)
        if at >= 0:
            try:
                start = text.index("{", at)
                return json.JSONDecoder().raw_decode(text[start:])[0]
            except ValueError:
                pass
    return None


def instance(schema: dict, defs: dict | None = None) -> object:
    """A minimal valid instance of a JSON schema (objects, arrays, enums, $ref, anyOf)."""
    defs = defs if defs is not None else schema.get("$defs", schema.get("definitions", {}))
    if "$ref" in schema:
        return instance(defs[schema["$ref"].split("/")[-1]], defs)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return instance(options[0], defs)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type", "object")
    if kind == "object":
        props = schema.get("properties", {})
        return {k: instance(v, defs) for k, v in props.items()}
    if kind == "array":
        return [instance(schema.get("items", {"type": "string"}), defs)]
    if kind == "integer":
        return int(schema.get("minimum", 0))
    if kind == "number":
        return float(schema.get("minimum", 0))
    if kind == "boolean":
        return False
    return "x"


# ---- the rules ---------------------------------------------------------------------------

def last_input(text: str) -> dict:
    """The input block a Ragas prompt ends with ("input: {...}")."""
    at = text.rfind("input: {")
    if at < 0:
        return {}
    try:
        return json.JSONDecoder().raw_decode(text[at + len("input: "):])[0]
    except ValueError:
        return {}


def rating(title: str, text: str) -> dict:
    data = last_input(text)
    a_key, b_key = FIELDS[title]
    a, b = data.get(a_key, ""), data.get(b_key, "")
    if not b and title != "AnswerAccuracyOutput":      # some prompts name the fields differently
        values = [v for v in data.values() if isinstance(v, str)]
        a, b = (values + ["", ""])[:2]
    top, mid = TOP[title]
    share = overlap(a, b)
    return {"rating": top if share >= 0.5 else mid if share >= 0.2 else 0}


def section(text: str, start: str, end: str | None) -> str:
    part = text.split(start, 1)[-1]
    return part.split(end, 1)[0] if end and end in part else part


def judge_verdict(text: str) -> dict:
    passages = section(text, "PASSAGES:", "REPLY:")
    reply = section(text, "REPLY:", "Answer with")
    unsupported = (set(NUMBER.findall(reply)) - set(NUMBER.findall(passages))) | \
                  (set(CHUNK.findall(reply)) - set(CHUNK.findall(passages)))
    ok = not unsupported and overlap(reply, passages) >= 0.3
    return {"verdict": "pass" if ok else "fail",
            "reason": "every fact is in the passages" if ok else f"not in the passages: {sorted(unsupported)[:3]}"}


def judge_pair(text: str) -> dict:
    passages = section(text, "PASSAGES:", "REPLY A:")
    a = section(text, "REPLY A:", "REPLY B:")
    b = section(text, "REPLY B:", "Answer with")
    better = "B" if overlap(b, passages) > overlap(a, passages) else "A"
    return {"better": better, "reason": "more of it is in the passages"}


def m06_decide(req: dict) -> dict:
    msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
    text = "\n".join(m05.text_of(m) for m in msgs)
    if "Task: grade one reply." in text:
        return {"content": json.dumps(judge_verdict(text))}
    if "Task: compare two replies." in text:
        return {"content": json.dumps(judge_pair(text))}
    if 'Reply with JSON: {"word": "OK"}' in text:      # check 1's judge probe
        return {"content": json.dumps({"word": "OK"})}
    schema = schema_in(req, text)
    if schema and schema.get("title") not in (None, "Plan", "Grade", "SqlQuery"):
        title = schema.get("title")
        if title in TOP:
            return {"content": json.dumps(rating(title, text))}
        return {"content": json.dumps(instance(schema))}
    if req.get("model") == "llama3.2:1b" and "Write the reply to the customer" in text:
        user = "\n".join(m05.text_of(m) for m in msgs if m.get("role") == "user")
        if "Manual passages:\n(none)" in text:
            return {"content": REFUSAL}
        if int(hashlib.md5(user.encode()).hexdigest(), 16) % 3 == 0:
            return {"content": VAGUE}
    return m05_decide(req)


REFUSAL = "Sorry, I can only help with your orders and our products."
VAGUE = "Thanks for reaching out. Please have a look at the product manual; it explains this."
m05_decide = m05.decide
m05.decide = m06_decide           # M5's Handler calls the module-level decide()


# ---- the server --------------------------------------------------------------------------

class Handler(m05.Handler):
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "fake"}
                                                              for m in MODELS]})
        if path == "/api/tags":
            return self.send(200, {"models": [{"name": m, "model": m} for m in MODELS]})
        return super().do_GET()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(slots: int = 4) -> tuple[subprocess.Popen, str]:
    """Start the fake in a child process; return (process, root URL like http://127.0.0.1:PORT)."""
    port = free_port()
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--port", str(port), "--slots", str(slots)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("the fake server did not start")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=11999)
    ap.add_argument("--delay", type=float, default=0.01, help="seconds per chat request")
    ap.add_argument("--slots", type=int, default=4, help="chat requests served at the same time")
    a = ap.parse_args()
    m05.STATE.update(delay=a.delay, slots=threading.Semaphore(a.slots))
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
