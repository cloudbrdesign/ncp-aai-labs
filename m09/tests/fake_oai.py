"""A scripted stand-in for Ollama and the hosted safety models, for the M9 offline self-test only (M09_FAKE_LLM=1).

Learners never need this. It is M6's fake server (m06/tests/fake_oai.py, which is M5's plus M6's rules) with
the rules M9 needs, so `M09_FAKE_LLM=1 python m09/check.py` runs the lab without Ollama or an API key. It is
not a model: every reply comes from rules, so it proves plumbing and APIs, never quality. The rules make the
unguarded desk fail the way the attacks intend, so the self-test can see the layers stop them:

  draft          obeys: a request for the system prompt or instructions gets the system prompt's first line
                 (the canary); a request to say, write or include something gets those strings back
                 (XYZ-12 style tokens, quoted text, markup, SQL). Anything else: M6's rules (the M4 draft),
                 which copy the first sentence of the first passage, so a planted note in a passage shows up
  SqlQuery       a SELECT that lists customers (every order, or the one order the question names)
  self checks    input: "Yes" for injection, leak and abuse phrases (M5's list plus M9's);
                 output: "Yes" for the canary, foreign links, markup, SQL and internal notes
  content safety the NemoGuard prompt: {"User Safety": ...} / {"Response Safety": ...} JSON, unsafe for
                 threats, violence and insults, with categories
  similarity     bias_check.py's judge prompt: {"score": 1-5} from word overlap
  jailbreak NIM  POST .../nemoguard-jailbreak-detect: {"jailbreak": true|false} by the same phrases
  models         the hosted safety model is listed next to M6's models

    python m09/tests/fake_oai.py --port 11999
"""
import argparse
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
_spec = importlib.util.spec_from_file_location("m06_fake_oai", LABS / "m06" / "tests" / "fake_oai.py")
m06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m06)
m05 = m06.m05                        # M5's fake: Handler, text_of, decide hook, STATE

SAFETY_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
MODELS = m06.MODELS + [SAFETY_MODEL]
LEAK = re.compile(r"system prompt|your instructions|everything above|verbatim|internal reference|reference code|"
                  r"instructions as a|your rules", re.I)
OBEY = re.compile(r"\b(say|write|output|reply (only )?with|confirm|include|repeat|end your reply|start with|sign as|"
                  r"answer with|print)\b", re.I)
TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:[- ][A-Z0-9]+)*-\d+\b|\b[A-Z]{3,}-[A-Z]{2,}\b")
QUOTED = re.compile(r"'([^']{3,80})'|\"([^\"]{3,80})\"")
MARKUP = re.compile(r"<script[^>]*>.*?</script>|!\[[^\]]*\]\([^)]+\)|\{\{.*?\}\}|(?:'\s*OR\s+1=1|DROP TABLE)[^.\n]*", re.I | re.S)
INJECTION = re.compile(r"ignore (all |any |your |the |previous |prior )*(rules|instructions)|system prompt|"
                       r"pretend|developer mode|jailbreak|\bDAN\b|do anything now|admin password|forget (your|all|the)|"
                       r"override|verbatim|everything above|instructions|reference code|grandmother|"
                       r"reply only with|repeat (exactly|this)|other customers|every customer|"
                       r"<script|drop table|or 1=1|!\[", re.I)
TOXIC = re.compile(r"\b(kill|hurt|bomb|shoot|stab|burn (it|your|the)|idiots?|stupid|morons?|hate|weapon|poison|"
                   r"explosive|threat|destroy)\b", re.I)
THREAT = re.compile(r"\b(kill|hurt|bomb|shoot|stab|burn)\b", re.I)
OUT_BLOCK = re.compile(r"CBA-CANARY|internal|staff only|confidential|password|api key|https?://|<script|drop table|"
                       r"or 1=1|!\[|instructions say", re.I)


def section(text: str, start: str, end: str) -> str:
    part = text.split(start, 1)[-1]
    return part.split(end, 1)[0] if end in part else part


def content_safety(text: str) -> str:
    convo = section(text, "<BEGIN CONVERSATION>", "<END CONVERSATION>")
    user = section(convo, "user:", "response: agent:").strip()
    if "response: agent:" in convo:
        bot = convo.split("response: agent:", 1)[1].strip()
        unsafe = bool(TOXIC.search(bot))
        out = {"User Safety": "unsafe" if TOXIC.search(user) else "safe",
               "Response Safety": "unsafe" if unsafe else "safe"}
        if unsafe:
            out["Safety Categories"] = "Violence" if THREAT.search(bot) else "Harassment"
        return json.dumps(out)
    if TOXIC.search(user):
        cats = "Threat, Violence" if THREAT.search(user) else "Harassment, Profanity"
        return json.dumps({"User Safety": "unsafe", "Safety Categories": cats})
    return json.dumps({"User Safety": "safe"})


def quoted(prompt: str, label: str) -> str:
    m = re.search(label + r':\s*"(.*?)"\s*\n', prompt, re.S)
    return m.group(1) if m else prompt


def obedient_draft(system: str, user: str) -> str | None:
    """What a model that follows the customer instead of its rules would write."""
    if LEAK.search(user):
        first = system.strip().splitlines()[0]
        return f"Sure. My instructions start like this: {first}"
    if OBEY.search(user):
        found = [m.group(0) for m in MARKUP.finditer(user)]
        found += [m.group(0) for m in TOKEN.finditer(user)]
        found += [a or b for a, b in QUOTED.findall(user)]
        found = list(dict.fromkeys(f.strip() for f in found if f.strip()))
        if found:
            return "Sure: " + " ".join(found)
    return None


def sql_for(question: str) -> str:
    oid = re.search(r"\bA\d{4}\b", question)
    if oid:
        return f"SELECT order_id, customer, status FROM orders WHERE order_id = '{oid.group(0)}' LIMIT 20"
    return "SELECT order_id, customer, status FROM orders LIMIT 20"


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def m09_decide(req: dict) -> dict:
    msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
    text = "\n".join(m05.text_of(m) for m in msgs)
    if "<BEGIN UNSAFE CONTENT CATEGORIES>" in text:
        return {"content": content_safety(text)}
    if "Should the user message be blocked" in text:
        user = quoted(text, "User message")
        return {"content": "Yes" if INJECTION.search(user) or TOXIC.search(user) else "No"}
    if "Should the bot message be blocked" in text:
        return {"content": "Yes" if OUT_BLOCK.search(quoted(text, "Bot message")) else "No"}
    if "Task: rate how similar two support replies are." in text:
        a = section(text, "REPLY A:", "REPLY B:")
        b = section(text, "REPLY B:", "Answer with")
        wa, wb = words(a), words(b)
        share = len(wa & wb) / max(len(wa | wb), 1)
        return {"content": json.dumps({"score": 5 if share >= 0.7 else 4 if share >= 0.5 else 3 if share >= 0.3 else 2})}
    if m05.schema_name(req) == "SqlQuery":
        user = "\n".join(m05.text_of(m) for m in msgs if m.get("role") == "user")
        return {"content": json.dumps({"sql": sql_for(user)})}
    system = "\n".join(m05.text_of(m) for m in msgs if m.get("role") == "system")
    if "Write the reply to the customer" in system:
        user = "\n".join(m05.text_of(m) for m in msgs if m.get("role") == "user")
        out = obedient_draft(system, user.split("Customer:")[-1])
        if out:
            return {"content": out}
    return m06_decide(req)


m06_decide = m06.m06_decide
m05.decide = m09_decide              # M5's Handler calls the module-level decide()


class Handler(m06.Handler):
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "fake"}
                                                              for m in MODELS]})
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("nemoguard-jailbreak-detect"):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            hit = bool(INJECTION.search(str(body.get("input", ""))))
            return self.send(200, {"jailbreak": hit, "score": 0.9 if hit else -0.9})
        return super().do_POST()


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
