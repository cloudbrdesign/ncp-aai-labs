"""A scripted stand-in for Ollama (and, with --nim, for a NIM), for the offline self-test only.

Learners never need this. It lets `M05_FAKE_LLM=1 python m05/check.py` run the whole lab
on a machine without Ollama or a GPU. It is not a model: every reply comes from a few
rules below, so it proves plumbing and APIs, never quality.

    python m05/tests/fake_oai.py --port 11999          # looks like Ollama
    python m05/tests/fake_oai.py --port 18000 --nim    # also answers the NIM management endpoints

It speaks two APIs on one port:
  OpenAI-compatible   GET /v1/models, POST /v1/chat/completions (stream, response_format,
                      guided_json, tools), POST /v1/embeddings
  Ollama's own        POST /api/chat (what ChatOllama uses), POST /api/embed (what ollama.embed()
                      uses), GET /api/tags, GET /api/version
  --nim only          GET /v1/health/live, /v1/health/ready, /v1/metadata, /v1/metrics
                      (without --nim these return 404, as on Ollama)

The rules:
  desk tasks    the M4 fake (m04/tests/fake_llm.py): router plan, groundedness grade, draft
  self checks   input: "Yes" (block) for injection phrases; output: "Yes" for internal notes;
                facts: "no" when the reply names an error code or order ID the evidence lacks
  JSON mode     {"product": ..., "error_code": ...} read from the question
  tools         a call to the first tool, with the order ID from the question
  embeddings    M4's hashed bag of words (shared words = similar vectors)
"""
import argparse
import json
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent.parent
for p in (LABS / "m04" / "tests", LABS / "m04", LABS / "m03", LABS / "setup"):
    sys.path.append(str(p))

from pydantic import BaseModel  # noqa: E402

import fake_llm  # noqa: E402   m04/tests/fake_llm.py

INJECTION = re.compile(r"ignore (all |any |your |the |previous |prior )*(rules|instructions)|system prompt|"
                       r"pretend (you|to be)|developer mode|jailbreak|admin password|forget (your|all)", re.I)
INTERNAL = re.compile(r"internal|staff only|confidential|password|api key", re.I)
CODES = re.compile(r"\b(E\d\d|A\d{4})\b")
LONG_ANSWER = ("The D300 dock drives two external displays over DisplayPort and charges the laptop "
               "through the same USB-C cable when the power adapter is connected.")
STATE = {"nim": False, "delay": 0.01, "requests": 0, "slots": None}


class Grade(BaseModel):      # same shape as m03/critic.py's Grade
    score: int
    reason: str


class SqlQuery(BaseModel):   # same shape as m04/orders_sql.py's SqlQuery
    sql: str


# ---- the rules ----------------------------------------------------------------------

def text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def schema_name(req: dict) -> str | None:
    """The structured-output schema the client asked for, from any of the request shapes."""
    fmt = req.get("format")                                   # Ollama /api/chat
    if isinstance(fmt, dict):
        return fmt.get("title", "json")
    if fmt == "json":
        return "json"
    rf = req.get("response_format") or {}                     # OpenAI
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", {}).get("title") or js.get("name")
    if rf.get("type") == "json_object":
        return "json"
    guided = req.get("guided_json") or (req.get("nvext") or {}).get("guided_json")   # ChatNVIDIA
    if isinstance(guided, dict):
        return guided.get("title", "json")
    return None


def quoted(prompt: str, label: str) -> str:
    m = re.search(label + r':\s*"(.*?)"\s*\n', prompt, re.S)
    return m.group(1) if m else prompt


def self_check(prompt: str) -> str | None:
    """The three Guardrails self-check prompts in m05/guardrails/prompts.yml."""
    if "Should the user message be blocked" in prompt:
        return "Yes" if INJECTION.search(quoted(prompt, "User message")) else "No"
    if "Should the bot message be blocked" in prompt:
        return "Yes" if INTERNAL.search(quoted(prompt, "Bot message")) else "No"
    if '"entails":' in prompt:
        evidence = prompt.split('"evidence":', 1)[1].split('"hypothesis":', 1)[0]
        hypothesis = prompt.split('"hypothesis":', 1)[1].split('"entails":', 1)[0]
        return "no" if set(CODES.findall(hypothesis)) - set(CODES.findall(evidence)) else "yes"
    return None


def structured(name: str, pairs: list[tuple[str, str]]) -> str:
    if name == "Plan":
        import router   # m04/router.py (the keyword router stands in for the model's plan)
        return fake_llm.chat(pairs, router.Plan).model_dump_json()
    if name == "Grade":
        return fake_llm.chat(pairs, Grade).model_dump_json()
    if name == "SqlQuery":
        return fake_llm.chat(pairs, SqlQuery).model_dump_json()
    last = pairs[-1][1] if pairs else ""
    product = re.search(r"\b(H200|D300|M270)\b", last)
    code = re.search(r"\bE\d\d\b", last)
    return json.dumps({"product": product.group(1) if product else "", "error_code": code.group(0) if code else ""})


def decide(req: dict) -> dict:
    """Return {"content": text} or {"tool": name, "args": {...}} for one chat request."""
    msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
    pairs = [(m.get("role", "user"), text_of(m)) for m in msgs]
    everything = "\n".join(t for _, t in pairs)
    checked = self_check(everything)
    if checked is not None:
        return {"content": checked}
    tools = req.get("tools")
    if tools and not any(r == "tool" for r, _ in pairs):
        oid = re.search(r"\bA\d{4}\b", everything)
        return {"tool": tools[0]["function"]["name"], "args": {"order_id": oid.group(0) if oid else "A1003"}}
    name = schema_name(req)
    if name:
        return {"content": structured(name, pairs)}
    out = fake_llm.chat(pairs)
    if out == "OK" and "single word OK" not in everything:
        out = LONG_ANSWER
    return {"content": out}


def vector(text: str) -> list[float]:
    return fake_llm.embed([str(text)])[0]


def words(n_text: str) -> int:
    return max(1, len(n_text.split()))


# ---- the HTTP server ----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code: int, body, ctype: str = "application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream_start(self, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def chunk(self, data: str):
        b = data.encode()
        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def stream_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "fake"}
                                                              for m in ("llama3.2:3b", "embeddinggemma",
                                                                        "meta/llama-3.1-8b-instruct")]})
        if path == "/api/tags":
            return self.send(200, {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b"},
                                              {"name": "embeddinggemma:latest", "model": "embeddinggemma:latest"}]})
        if path == "/api/version":
            return self.send(200, {"version": "0.0.0-fake"})
        if path in ("", "/"):
            return self.send(200, "Ollama is running (fake)", "text/plain")
        if STATE["nim"]:
            if path in ("/v1/health/live", "/v1/health/ready"):
                return self.send(200, {"object": "health.response", "message": "fake: ready"})
            if path == "/v1/metadata":
                return self.send(200, {"note": "scripted stand-in for the self-test, not a NIM",
                                       "profile": {"name": "fake-bf16-tp1", "id": "0" * 8}})
            if path == "/v1/metrics":
                return self.send(200, f"# fake metrics\nfake_requests_total {STATE['requests']}\n", "text/plain")
        return self.send(404, "404 page not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        path = self.path.split("?")[0].rstrip("/")
        if path == "/api/embed":
            inp = req.get("input")
            inp = [inp] if isinstance(inp, str) else inp
            return self.send(200, {"model": req.get("model"), "embeddings": [vector(t) for t in inp]})
        if path == "/v1/embeddings":
            inp = req.get("input")
            inp = [inp] if isinstance(inp, str) else inp
            data = [{"object": "embedding", "index": i, "embedding": vector(t)} for i, t in enumerate(inp)]
            return self.send(200, {"object": "list", "data": data, "model": req.get("model"),
                                   "usage": {"prompt_tokens": len(inp), "total_tokens": len(inp)}})
        if path not in ("/api/chat", "/v1/chat/completions"):
            return self.send(404, "404 page not found", "text/plain")
        STATE["requests"] += 1
        with STATE["slots"]:            # like OLLAMA_NUM_PARALLEL: requests beyond the slots wait
            time.sleep(STATE["delay"])
            out = decide(req)
        prompt_tokens = sum(words(text_of(m)) for m in req.get("messages", []))
        if path == "/api/chat":
            return self.ollama_chat(req, out, prompt_tokens)
        return self.openai_chat(req, out, prompt_tokens)

    def ollama_chat(self, req, out, prompt_tokens):
        model, now = req.get("model"), time.strftime("%Y-%m-%dT%H:%M:%SZ")
        done = {"model": model, "created_at": now, "done": True, "done_reason": "stop",
                "prompt_eval_count": prompt_tokens, "eval_count": words(out.get("content", "x")),
                "total_duration": 1, "load_duration": 1, "prompt_eval_duration": 1, "eval_duration": 1}
        if "tool" in out:
            message = {"role": "assistant", "content": "",
                       "tool_calls": [{"function": {"name": out["tool"], "arguments": out["args"]}}]}
        else:
            message = {"role": "assistant", "content": out["content"]}
        if req.get("stream", True) is False:
            return self.send(200, {**done, "message": message})
        self.stream_start("application/x-ndjson")
        pieces = re.findall(r"\S+\s*", message["content"]) if message["content"] else []
        for p in pieces:
            self.chunk(json.dumps({"model": model, "created_at": now, "done": False,
                                   "message": {"role": "assistant", "content": p}}) + "\n")
        if "tool_calls" in message:
            self.chunk(json.dumps({"model": model, "created_at": now, "done": False, "message": message}) + "\n")
        self.chunk(json.dumps({**done, "message": {"role": "assistant", "content": ""}}) + "\n")
        self.stream_end()

    def openai_chat(self, req, out, prompt_tokens):
        cid, model, created = "chatcmpl-" + uuid.uuid4().hex[:8], req.get("model"), int(time.time())
        completion = words(out.get("content", "x"))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                 "total_tokens": prompt_tokens + completion}
        if "tool" in out:
            calls = [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                      "function": {"name": out["tool"], "arguments": json.dumps(out["args"])}}]
            message, finish = {"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": out["content"]}, "stop"
        if not req.get("stream"):
            return self.send(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                   "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                   "usage": usage})
        self.stream_start("text/event-stream")
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        if "tool" in out:
            deltas = [{"role": "assistant", "tool_calls": [{**calls[0], "index": 0}]}]
        else:
            deltas = [{"role": "assistant", "content": p} for p in re.findall(r"\S+\s*", out["content"])]
        for d in deltas:
            self.chunk("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}) + "\n\n")
        self.chunk("data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n")
        if (req.get("stream_options") or {}).get("include_usage"):
            self.chunk("data: " + json.dumps({**base, "choices": [], "usage": usage}) + "\n\n")
        self.chunk("data: [DONE]\n\n")
        self.stream_end()


# ---- start and stop ------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(nim: bool = False, slots: int = 4) -> tuple[subprocess.Popen, str]:
    """Start the fake in a child process; return (process, root URL like http://127.0.0.1:PORT)."""
    port = free_port()
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--port", str(port), "--slots", str(slots)]
    proc = subprocess.Popen(cmd + (["--nim"] if nim else []), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    ap.add_argument("--nim", action="store_true", help="also answer the NIM health, metadata and metrics endpoints")
    ap.add_argument("--delay", type=float, default=0.01, help="seconds per chat request")
    ap.add_argument("--slots", type=int, default=4, help="chat requests served at the same time")
    a = ap.parse_args()
    STATE.update(nim=a.nim, delay=a.delay, slots=threading.Semaphore(a.slots))
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
