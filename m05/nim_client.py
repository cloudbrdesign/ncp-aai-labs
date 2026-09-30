"""Probe an OpenAI-compatible server: is it a NIM, and what does its API do?

    python m05/nim_client.py                                                   # Ollama (free mode)
    python m05/nim_client.py --base-url http://localhost:8000/v1 --model meta/llama-3.1-8b-instruct   # the NIM

Six probes, in this order:
  1. GET the NIM management endpoints: /v1/health/live (container up), /v1/health/ready
     (model loaded), /v1/metadata (active profile). Ollama doesn't have them: that is how
     the script tells a NIM from "any OpenAI-compatible server".
  2. GET /v1/models: the model names the server accepts.
  3. A chat completion (POST /v1/chat/completions).
  4. The same, streamed: the reply arrives in chunks.
  5. JSON mode: response_format {"type": "json_object"}; the reply must parse as JSON.
  6. A tool call: one tool in `tools` (no tool_choice: Ollama doesn't support that field).
     A NIM only returns tool calls when it was started with --enable-auto-tool-choice
     and a --tool-call-parser (llama3_json for Llama 3.1).

The same `openai` client does all of it; only the base URL and the model name change.
"""
import argparse
import json
import os
import pathlib
import sys

import httpx
from openai import OpenAI

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402

MANAGEMENT = ["health/live", "health/ready", "metadata"]
TOOLS = [{"type": "function", "function": {
    "name": "order_status", "description": "Look up the status of one order by its ID.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "like A1003"}},
                   "required": ["order_id"]}}}]


def defaults() -> tuple[str, str]:
    """The base URL and model of the configured chat provider (Ollama's /v1, or the NIM)."""
    return llm_calls.chat_base_url(), llm_calls.model_name()


def find_profile(meta) -> list[str]:
    """Every value under a key that mentions 'profile' (the metadata layout is the NIM's, not ours)."""
    found = []
    if isinstance(meta, dict):
        for k, v in meta.items():
            if "profile" in k.lower():
                found.append(json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            else:
                found += find_profile(v)
    elif isinstance(meta, list):
        for v in meta:
            found += find_profile(v)
    return found


def probe(base_url: str, model: str, say=print) -> dict:
    base_url = base_url.rstrip("/")
    r = {"base_url": base_url, "model": model, "endpoints": {}}
    client = OpenAI(base_url=base_url, api_key=os.environ.get("NIM_API_KEY", "not-needed"), timeout=120)

    for path in MANAGEMENT:
        try:
            resp = httpx.get(f"{base_url}/{path}", timeout=10)
            r["endpoints"][path] = resp.status_code
            if path == "metadata" and resp.status_code == 200:
                r["metadata"] = resp.json()
                r["profile"] = find_profile(r["metadata"])
        except httpx.HTTPError as e:
            r["endpoints"][path] = f"{type(e).__name__}"
        say(f"[nim] GET /v1/{path:<13} -> {r['endpoints'][path]}")
    r["is_nim"] = all(r["endpoints"][p] == 200 for p in MANAGEMENT)
    if r["is_nim"]:
        say(f"[nim] NIM management endpoints answer: live, ready, metadata. Active profile: "
            f"{'; '.join(r.get('profile', [])) or '(no profile field found; see --show-metadata)'}")
    else:
        say("[nim] not a NIM: the management endpoints are missing (Ollama serves the OpenAI API, not these)")

    r["models"] = [m.id for m in client.models.list().data]
    say(f"[api] GET /v1/models -> {', '.join(r['models'])}")

    msgs = [{"role": "user", "content": "Reply with the single word OK."}]
    reply = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=10)
    r["chat"] = reply.choices[0].message.content or ""
    say(f"[api] chat completion -> {r['chat']!r} ({reply.usage.completion_tokens if reply.usage else '?'} tokens)")

    chunks = [c.choices[0].delta.content or "" for c in client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Name the three colours of a traffic light."}],
        temperature=0, max_tokens=40, stream=True) if c.choices]
    r["stream_chunks"] = sum(1 for c in chunks if c)
    say(f"[api] streamed -> {r['stream_chunks']} chunks: {''.join(chunks)!r}")

    ask = ('Return a JSON object with the keys "product" and "error_code" for this message: '
           '"My D300 dock shows E42."')
    raw = client.chat.completions.create(model=model, messages=[{"role": "user", "content": ask}], temperature=0,
                                         max_tokens=60, response_format={"type": "json_object"})
    text = raw.choices[0].message.content or ""
    try:
        r["json"] = json.loads(text)
        say(f"[api] JSON mode -> parsed: {r['json']}")
    except json.JSONDecodeError:
        r["json"] = None
        say(f"[api] JSON mode -> not valid JSON: {text[:80]!r}")

    try:
        tc = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Where is order A1003?"}], tools=TOOLS, temperature=0)
        calls = tc.choices[0].message.tool_calls or []
        r["tool_calls"] = [{"name": c.function.name, "arguments": c.function.arguments} for c in calls]
        say(f"[api] tool call -> {r['tool_calls'] or 'no tool call, text: ' + repr((tc.choices[0].message.content or '')[:60])}")
    except Exception as e:   # a NIM started without the tool-calling flags answers 400
        r["tool_calls"] = []
        r["tool_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        say(f"[api] tool call -> error {r['tool_error']}")
    return r


def main():
    base, model = defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=base, help=f"OpenAI-compatible base URL (default {base})")
    ap.add_argument("--model", default=model, help=f"model name as the server knows it (default {model})")
    ap.add_argument("--show-metadata", action="store_true", help="print /v1/metadata in full")
    ap.add_argument("--json", action="store_true", help="print the results as JSON at the end")
    a = ap.parse_args()
    print(f"[INFO] probing {a.base_url} with model {a.model}")
    r = probe(a.base_url, a.model)
    if a.show_metadata and r.get("metadata"):
        print(json.dumps(r["metadata"], indent=2))
    if a.json:
        print(json.dumps({k: v for k, v in r.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
