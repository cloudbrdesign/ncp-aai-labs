"""Stream an answer from the support desk, printing each step and each piece of text as it arrives.

    # terminal 1
    nat serve --config_file m02/configs/support_desk.yml --port 8000
    # terminal 2
    python m02/stream_client.py "What is the status of order A1003?"

The server's /generate/stream endpoint sends server-sent events: `intermediate_data:`
lines for the agent's steps (tool starts and ends) and `data:` lines with the answer text.
"""
import json
import os
import sys
import time

import httpx

URL = os.environ.get("DESK_URL", "http://localhost:8000") + "/generate/stream"


def stream(question: str, show=True) -> dict:
    """Return the tools the agent started and the streamed answer text."""
    tools, text, t0, first = [], "", time.monotonic(), None
    with httpx.stream("POST", URL, json={"input_message": question}, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("intermediate_data:"):
                name = json.loads(line.split(":", 1)[1]).get("name", "")
                if name.startswith("Function Start:") and "<workflow>" not in name:
                    tools.append(name.split(":", 1)[1].strip())
                    if show:
                        print(f"\n[{time.monotonic() - t0:5.1f}s] step: {name}", flush=True)
            elif line.startswith("data:"):
                chunk = json.loads(line.split(":", 1)[1]).get("value", "")
                if first is None:
                    first = time.monotonic() - t0
                    if show:
                        print(f"[{first:5.1f}s] answer: ", end="", flush=True)
                text += chunk
                if show:
                    print(chunk, end="", flush=True)
    if show:
        print(f"\n[{time.monotonic() - t0:5.1f}s] done")
    return {"tools": tools, "answer": text, "first_token_s": first}


if __name__ == "__main__":
    stream(" ".join(sys.argv[1:]) or "What is the status of order A1003?")
