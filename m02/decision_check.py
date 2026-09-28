"""Does the agent pick the right tool? Score two prompt versions on the same 10 questions.

    # terminal 1: the ticket API (the tools call it)
    python m02/ticket_api.py
    # terminal 2:
    python m02/decision_check.py

For each config it starts `nat serve`, asks the 10 questions below through the streaming
endpoint, records which tool the agent started first, and compares that with the expected
tool. v1 has a bare prompt; v2 adds the support-desk rules (prompts/support_rules.md).
Same model, same tools: only the prompt changes.
"""
import os
import pathlib
import subprocess
import sys
import time

import httpx

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_client  # noqa: E402

CASES = [  # (question, the tool the agent should use first; None = no tool)
    ("Where is order A1001?", "lookup_order"),
    ("Has order A1003 shipped yet?", "lookup_order"),
    ("When will my order A1001 arrive?", "lookup_order"),
    ("My USB-C dock from order A1002 arrived cracked.", "create_ticket"),
    ("Order A1003 came with the power cable missing.", "create_ticket"),
    ("I got the wrong item in order A1001.", "create_ticket"),
    ("What's the status of ticket T-1001?", "get_ticket"),
    ("Any update on my support ticket T-1002?", "get_ticket"),
    ("Is ticket T-1001 still open?", "get_ticket"),
    ("Do you sell gift cards?", None),
]
VERSIONS = [("v1 bare prompt", "support_desk_v1.yml", 8001), ("v2 with rules", "support_desk.yml", 8002)]


def serve(config: str, port: int) -> subprocess.Popen:
    p = subprocess.Popen(["nat", "serve", "--config_file", str(HERE / "configs" / config), "--port", str(port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(1)
    p.kill()
    raise RuntimeError(f"nat serve for {config} did not start on port {port}")


def score(label: str, config: str, port: int) -> float:
    print(f"\n== {label} ({config})")
    server = serve(config, port)
    stream_client.URL = f"http://localhost:{port}/generate/stream"
    right = 0
    try:
        for question, expected in CASES:
            try:
                tools = stream_client.stream(question, show=False)["tools"]
            except httpx.HTTPError as e:
                tools = [f"error: {e}"]
            first = tools[0] if tools else None
            ok = first == expected
            right += ok
            print(f"  [{'ok ' if ok else 'BAD'}] {question:<50} expected {expected or '(no tool)':<14} got {first or '(no tool)'}",
                  flush=True)
    finally:
        server.terminate()
        server.wait(timeout=30)
    print(f"  tool-selection accuracy: {right}/{len(CASES)}")
    return right / len(CASES)


def main():
    results = {label: score(label, config, port) for label, config, port in VERSIONS}
    print("\nSummary: " + ", ".join(f"{k} {v:.0%}" for k, v in results.items()))
    return results


if __name__ == "__main__":
    os.chdir(HERE.parent)
    main()
