"""Check that the M02 lab works end to end.

    # terminal 1: the ticket API
    python m02/ticket_api.py
    # terminal 2: the MCP server with the desk's tools
    nat mcp serve --config_file m02/configs/ticket_tools.yml \
      --tool_names lookup_order --tool_names create_ticket --tool_names get_ticket
    # terminal 3:
    python m02/check.py

1. The prompt chain sorts a damaged-item message into the right category (2.1).
2. The MCP server publishes the three desk tools, and the agent opens a ticket through it (2.2).
3. The vision model reads the order number off the shipping-label photo (2.3).
4. The circuit breaker opens after repeated failures and recovers after the cooldown (2.4).
5. `nat serve` streams the agent's steps and its answer (2.5).
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time

import httpx

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MCP_URL = os.environ.get("DESK_MCP_URL", "http://localhost:9901/mcp")
TICKET_API = os.environ.get("TICKET_API_URL", "http://localhost:8765")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)


def run(cmd: list[str], timeout=600) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def main():
    # 1. prompt chain
    import triage_chain
    r = triage_chain.run("My USB-C dock from order A1002 arrived cracked.")
    check("Prompt chain sorted the message as damaged_item", r["category"] == "damaged_item", r)

    # 2. MCP
    try:
        httpx.get(f"{TICKET_API}/health", timeout=5).raise_for_status()
    except httpx.HTTPError as e:
        check("Ticket API is running", False, f"{e}. Start it: python m02/ticket_api.py")
        return
    listed = run(["nat", "mcp", "client", "tool", "list", "--url", MCP_URL])
    tools = {t for t in ("lookup_order", "create_ticket", "get_ticket") if f"\n{t}\n" in f"\n{listed}\n"}
    check("MCP server publishes lookup_order, create_ticket, get_ticket", len(tools) == 3,
          f"found {sorted(tools)}. Start it: nat mcp serve --config_file m02/configs/ticket_tools.yml "
          "--tool_names lookup_order --tool_names create_ticket --tool_names get_ticket")
    log = run(["nat", "run", "--config_file", str(HERE / "configs" / "support_mcp.yml"),
               "--input", "My USB-C dock from order A1002 arrived cracked. Please help."])
    check("Agent opened a ticket through MCP (desk__create_ticket)", "Calling tools: desk__create_ticket" in log,
          log[-400:])
    check("Agent produced a final answer", "Workflow Result" in log, log[-400:])

    # 3. vision
    import photo_question
    try:
        answer = photo_question.ask_vision_model(HERE / "data" / "shipping_label.jpg")
        check("Vision model read order A1002 from the label photo", "A1002" in answer.upper(), answer)
    except Exception as e:
        check("Vision model read order A1002 from the label photo", False,
              f"{e}. Pull the model first: ollama pull {os.environ.get('VLM_MODEL', 'qwen3-vl:8b')}")

    # 4. circuit breaker (its own ticket API on another port, failing twice)
    api = subprocess.Popen([sys.executable, str(HERE / "ticket_api.py"), "--port", "8766", "--fail-first", "2"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        os.environ["TICKET_API_URL"] = "http://localhost:8766"
        import breaker_demo
        lines = []
        breaker_demo.print = lambda *a, **k: lines.append(" ".join(str(x) for x in a))  # capture its output
        asyncio.run(breaker_demo.main())
        text = "\n".join(lines)
        check("Circuit breaker short-circuited call 3", "call 3" in text and "CircuitBreakerOpenError" in text, text)
        check("Circuit breaker recovered after the cooldown (calls 4 and 5 OK)", text.count("-> OK") == 2, text)
    finally:
        api.terminate()
        os.environ["TICKET_API_URL"] = TICKET_API

    # 5. streaming
    server = subprocess.Popen(["nat", "serve", "--config_file", str(HERE / "configs" / "support_desk.yml"),
                               "--port", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        import stream_client
        s = stream_client.stream("What is the status of order A1003?", show=False)
        check("Streaming endpoint sent the agent's steps (lookup_order)", "lookup_order" in s["tools"], json.dumps(s))
        check("Streaming endpoint sent the answer text", bool(s["answer"].strip()), json.dumps(s))
    finally:
        server.terminate()


if __name__ == "__main__":
    print("NCP-AAI M02 lab check\n")
    main()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
