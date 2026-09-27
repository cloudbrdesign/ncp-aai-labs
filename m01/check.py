"""Check that the M01 lab works end to end.

    # terminal 1: serve the returns agent over A2A
    nat a2a serve --config_file m01/configs/returns_agent.yml --port 11000 --name returns_agent
    # terminal 2:
    python m01/check.py

1. The hand-built ReAct loop calls the lookup_order tool and reaches a Final Answer.
2. The NeMo Agent Toolkit ReAct agent (support_agent.yml) answers using lookup_order.
3. The returns agent's A2A Agent Card is published.
4. The support agent hands a returns question to the returns agent over A2A.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RETURNS_URL = os.environ.get("RETURNS_AGENT_URL", "http://localhost:11000")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""))


def nat_run(config: str, question: str) -> str:
    p = subprocess.run(["nat", "run", "--config_file", str(HERE / "configs" / config), "--input", question],
                       capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr


def main():
    import react_by_hand
    out = react_by_hand.run("Where is order A1001?", show=lambda *_: None)
    check("Hand-built ReAct loop called lookup_order",
          any(n == "lookup_order" for n, _ in out["calls"]), f"tool calls: {out['calls']}")
    check("Hand-built ReAct loop reached a Final Answer", bool(out["answer"]),
          "The model never wrote 'Final Answer:'. Try again, or try another model.")

    log = nat_run("support_agent.yml", "What is the status of order A1003?")
    check("NAT ReAct agent used lookup_order", "Calling tools: lookup_order" in log, log[-400:])
    check("NAT ReAct agent produced a final answer", "Workflow Result" in log, log[-400:])

    try:
        card = json.load(urllib.request.urlopen(f"{RETURNS_URL}/.well-known/agent-card.json", timeout=10))
        check(f"Returns agent's Agent Card is published ({card.get('name')})", bool(card.get("name")))
    except Exception as e:
        check("Returns agent's Agent Card is published", False,
              f"{e}. Start it first: nat a2a serve --config_file m01/configs/returns_agent.yml "
              "--port 11000 --name returns_agent")
        return

    log = nat_run("support_with_returns.yml",
                  "Order A1002 was delivered 3 days ago. Can I still return it, and how soon is the refund?")
    check("Support agent called the returns agent over A2A", "Calling tools: returns_agent__call" in log, log[-400:])
    check("Support agent produced a final answer", "Workflow Result" in log, log[-400:])


if __name__ == "__main__":
    print("NCP-AAI M01 lab check\n")
    main()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
