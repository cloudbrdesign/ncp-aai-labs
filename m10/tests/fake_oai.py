"""A scripted stand-in for Ollama, for the M10 offline self-test only (M10_FAKE_LLM=1).

Learners never need this. It is Module 9's fake server (m09/tests/fake_oai.py: M5's + M6's + M9's rules)
with the two rules Module 10 adds, so `M10_FAKE_LLM=1 python m10/check.py` runs without Ollama. Every reply
comes from rules: it proves the plumbing and the APIs, never the quality of a model's decision.

  RefundProposal   the structured proposal oversight_desk.py asks for: the order from the message and the
                   most that can be refunded from the prompt; the unit price when the customer asks for
                   "one of" the items (a partial refund)
  tools            hitl_middleware_demo.py's agent: a refund request calls issue_refund (order, the order's
                   value from data/prices.json and M4's orders.db, a reason); a status question calls
                   order_status; after a tool result it repeats the result in one sentence
Everything else goes to Module 9's rules.

    python m10/tests/fake_oai.py --port 11999
"""
import argparse
import importlib.util
import json
import pathlib
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("m09_fake_oai", LABS / "m09" / "tests" / "fake_oai.py")
m09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m09)
m05 = m09.m05
PRICES = json.loads((LABS / "m10" / "data" / "prices.json").read_text())["unit_price"]
REFUND = re.compile(r"refund|money back|send (it )?back", re.I)
STATUS = re.compile(r"where|status|shipped|arriv", re.I)


def order_value(oid: str) -> float:
    con = sqlite3.connect(LABS / "m04" / "data" / "orders.db")
    row = con.execute("SELECT product_code, qty FROM order_items WHERE order_id = ?", (oid,)).fetchone()
    con.close()
    return round(PRICES[row[0]] * row[1], 2) if row else 0.0


def proposal(text: str) -> str:
    customer = text.split("Customer's message:", 1)[-1]
    oid = (re.findall(r"\bA\d{4}\b", customer) or re.findall(r"\bA\d{4}\b", text) or ["A1002"])[0]
    most = re.search(r"most that can be refunded: [A-Z]{3} ([\d,.]+)", text, re.I)
    unit = re.search(r"unit price: [A-Z]{3} ([\d,.]+)", text, re.I)
    amount = float((most.group(1) if most else "0").replace(",", ""))
    if unit and re.search(r"\bone of\b|\bone (headset|dock|monitor)\b", customer, re.I):
        amount = float(unit.group(1).replace(",", ""))
    return json.dumps({"order_id": oid, "amount": amount,
                       "reason": "Customer asked for a refund: " + " ".join(customer.split())[:80]})


def m10_decide(req: dict) -> dict:
    msgs = req.get("messages") or []
    text = "\n".join(m05.text_of(m) for m in msgs)
    if m05.schema_name(req) == "RefundProposal":
        return {"content": proposal(text)}
    tools = [t["function"]["name"] for t in req.get("tools") or []]
    if tools:
        done = [m for m in msgs if m.get("role") == "tool"]
        if done:
            return {"content": "Update on your request: " + m05.text_of(done[-1])[:300]}
        user = "\n".join(m05.text_of(m) for m in msgs if m.get("role") == "user")
        oid = (re.findall(r"\bA\d{4}\b", user) or ["A1002"])[-1]
        if "issue_refund" in tools and REFUND.search(user):
            return {"tool": "issue_refund", "args": {"order_id": oid, "amount": order_value(oid),
                                                     "reason": "customer asked for a refund"}}
        if "order_status" in tools and STATUS.search(user):
            return {"tool": "order_status", "args": {"order_id": oid}}
        return {"content": "I can check an order's status or start a refund. Which order is it?"}
    return m09_decide(req)


m09_decide = m09.m09_decide
m05.decide = m10_decide              # M5's Handler calls the module-level decide()
Handler = m09.Handler


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
