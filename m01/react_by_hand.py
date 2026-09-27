"""ReAct by hand: the reason -> act -> observe loop in plain Python.

    python m01/react_by_hand.py "Where is order A1001?"

The model writes a Thought and an Action; we run the Action (a Python function),
append the Observation, and ask the model to continue, until it writes a Final Answer.
The model comes from setup/llm.py, so this runs on NVIDIA's API catalog or local Ollama.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "setup"))
import llm  # noqa: E402

ORDERS = json.loads((pathlib.Path(__file__).parent / "data" / "orders.json").read_text())


def lookup_order(order_id: str) -> str:
    """Look up a customer order by its ID (for example A1001) and return its status."""
    order = ORDERS.get(order_id.strip().strip("'\"").upper())
    return json.dumps(order) if order else f"No order found with ID {order_id}"


TOOLS = {"lookup_order": lookup_order}

PROMPT = """You are a support-desk agent for an online electronics shop.
Answer the customer's question. You can use these tools:

{tools}

Use exactly this format:

Thought: what you need to do next
Action: the tool name, one of [{names}]
Action Input: the input for the tool
Observation: the tool's result (this is written for you; never write it yourself)
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the answer
Final Answer: the answer for the customer

Question: {question}
"""

ACTION = re.compile(r"Action:\s*(\w+)\s*\n\s*Action Input:\s*(.+)", re.IGNORECASE)
FINAL = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def run(question: str, max_steps: int = 5, show=print) -> dict:
    tools = "\n".join(f"- {n}: {f.__doc__}" for n, f in TOOLS.items())
    transcript = PROMPT.format(tools=tools, names=", ".join(TOOLS), question=question)
    model = llm.get_llm(stop=["Observation:"])     # stop so the model can't invent tool results
    calls = []
    for step in range(1, max_steps + 1):
        reply = model.invoke(transcript).content.strip()
        show(f"--- step {step}\n{reply}")
        transcript += reply + "\n"
        final = FINAL.search(reply)
        if final:
            return {"answer": final.group(1).strip(), "calls": calls, "steps": step}
        action = ACTION.search(reply)
        if not action:
            transcript += "Thought: I must use the format above: an Action or a Final Answer.\n"
            continue
        name, arg = action.group(1).strip(), action.group(2).strip().splitlines()[0]
        result = TOOLS[name](arg) if name in TOOLS else f"Unknown tool {name}"
        calls.append((name, arg))
        show(f"Observation: {result}")
        transcript += f"Observation: {result}\n"
    return {"answer": None, "calls": calls, "steps": max_steps}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Where is my order A1001, and when will it arrive?"
    print(f"Question: {q}  [{llm.provider()}: {llm.model_name()}]\n")
    out = run(q)
    print(f"\nFinal answer: {out['answer']}\nTool calls: {out['calls']}")
