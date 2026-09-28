"""A dynamic prompt chain: classify the message, then answer with that category's prompt.

    python m02/triage_chain.py "My USB-C dock from order A1002 arrived cracked."
    python m02/triage_chain.py --all          # the four sample messages below

Step 1 uses prompts/triage/classify.txt to pick a category. Step 2 loads the template
for that category (prompts/triage/<category>.txt), fills in the order ID and the
message, and asks the model for the reply. The prompts live in files, so you can
change the wording without touching this code.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "setup"))
import llm  # noqa: E402  (setup/llm.py picks NVIDIA or Ollama)

PROMPTS = HERE / "prompts" / "triage"
CATEGORIES = ["order_status", "return", "damaged_item", "other"]
SAMPLES = [
    "Where is my order A1001? It should have arrived by now.",
    "Can I send back the monitor from order A1003 and get my money back?",
    "My USB-C dock from order A1002 arrived cracked.",
    "Do you have gift cards?",
]


def classify(model, message: str) -> str:
    reply = model.invoke([("system", (PROMPTS / "classify.txt").read_text()), ("user", message)]).content
    word = reply.strip().lower()
    # small models sometimes add words around the label; take the first known category
    return next((c for c in CATEGORIES if c in word), "other")


def answer(model, category: str, message: str) -> str:
    order_id = (re.findall(r"\b[Aa]\d{4}\b", message) or ["(no order ID given)"])[0].upper()
    prompt = (PROMPTS / f"{category}.txt").read_text().format(order_id=order_id, message=message)
    return model.invoke([("user", prompt)]).content.strip()


def run(message: str, model=None) -> dict:
    model = model or llm.get_llm(temperature=0)
    category = classify(model, message)
    return {"message": message, "category": category, "reply": answer(model, category, message)}


def main():
    msgs = SAMPLES if sys.argv[1:] == ["--all"] else [" ".join(sys.argv[1:]) or SAMPLES[2]]
    print(f"[model: {llm.provider()} {llm.model_name()}]")
    model = llm.get_llm(temperature=0)
    for m in msgs:
        r = run(m, model)
        print(f"\nCustomer: {m}\nStep 1, category: {r['category']}\nStep 2, reply: {r['reply']}")


if __name__ == "__main__":
    main()
