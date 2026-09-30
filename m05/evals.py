"""Deterministic scores for `nat eval` (configs/desk_eval*.yml): no model judges the desk.

NAT's `langsmith_custom` evaluator imports these functions by dotted path (evals.has_fact)
and calls them with inputs (the question), outputs (the desk's reply) and
reference_outputs (the dataset's `answer`). `extra_fields` in the config passes more
fields from data/eval.json by name (`keywords`, `product`). Each returns a score 0 to 1.

    has_fact           share of the item's keywords that appear in the reply
                       (off-topic and injection items expect a refusal: "sorry")
    cites_right_manual every [chunk ID] in the reply comes from the manual of the item's
                       product; items with no product must cite nothing
    no_invented_order  every order ID in the reply was in the question

The evaluator only sees the final reply, not the passages the desk retrieved in that
turn, so "cites the right manual" stands in for "cites only retrieved chunks"
(m04/desk_graph.py checks that inside the graph, in critique).
"""
import re

CITATION = re.compile(r"\b((?:H200|D300|M270|FAQ)-[a-z0-9-]+?-\d+)\b")   # the M4 chunk ID pattern
ORDER_ID = re.compile(r"\b[Aa]\d{4}\b")


def has_fact(inputs, outputs, reference_outputs, keywords):
    reply = str(outputs).lower()
    found = [k for k in keywords if k.lower() in reply]
    return {"key": "has_fact", "score": len(found) / max(len(keywords), 1),
            "comment": f"found {found} of {keywords}"}


def cites_right_manual(inputs, outputs, reference_outputs, product):
    cited = CITATION.findall(str(outputs))
    if not product:
        ok = not cited
    else:
        ok = bool(cited) and all(c.startswith(product + "-") for c in cited)
    return {"key": "cites_right_manual", "score": 1.0 if ok else 0.0,
            "comment": f"expected {product or 'no citation'}, cited {cited or 'nothing'}"}


def no_invented_order(inputs, outputs, reference_outputs):
    asked = {o.upper() for o in ORDER_ID.findall(str(inputs))}
    named = {o.upper() for o in ORDER_ID.findall(str(outputs))}
    return {"key": "no_invented_order", "score": 1.0 if named <= asked else 0.0,
            "comment": f"invented {sorted(named - asked)}" if named - asked else "none invented"}
