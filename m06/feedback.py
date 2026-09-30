"""Step 7: structured user feedback, and turning the bad answers into new test cases.

    python m06/feedback.py rate --question "How many devices can the H200 remember?"     # asks you
    python m06/feedback.py rate --question "Where is order A1003?" --answer down \\
        --reason wrong_fact --comment "It said shipped" --correction "A1003 is still processing."
    python m06/feedback.py report
    python m06/feedback.py promote                    # -> m06/data/testset_feedback.jsonl
    nat eval --config_file m06/configs/eval_A.yml \\
        --override eval.general.dataset.file_path m06/data/testset_feedback.jsonl \\
        --override eval.general.output_dir m06/state/runs/feedback

rate     runs the desk on the question, shows the reply, and records one feedback record:
         a thumb (up/down), a reason from a fixed list, an optional comment and an optional
         correct answer. Without --answer it asks you; with --answer it is scripted.
report   counts the records by thumb and by reason (the seed records and yours).
promote  turns every thumbs-down record that has a correction into a test-set candidate
         (source "feedback", split "dev", needs_review true), skipping questions that
         are near-duplicates of test-set questions or of candidates already promoted.

The record follows the usual feedback shape: a key ("user_rating") with a numeric score
(1 up, 0 down) and a categorical value, plus comment and correction, attached to one run
by its trace ID. The schema is fixed up front and the client may only send allowed
values (REASONS below); anything else is refused, so the report can count reasons.
Each record also keeps what the desk did: the model, the route and the retrieved chunk IDs.

    m06/data/feedback_seed.jsonl   six records as if from users, so the loop runs without typing
    m06/state/feedback.jsonl       what `rate` appends

Near-duplicate: the two questions share at least 60% of their words (Jaccard similarity of
the word sets, the same idea as M4's near-duplicate filter, on words instead of shingles).

Promoted candidates are not labelled yet: a person checks the reference answer, the
category and the keywords (taken from the correction's numbers and codes), adds
ref_sections, and only then moves them into data/testset.jsonl. This is how the test set
grows from real failures. It does not retrain anything. Module 10 wires feedback buttons
to the same file.
"""
import argparse
import collections
import json
import pathlib
import re
import sys
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402  (first)
import testset  # noqa: E402

SEED = HERE / "data" / "feedback_seed.jsonl"
LOG = testset.STATE / "feedback.jsonl"
KEY = "user_rating"
THUMBS = {"up": 1, "down": 0}
REASONS = ["wrong_fact", "wrong_product", "missing_citation", "should_refuse", "should_answer", "other"]
DUPLICATE = 0.6
WORD = re.compile(r"[a-z0-9]+")
CODE = re.compile(r"\b(?:[A-Z]\d{2,4}|\d+(?:[.,]\d+)?(?:\s?(?:W|Hz|mm|days?|hours?|seconds?|minutes?|business days|devices))?)\b")
ORDER = re.compile(r"\b[Aa]\d{4}\b")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def records(paths=(SEED, LOG)) -> list[dict]:
    return [r for p in paths for r in load(p)]


def make_record(question: str, run: dict, thumb: str, reason: str | None, comment: str = "",
                correction: str = "", source: str = "user") -> dict:
    """One feedback record; refuses values outside the schema."""
    if thumb not in THUMBS:
        raise ValueError(f"thumb must be one of {list(THUMBS)}, got {thumb!r}")
    if thumb == "down" and reason not in REASONS:
        raise ValueError(f"a thumbs-down needs a reason from {REASONS}, got {reason!r}")
    if thumb == "up" and reason not in (None, "", *REASONS):
        raise ValueError(f"reason must be one of {REASONS}")
    return {"id": f"fb-{uuid.uuid4().hex[:8]}", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trace_id": run.get("trace_id", uuid.uuid4().hex), "key": KEY, "score": THUMBS[thumb], "value": thumb,
            "reason": reason or "", "comment": comment, "correction": correction, "question": question,
            "reply": run.get("reply", ""), "config": run.get("config", ""), "model": run.get("model", ""),
            "route": run.get("route", []), "retrieved_ids": [p["id"] for p in run.get("passages", [])],
            "source": source}


def rate(question: str, answer: str | None, reason: str | None, comment: str, correction: str,
         say=print, ask=input) -> dict:
    import desk_app
    testset.ensure_index(say=say)
    run = desk_app.ask(question)
    desk_app.close()
    run["trace_id"] = uuid.uuid4().hex
    run["config"] = f"{llm_calls.model_name()} T={llm_calls.temperature():g}"
    say(desk_app.desk_graph.wrap(f"Reply: {run['reply']}"))
    say(f"[rag] {', '.join(p['id'] for p in run['passages']) or 'no passages'}")
    if answer is None:                                   # interactive: a thumb, then a reason from the list
        while answer not in THUMBS:
            answer = ask("Was this reply helpful? [up/down]: ").strip().lower()
        if answer == "down":
            while reason not in REASONS:
                reason = ask(f"Why? {', '.join(REASONS)}: ").strip()
            comment = ask("Comment (optional): ").strip()
            correction = ask("The correct answer (optional): ").strip()
    rec = make_record(question, run, answer, reason, comment, correction)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    say(f"[feedback] {rec['value']} ({rec['reason'] or 'no reason'}) saved to {testset.rel(LOG)} "
        f"as {rec['id']}, trace {rec['trace_id'][:8]}")
    return rec


def report(say=print) -> dict:
    recs = records()
    thumbs = collections.Counter(r["value"] for r in recs)
    reasons = collections.Counter(r["reason"] for r in recs if r["value"] == "down")
    say(f"{len(recs)} feedback records ({len(load(SEED))} seeded, {len(load(LOG))} from rate): "
        f"{thumbs['up']} up, {thumbs['down']} down")
    for reason in REASONS:
        if reasons[reason]:
            say(f"  {reason:<18}{reasons[reason]:>3}")
    with_corr = sum(1 for r in recs if r["value"] == "down" and r.get("correction"))
    say(f"[INFO] {with_corr} thumbs-down records carry a correction (promote turns those into test cases)")
    return {"n": len(recs), "thumbs": dict(thumbs), "reasons": dict(reasons), "with_correction": with_corr}


def category_for(question: str, reason: str) -> str:
    import router   # m04/router.py (testset.py put m04 on the path)
    if reason == "should_refuse":
        return "off_topic"
    has_order, manual = bool(ORDER.search(question)), bool(router.MANUAL.search(question))
    if has_order and manual:
        return "mixed"
    if has_order:
        return "order"
    return "manual_paraphrase"


def promote(out: pathlib.Path = testset.FEEDBACK_SET, say=print) -> dict:
    import router
    existing = testset.load()
    kept, skipped = [], []
    for r in records():
        if r["value"] != "down":
            continue
        if not r.get("correction"):
            skipped.append((r["id"], "no correction"))
            continue
        near = max(((jaccard(r["question"], x["question"]), x["id"]) for x in existing + kept), default=(0, ""))
        if near[0] >= DUPLICATE:
            skipped.append((r["id"], f"near-duplicate of {near[1]} (Jaccard {near[0]:.2f})"))
            continue
        category = category_for(r["question"], r["reason"])
        refuse = category in testset.REFUSE
        kept.append({"id": f"f{len(kept) + 1:02d}", "split": "dev", "category": category, "question": r["question"],
                     "answer": r["correction"],
                     "keywords": [] if refuse else list(dict.fromkeys(CODE.findall(r["correction"])))[:3],
                     "product": "" if refuse else router.named_product(r["question"]),
                     "ref_sections": [], "source": "feedback", "needs_review": True,
                     "feedback_id": r["id"], "feedback_reason": r["reason"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    for item in kept:
        say(f"[promote] {item['id']} <- {item['feedback_id']} ({item['feedback_reason']}): {item['question']}")
    for fid, why in skipped:
        say(f"[skip]    {fid}: {why}")
    say(f"[INFO] {len(kept)} candidates in {testset.rel(out)} (needs_review: a person checks them "
        "before they join data/testset.jsonl)")
    return {"kept": kept, "skipped": skipped, "out": str(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate", help="run the desk and rate its reply")
    r.add_argument("--question", required=True)
    r.add_argument("--answer", choices=list(THUMBS), help="up or down (scripted; otherwise you are asked)")
    r.add_argument("--reason", choices=REASONS)
    r.add_argument("--comment", default="")
    r.add_argument("--correction", default="", help="the correct answer, if you know it")
    sub.add_parser("report", help="count the feedback by thumb and reason")
    p = sub.add_parser("promote", help="thumbs-down records with a correction -> test-set candidates")
    p.add_argument("--out", type=pathlib.Path, default=testset.FEEDBACK_SET)
    a = ap.parse_args()
    if a.cmd == "rate":
        if a.answer == "down" and not a.reason:
            ap.error("--answer down needs --reason")
        rate(a.question, a.answer, a.reason, a.comment, a.correction)
    elif a.cmd == "report":
        report()
    else:
        promote(a.out)


if __name__ == "__main__":
    main()
