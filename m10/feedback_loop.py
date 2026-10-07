"""Structured feedback that changes the desk, and a measurement before the change is trusted (lesson 10.2).

    python m10/feedback_loop.py ingest                 # the 12 scripted events (data/feedback_script.jsonl)
    python m10/feedback_loop.py report                 # by channel, thumb and reason, and where each goes
    python m10/feedback_loop.py promote                # M6's promote -> state/feedback/testset_feedback.jsonl
    python m10/feedback_loop.py review --accept-all --reviewer "Ana"     # or --accept f01 f03 ...
    python m10/feedback_loop.py eval --label before --reps 3
    python m10/feedback_loop.py fix --reviewer "Ana"   # reviewed corrections -> FAQ passages, re-index, v10.1
    python m10/feedback_loop.py eval --label after --reps 3
    python m10/feedback_loop.py compare                # paired flips + exact McNemar (m06/compare_configs.py)
    python m10/feedback_loop.py fix --revert           # back to the v10 index (to run the loop again)

Two channels, one store (state/feedback/feedback.jsonl), never mixed in one number:
  user      thumbs from the page (POST /v1/feedback) and the scripted events: M6's record shape
            (m06/feedback.make_record: key user_rating, score 1/0, value up/down, a reason from M6's fixed
            list, comment, correction, the reply, the route, the retrieved chunk IDs) + request_id, customer
  reviewer  one reviewer_decision record per approval decision (approvals.py): approve / edit / reject,
            proposed vs final arguments, wait seconds. Edits and rejections are overrides of the model.

Where each signal goes (report prints this): a wrong fact with a correction -> knowledge (a reviewed FAQ
passage) and the test set; a missing citation -> the prompt's citation rule; should_refuse / should_answer ->
the rails and the refusal tests; reviewer edits -> the proposal prompt; reviewer rejections -> the
eligibility rules. Nothing here changes model weights.

promote  M6's own promote() on this store's user records: thumbs-down with a correction become test-set
         candidates (needs_review true), near-duplicates are skipped (Jaccard >= 0.6 of the words).
review   a person accepts candidates; each accepted item gets ref_sections (the index's best section for
         the correction, shown for the person to confirm) and needs_review false.
fix      accepted corrections (not the refusal items: a passage can't teach a refusal) become passages in
         state/feedback/feedback_faq.md, one section per item, written answer-first. They are added to the
         cleaned manuals (one file per product, so M4's product filter finds them), the index is rebuilt
         (M4's clean + ingest) and the desk version becomes v10.1. --revert rebuilds without them.
eval     the desk turn (Module 9's Desk at --layer, default L1; no refund cards) on the reviewed items and on
         the 42-item M6 test set, --reps times, scored with M6's deterministic checks (keywords_score,
         refusal_score, item_passes). Each customer is the order's owner, as in Module 9.
compare  per set: pass rate before and after (an item passes in more than half of its reps), paired flips,
         exact McNemar (m06/compare_configs.mcnemar). The items the fix was written for are expected to
         improve; on the full set a handful of items rarely gives p < 0.05, and the verdict says so.
"""
import argparse
import collections
import importlib.util
import json
import sys
import time
import uuid

import bootstrap
from bootstrap import LABS, STATE, audit_log, gd

FB = STATE / "feedback"
STORE = FB / "feedback.jsonl"
CANDIDATES = FB / "testset_feedback.jsonl"
REVIEWED = FB / "testset_reviewed.jsonl"
FAQ = FB / "feedback_faq.md"
EVAL = FB / "eval"
SCRIPT = bootstrap.HERE / "data" / "feedback_script.jsonl"
_spec = importlib.util.spec_from_file_location("m06_feedback", LABS / "m06" / "feedback.py")
m06_feedback = importlib.util.module_from_spec(_spec)
sys.modules["m06_feedback"] = m06_feedback
_spec.loader.exec_module(m06_feedback)
testset = gd.testset
ROUTES = {"wrong_fact": "knowledge (reviewed FAQ passage) + test set", "wrong_product": "retrieval filter + test set",
          "missing_citation": "prompt (citation rule) + test set", "should_refuse": "rails + refusal tests",
          "should_answer": "rails thresholds + test set", "other": "read by a person",
          "edit": "proposal prompt (the model's amount was wrong)", "reject": "eligibility rules (should a rule have refused it?)"}


def load(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(rec: dict) -> dict:
    FB.mkdir(parents=True, exist_ok=True)
    with STORE.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def records(channel: str | None = None) -> list[dict]:
    return [r for r in load(STORE) if channel is None or r.get("channel") == channel]


# ---- the two channels ---------------------------------------------------------------------------

def _run_for(request_id: str) -> dict:
    """What M6's make_record needs about the desk's reply, from the audit record and the retrieved IDs."""
    import oversight_desk as ov
    import decision_record
    turn = next((r for r in audit_log.query(request_id=request_id) if r.get("action") != "approval"), None)
    if turn is None:
        raise ValueError(f"no desk turn with request_id {request_id!r} in the audit log")
    ids = ov.PASSAGES.get(request_id) or decision_record.retrieved_ids(turn)
    return {"trace_id": turn["trace_id"], "reply": turn.get("reply") or "", "model": turn.get("model") or "",
            "config": f"{turn.get('model')} {turn.get('desk_version')} layer {turn.get('layer')}",
            "route": turn.get("route") or [], "passages": [{"id": i} for i in ids], "question": turn.get("input")}


def record_click(body: dict) -> dict:
    """A click on the page (POST /v1/feedback): M6's schema check, linked to the turn by request_id."""
    rid = str(body.get("request_id") or "")
    run = _run_for(rid)
    rec = m06_feedback.make_record(run["question"], run, str(body.get("value", "")), body.get("reason") or None,
                                   str(body.get("comment") or "")[:500], str(body.get("correction") or "")[:500],
                                   source="page")
    return append({**rec, "channel": "user", "request_id": rid, "customer": body.get("customer")})


def record_reviewer(card: dict, decision: str, reviewers: list[str], proposed: dict, final: dict | None,
                    message: str | None, out: dict) -> dict:
    """One reviewer decision as feedback: the second channel (approvals.py calls this)."""
    return append({"id": f"rd-{uuid.uuid4().hex[:8]}", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "channel": "reviewer", "key": "reviewer_decision", "value": decision,
                   "score": {"approve": 1.0, "edit": 0.5, "reject": 0.0}[decision], "reviewers": reviewers,
                   "request_id": card["request_id"], "trace_id": card["trace_id"], "thread_id": card["thread_id"],
                   "card_id": card["card_id"], "order_id": card["order"]["order_id"], "proposed_args": proposed,
                   "final_args": final, "message": audit_log.mask(message or ""), "wait_s": out["wait_s"],
                   "outcome": out["status"], "proposal_source": card["proposal"]["source"], "source": "approvals"})


def ingest(say=print) -> list[dict]:
    """Run each scripted question through the v10 desk, then record the scripted thumb on that reply."""
    import oversight_desk as ov
    od = ov.OversightDesk(layer="L1")
    out = []
    try:
        for ev in load(SCRIPT):
            res = od.chat(ev["question"], customer=ev["customer"], thread_id=f"fb-{ev['id']}-{uuid.uuid4().hex[:6]}")
            run = _run_for(res["request_id"])
            rec = m06_feedback.make_record(ev["question"], run, ev["value"], ev["reason"] or None, ev["comment"],
                                           ev["correction"], source="script")
            out.append(append({**rec, "channel": "user", "request_id": res["request_id"], "customer": ev["customer"],
                               "script_id": ev["id"]}))
            say(f"[ingest] {ev['id']} {ev['value']:<4} {ev['reason'] or '-':<17} request {res['request_id']}: {ev['question']}")
    finally:
        od.close()
    say(f"[ingest] {len(out)} records appended to {testset.rel(STORE)}")
    return out


def report(say=print) -> dict:
    user, rev = records("user"), records("reviewer")
    thumbs = collections.Counter(r["value"] for r in user)
    reasons = collections.Counter(r["reason"] for r in user if r["value"] == "down")
    srcs = collections.Counter(r.get("source") for r in user)
    say(f"user channel: {len(user)} records ({', '.join(f'{k} {v}' for k, v in sorted(srcs.items()))}): "
        f"{thumbs['up']} up, {thumbs['down']} down")
    say(f"  {'reason':<18}{'n':>3}  {'with correction':>15}  goes to")
    for reason in m06_feedback.REASONS:
        if reasons[reason]:
            corr = sum(1 for r in user if r["value"] == "down" and r["reason"] == reason and r.get("correction"))
            say(f"  {reason:<18}{reasons[reason]:>3}  {corr:>15}  {ROUTES[reason]}")
    decided = [r for r in rev]
    dec = collections.Counter(r["value"] for r in decided)
    over = (dec["edit"] + dec["reject"]) / len(decided) if decided else None
    say(f"reviewer channel: {len(decided)} decisions: approve {dec['approve']}, edit {dec['edit']}, reject {dec['reject']}"
        + (f"; override rate {over:.0%} (edited + rejected / reviewed)" if over is not None else ""))
    for k in ("edit", "reject"):
        if dec[k]:
            say(f"  {k:<18}{dec[k]:>3}  {'':>15}  {ROUTES[k]}")
    return {"user": len(user), "reviewer": len(decided), "thumbs": dict(thumbs), "reasons": dict(reasons),
            "decisions": dict(dec), "override_rate": over}


# ---- promote, review, fix ---------------------------------------------------------------------

def promote(say=print) -> dict:
    """M6's promote() on this store's user channel (reviewer decisions are not test cases)."""
    m06_feedback.records = lambda *a, **k: records("user")
    return m06_feedback.promote(out=CANDIDATES, say=say)


def suggest_sections(items: list[dict]) -> dict[str, list[dict]]:
    """The index's best section for each correction (hybrid search, filtered to the item's product)."""
    import retrieve
    out = {}
    with testset.open_index() as client:
        for it in items:
            if it["category"] in testset.REFUSE:
                out[it["id"]] = []
                continue
            hits = retrieve.search(client, it["answer"], "hybrid", 1, product=it.get("product") or None)
            out[it["id"]] = [{"product": h["product"], "section": h["section"]} for h in hits[:1]]
    return out


def review(accept: list[str] | None, accept_all: bool, reviewer: str, say=print, ask=input) -> list[dict]:
    cands = load(CANDIDATES)
    if not cands:
        raise SystemExit("[review] no candidates: run promote first")
    secs = suggest_sections(cands)
    kept = []
    for it in cands:
        say(f"{it['id']}  {it['category']:<18} {it['question']}\n      answer:   {it['answer']}\n"
            f"      keywords: {it['keywords']}  section: {secs[it['id']] or '-'}")
        if accept_all or (accept and it["id"] in accept):
            ok = True
        elif accept is None:
            ok = ask("      accept? [y/n]: ").strip().lower().startswith("y")
        else:
            ok = False
        if ok:
            kept.append({**it, "ref_sections": secs[it["id"]], "needs_review": False, "reviewed_by": reviewer,
                         "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        say(f"      -> {'accepted' if ok else 'dropped'}")
    REVIEWED.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    say(f"[review] {len(kept)} of {len(cands)} accepted by {reviewer} -> {testset.rel(REVIEWED)}")
    return kept


def faq_files(items: list[dict], reviewer: str) -> tuple[str, dict[str, str]]:
    """The reviewed FAQ: one Markdown for people, and one file per product for the index."""
    stamp = time.strftime("%Y-%m-%d")
    by_product = collections.defaultdict(list)
    for it in items:
        by_product[it.get("product") or "FAQ"].append(it)
    files, doc = {}, [f"# Reviewed answers from customer feedback (v10.1)\n\nReviewed by {reviewer} on {stamp}. "
                      "Each answer is a customer's correction, checked by a person against the manuals.\n"]
    for product, its in sorted(by_product.items()):
        body = [f"# {product} reviewed answers from customer feedback\n\nProduct code: {product}\n"
                f"Revision: v10.1 (reviewed by {reviewer}, {stamp})\n"]
        for it in its:
            body.append(f"## Reviewed answer {it['id']}\n\n{it['answer']} (Reviewed answer to the question: "
                        f"\"{it['question']}\")\n")
        files[f"feedback_faq_{product.lower()}.md"] = "\n".join(body)
        doc.append("\n".join(body).replace("# ", "## ", 1))
    return "\n".join(doc), files


def fix(reviewer: str, revert: bool = False, say=print) -> dict:
    """Rebuild the desk's index with (or, with revert, without) the reviewed FAQ passages."""
    import clean
    import ingest
    import vector_store
    items = [i for i in load(REVIEWED) if i["category"] not in testset.REFUSE] if not revert else []
    if not revert and not items:
        raise SystemExit("[fix] no accepted corrections: run review first")
    quiet = lambda *a, **k: None  # noqa: E731
    clean.clean(say=quiet)
    files = {}
    if items:
        doc, files = faq_files(items, reviewer)
        FAQ.write_text(doc)
        for name, text in files.items():
            (clean.CLEAN_DIR / name).write_text(text, encoding="utf-8")
    res = ingest.ingest(clean.CLEAN_DIR, vector_store.COLLECTION, say=quiet)
    version = "v10" if revert else "v10.1"
    bootstrap.VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.VERSION_FILE.write_text(json.dumps({"version": version, "faq_items": [i["id"] for i in items],
                                                  "reviewed_by": reviewer, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    new = [c["id"] for c in res["chunks"] if c["source"].startswith("feedback_faq_")]
    say(f"[fix] {'reverted: ' if revert else ''}{len(items)} reviewed passages in {len(files)} file(s) "
        f"({', '.join(files) or '-'}); index rebuilt: {res['count']} chunks ({len(new)} new: "
        f"{', '.join(new[:6])}{' ...' if len(new) > 6 else ''}); desk version {version}")
    if items:
        say(f"[fix] the reviewed FAQ for people: {testset.rel(FAQ)}")
    return {"items": len(items), "chunks": res["count"], "new_chunks": new, "version": version}


# ---- eval and compare ---------------------------------------------------------------------------

def owners() -> dict[str, str]:
    """M6 item ID -> the customer who owns its order (Module 9's benign set), so the scope lets it through."""
    return {r["id"]: r["customer"] for r in load(LABS / "m09" / "data" / "benign.jsonl")}


def eval_items(limit: int | None = None) -> list[dict]:
    reviewed = [{**i, "set": "reviewed"} for i in load(REVIEWED)]
    m6 = [{**i, "set": "m6"} for i in testset.load()]
    return reviewed[:limit] + m6[:limit] if limit else reviewed + m6


def run_eval(label: str, reps: int, layer: str, limit: int | None, say=print) -> dict:
    import evaluators   # m06: the deterministic checks (importing it loads NAT, which the venv has)
    import oversight_desk as ov
    items = eval_items(limit)
    if not any(i["set"] == "reviewed" for i in items):
        raise SystemExit("[eval] no reviewed items: run promote and review first (the same items before and after)")
    who = owners()
    desk = gd.Desk(layer=layer)
    EVAL.mkdir(parents=True, exist_ok=True)
    path = EVAL / f"{label}.jsonl"
    rows, t0 = [], time.time()
    with path.open("w") as f:
        for rep in range(reps):
            for it in items:
                out = ov.run_async(desk.respond(it["question"], customer=who.get(it["id"], "Tom B."),
                                                session=f"ev-{label}-{it['id']}-{rep}-{uuid.uuid4().hex[:4]}"))
                k, _ = evaluators.keywords_score(it.get("keywords"), out["reply"])
                r, _ = evaluators.refusal_score(it["category"], out["reply"])
                row = {"label": label, "version": bootstrap.desk_version(), "layer": layer, "set": it["set"],
                       "item": it["id"], "rep": rep, "category": it["category"], "keywords": k, "refusal": r,
                       "pass": evaluators.item_passes(k, r), "request_id": out["request_id"], "reply": out["reply"]}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gd.close()                                   # free the Milvus Lite file for the next step
    for s in ("reviewed", "m6"):
        maj = majority([r for r in rows if r["set"] == s])
        say(f"[eval] {label} ({bootstrap.desk_version()}, layer {layer}, {reps} rep{'s' if reps > 1 else ''}): "
            f"{s:<8} {sum(maj.values())}/{len(maj)} items pass")
    say(f"[eval] {len(rows)} turns in {time.time() - t0:.0f} s -> {testset.rel(path)}")
    return {"rows": len(rows), "path": str(path)}


def majority(rows: list[dict]) -> dict[str, bool]:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["item"]].append(r["pass"])
    return {i: sum(v) * 2 > len(v) for i, v in by.items()}


def verdict(regressions: int, improvements: int, p: float | None, n: int) -> str:
    if p is None:
        return f"INCONCLUSIVE: no item changed its verdict on these {n} items."
    flips = f"{improvements} improved, {regressions} regressed"
    if p >= 0.05:
        return (f"INCONCLUSIVE: {flips}, exact McNemar p = {p:.3f}. Only the flips count: it takes at least 6 in one "
                f"direction and none back for p < 0.05, so {n} items can show only a large change.")
    return (f"{'BETTER' if improvements > regressions else 'WORSE'} after the fix: {flips}, exact McNemar p = {p:.3g} "
            f"(one {n}-item set; confirm on new items before trusting it).")


def compare(say=print) -> dict:
    import compare_configs   # m06: mcnemar()
    before, after = load(EVAL / "before.jsonl"), load(EVAL / "after.jsonl")
    if not before or not after:
        raise SystemExit("[compare] run eval --label before and eval --label after first")
    lines = ["| set | items | before | after | improved | regressed | exact McNemar p | verdict |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for s in ("reviewed", "m6", "all"):
        b = majority([r for r in before if s == "all" or r["set"] == s])
        a = majority([r for r in after if s == "all" or r["set"] == s])
        common = sorted(set(b) & set(a))
        imp = [i for i in common if a[i] and not b[i]]
        reg = [i for i in common if b[i] and not a[i]]
        p = compare_configs.mcnemar(len(reg), len(imp))
        v = verdict(len(reg), len(imp), p, len(common))
        out[s] = {"items": len(common), "before": sum(b[i] for i in common), "after": sum(a[i] for i in common),
                  "improved": imp, "regressed": reg, "p": p, "verdict": v}
        lines.append(f"| {s} | {len(common)} | {out[s]['before']}/{len(common)} | {out[s]['after']}/{len(common)} | "
                     f"{len(imp)} | {len(reg)} | {'-' if p is None else f'{p:.3f}'} | {v.split(':')[0]} |")
    reps = max(r["rep"] for r in before) + 1
    head = (f"Before: {before[0]['version']}, after: {after[0]['version']}; layer {before[0]['layer']}; {reps} reps per item, "
            "an item passes in more than half of its reps (M6's keywords and refusal checks).")
    text = "\n".join([head, ""] + lines + ["", f"Reviewed items: {out['reviewed']['verdict']}",
                                           f"Full M6 test set: {out['m6']['verdict']}",
                                           f"Improved: {', '.join(out['all']['improved']) or '-'}; "
                                           f"regressed: {', '.join(out['all']['regressed']) or '-'}"])
    (EVAL / "compare.md").write_text(text + "\n")
    say(text)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="the 12 scripted feedback events")
    sub.add_parser("report", help="by channel and reason")
    sub.add_parser("promote", help="thumbs-down with a correction -> test-set candidates")
    r = sub.add_parser("review", help="a person accepts candidates")
    r.add_argument("--accept", nargs="*", help="candidate IDs to accept (others are dropped)")
    r.add_argument("--accept-all", action="store_true")
    r.add_argument("--reviewer", required=True)
    f = sub.add_parser("fix", help="reviewed corrections -> FAQ passages in the index (v10.1)")
    f.add_argument("--reviewer", default="")
    f.add_argument("--revert", action="store_true", help="rebuild the index without them (v10)")
    e = sub.add_parser("eval", help="the desk on the reviewed items and the M6 test set")
    e.add_argument("--label", required=True, choices=["before", "after"])
    e.add_argument("--reps", type=int, default=3)
    e.add_argument("--layer", default="L1", choices=gd.layers.LAYERS)
    e.add_argument("--limit", type=int, help="first N items of each set (a quick run)")
    sub.add_parser("compare", help="before vs after: flips and McNemar")
    a = ap.parse_args()
    print(f"[INFO] model: {gd.llm_calls.describe()}; desk {bootstrap.desk_version()}")
    if a.cmd == "ingest":
        ingest()
    elif a.cmd == "report":
        report()
    elif a.cmd == "promote":
        promote()
    elif a.cmd == "review":
        review(a.accept, a.accept_all, a.reviewer)
    elif a.cmd == "fix":
        fix(a.reviewer, a.revert)
    elif a.cmd == "eval":
        run_eval(a.label, a.reps, a.layer, a.limit)
    else:
        compare()


if __name__ == "__main__":
    main()
