"""Module 11 lab check: the capstone run, end to end.

    python m11/check.py                       # Free mode: Ollama + ticket API + heuristics server, layer L4
    M11_FAKE_LLM=1 python m11/check.py        # offline self-test (CI): scripted models, layer L1

What must hold:
  1. map: ten exam domains, every file it names exists in the repo
  2. manual: answered from the manuals, the reply cites a chunk the desk retrieved, critiqued
  3. order: the order tool ran, inside the customer's own scope
  4. injection: blocked by a rail (at L2 and up the input rails stop it before the model; the offline
     self-test runs L1, where the scripted reply is stopped by the output rails)
  5. refund: one approval card, decided by Ana, exactly one ledger row
  6. every turn has a complete decision record (M10's coverage), and report.md is written
"""
import os
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
FAKE = os.environ.get("M11_FAKE_LLM") == "1"
os.environ.setdefault("M11_STATE_DIR", str(HERE / "state" / "check"))
shutil.rmtree(os.environ["M11_STATE_DIR"], ignore_errors=True)
sys.path.insert(0, str(HERE))
import capstone  # noqa: E402

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    missing = [f.strip() for _, _, files, _ in capstone.DOMAINS for f in files.split(",")
               if not (capstone.LABS / f.strip()).exists()]
    check("map", len(capstone.DOMAINS) == 10 and not missing, f"{len(capstone.DOMAINS)} domains; missing files {missing}")
    layer = "L1" if FAKE else "L4"
    out = capstone.run(layer, say=lambda *a: None)
    c = {x["name"]: x for x in out["conversations"]}
    rec = {k: capstone.decision_record.build(v["request_id"]) for k, v in c.items()}

    m = rec["manual"]
    check("manual", c["manual"]["action"] == "answered" and bool(m["cited"]) and set(m["cited"]) <= set(m["retrieved"])
          and bool(m["critique"]), f"retrieved {m['retrieved']}, cited {m['cited']}")
    o = rec["order"]["audit"]
    tools = [t["tool"] for t in o.get("tool_calls") or []]
    check("order", "order_status" in tools and o.get("scope", "").endswith(o.get("caller", "?")),
          f"tools {tools}, scope '{o.get('scope')}'")
    i = rec["injection"]["audit"]
    stopped_in = i.get("blocked_by") in [r["name"] for r in (i.get("input_rails") or {}).get("rails", [])]
    check("injection", c["injection"]["action"] == "blocked" and (FAKE or stopped_in),
          f"action {c['injection']['action']}, stopped by {i.get('blocked_by')} ({'input' if stopped_in else 'output'} rails, layer {layer})")
    r = rec["refund"]
    reviewers = [a["approval"].get("reviewers") for a in r["approvals"] if not a["approval"].get("partial")]
    check("refund", [x["status"] for x in r["cards"]] == ["decided"] and reviewers == [["Ana"]] and len(r["ledger"]) == 1,
          f"cards {[x['status'] for x in r['cards']]}, reviewers {reviewers}, ledger rows {len(r['ledger'])}")
    cov = capstone.decision_record.coverage(say=lambda *a: None)
    check("records", cov["complete"] == cov["turns"] and capstone.REPORT.exists(),
          f"{cov['complete']}/{cov['turns']} complete; {capstone.REPORT.relative_to(capstone.LABS)}")
    print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
