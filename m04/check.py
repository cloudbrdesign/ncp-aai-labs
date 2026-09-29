"""Check that the M04 lab works end to end.

    python m04/check.py          # Ollama running with llama3.2:3b and embeddinggemma pulled

0. The chat model and the embedding model answer.
1. clean.py drops the exact copy, the older revision, the garbled scan and the near-empty
   page, and keeps the four manuals (4.4).
2. ingest.py: the collection has one row per chunk, the embedding dimension, both indexes
   and the metadata; neighbouring chunks overlap (4.3, 4.2).
3. The product filter returns only that product; keyword search ranks the E42 chunk
   first; hybrid hit@3 is at least the lower of dense and keyword (4.1, 4.2).
4. The SQL tool answers A1003 and refuses a write (4.5).
5. The router sends order, manual and mixed requests to the right sources; an invalid
   plan falls back to the keyword router (4.5).
6. The desk's reply cites only retrieved chunks; a planted citation is caught (4.5).

It rebuilds m04/state/clean/, m04/state/manuals.db and m04/data/orders.db.
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_graph  # noqa: E402  (first: it puts m03 on the path for the modules M4 reuses)
import clean  # noqa: E402
import compare_retrieval  # noqa: E402
import desk_steps  # noqa: E402
import ingest  # noqa: E402
import make_orders_db  # noqa: E402
import orders_sql  # noqa: E402
import retrieve  # noqa: E402
import router  # noqa: E402
import vector_store  # noqa: E402
from llm_calls import chat, describe, embed, fake_mode  # noqa: E402

results = []
quiet = lambda *a, **k: None  # noqa: E731
EXPECTED_KEPT = ["d300_dock.md", "h200_headset.md", "m270_monitor.md", "warranty_returns_faq.md"]
EXPECTED_DROPPED = {"d300_dock_copy.md": "exact duplicate", "h200_headset_rev1.md": "near-duplicate",
                    "m270_monitor_scan.md": "garbled", "d300_quickstart.md": "too short"}


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       -> {detail}" if detail and not ok else ""), flush=True)


def main():
    # 0. both models answer
    try:
        check(f"Chat model answers ({describe().split(',')[0]})",
              bool(chat([("user", "Reply with the single word OK.")]).strip()))
        dim = len(embed(["test"])[0])
        check(f"Embedding model answers (dimension {dim})", dim > 0)
    except Exception as e:
        check("Models answer", False, f"{type(e).__name__}: {e}. Start Ollama (ollama serve), then "
              "ollama pull llama3.2:3b and ollama pull embeddinggemma.")
        return

    # 1. clean
    out = clean.clean(say=quiet)
    check(f"clean.py keeps the four manuals ({', '.join(out['kept'])})", out["kept"] == EXPECTED_KEPT, out)
    wrong = {n: r for n, r in EXPECTED_DROPPED.items() if r not in out["dropped"].get(n, "")}
    check("clean.py drops the exact copy, the older revision, the garbled scan and the near-empty page",
          not wrong and len(out["dropped"]) == 4, out["dropped"])

    # 2. ingest
    res = ingest.ingest(clean.CLEAN_DIR, vector_store.COLLECTION, say=quiet)
    client = vector_store.connect()
    vector_store.load(client)
    check(f"Collection has one row per chunk ({res['count']} rows, {len(res['chunks'])} chunks)",
          res["count"] == len(res["chunks"]) > 0)
    field = next(f for f in client.describe_collection(vector_store.COLLECTION)["fields"] if f["name"] == "dense")
    check(f"Collection dimension matches the embedding model ({field['params']['dim']} = {dim})",
          int(field["params"]["dim"]) == dim)
    idx = {f: client.describe_index(vector_store.COLLECTION, f)["index_type"] for f in ("dense", "sparse")}
    check(f"Both indexes exist (dense {idx['dense']}, sparse {idx['sparse']})",
          idx == {"dense": "HNSW", "sparse": "SPARSE_INVERTED_INDEX"}, idx)
    rows = client.query(vector_store.COLLECTION, filter='id != ""', limit=1000,
                        output_fields=["product", "section", "source"])
    products = sorted({r["product"] for r in rows})
    check(f"Every chunk has product, section and source metadata ({', '.join(products)})",
          all(r["product"] and r["section"] and r["source"] for r in rows) and products == ["D300", "FAQ", "H200", "M270"])
    pairs = ingest.overlaps(res["chunks"])
    check(f"Neighbouring chunks overlap ({len(pairs)} pairs share text)", pairs and all(s.strip() for _, _, s in pairs))

    # 3. retrieval
    hits = retrieve.search(client, "the light blinks orange", "dense", 10, product="D300")
    check(f"Product filter returns only D300 chunks ({len(hits)} hits)", hits and {h["product"] for h in hits} == {"D300"})
    hits = retrieve.search(client, "What does E42 mean?", "keyword", 3)
    check(f"Keyword search ranks the E42 chunk first ({hits[0]['id'] if hits else 'nothing'})",
          hits and hits[0]["id"].startswith("D300-troubleshooting") and "E42" in hits[0]["text"], hits[:1])
    client.close()
    cmp = compare_retrieval.run(say=quiet)
    d, k, h = (cmp[m]["hits"] for m in ("dense", "keyword", "hybrid"))
    check(f"Hybrid hit@3 is at least the lower of dense and keyword (dense {d}, keyword {k}, hybrid {h} of 20)",
          h >= min(d, k))

    # 4. SQL
    make_orders_db.build()
    row = orders_sql.order_status("A1003") or {}
    check(f"SQL tool answers A1003 ({row.get('status')}, {row.get('product')})",
          row.get("status") == "processing" and row.get("product") == "M270", row)
    w = orders_sql.planted_write(say=quiet)
    check("A planted DELETE is refused by the validator and by the read-only connection",
          w["validator"] and w["sqlite"] and "readonly" in w["sqlite"] and w["intact"], w)

    # 5. routing
    cases = [  # (request, what the plan must contain, what it must not contain)
        ("Where is order A1003?", lambda s: s.action == "order_status" and s.order_id == "A1003",
         lambda s: s.action == "manual_search"),
        ("My D300 dock shows E42. What does it mean?", lambda s: s.action == "manual_search" and s.product == "D300",
         lambda s: s.action in ("order_status", "return_check")),
        ("The dock from order A1002 won't drive my second screen.",   # the order decides the product (SQL)
         lambda s: s.action == "manual_search" and s.order_id == "A1002", lambda s: False)]
    for request, must, must_not in cases:
        lines = []
        p, source = router.make_plan(request, "", [], log=lines.append)
        got = [(s.action, s.order_id, s.product) for s in p.steps]
        ok = any(must(s) for s in p.steps) and not any(must_not(s) for s in p.steps)
        routes = ", ".join(router.describe_step(s.model_dump()) for s in p.steps)
        short = request if len(request) <= 36 else request[:33] + "..."
        check(f"Router: \"{short}\" -> {routes} ({source})", ok, lines or got)
    saved = router.chat
    if fake_mode():
        os.environ["M04_FAKE_BAD_PLAN"] = "1"
    else:
        router.chat = lambda *a, **k: router.Plan.model_validate({"steps": [{"action": "refund_now"}] * 5})
    try:
        lines = []
        p, source = router.make_plan("Where is order A1003?", "", [], log=lines.append)
    finally:
        router.chat = saved
        os.environ.pop("M04_FAKE_BAD_PLAN", None)
    check("An invalid plan is rejected and the keyword router is logged",
          source == "fallback" and any("keyword router" in l for l in lines), lines)

    # 6. the desk: SQL, then RAG filtered by the order's product; citations checked
    desk_graph.SHOW["log"] = False
    with desk_graph.open_desk("inmemory") as app:
        out = desk_graph.run_turn(app, "chk-mixed", "chk", "The dock from order A1002 won't drive my second screen.")
        manual = [e for e in out["evidence"] if e["action"] == "manual_search"]
        retrieved = [p["id"] for p in desk_steps.passages(out["evidence"])]
        check("Desk: order A1002 -> product D300 from SQL -> manual search filtered to D300",
              manual and manual[0]["product"] == "D300" and retrieved and all(r.startswith("D300-") for r in retrieved),
              out["evidence"])
        cited = desk_graph.CITATION.findall(out["reply"])
        check(f"Desk reply cites only retrieved chunks ({', '.join(cited) or 'none'})",
              cited and set(cited) <= set(retrieved) and not desk_graph.rag_checks(out["reply"], retrieved), out["reply"])
        lines = []
        desk_graph.log = lines.append
        desk_graph.SHOW["log"] = True
        out = desk_graph.run_turn(app, "chk-bad", "chk", "My D300 dock shows E42. What does it mean?", bad_draft=True)
        check("--bad-draft: the planted citation is caught and removed",
              any("FAIL unknown_chunk" in l for l in lines) and desk_graph.PLANTED_CHUNK not in out["reply"],
              "\n".join(lines))


if __name__ == "__main__":
    print(f"NCP-AAI M04 lab check  [model: {describe()}]\n")
    main()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
