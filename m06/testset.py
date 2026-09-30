"""Step 1: the labelled test set (data/testset.jsonl): what is in it, and do its labels still hold?

    python m06/testset.py --stats                         # counts per category and split; label check
    python m06/testset.py --show --category unanswerable  # print items
    python m06/testset.py --show --split dev

One JSON object per line:

    id            q01..q20 (M4's retrieval questions), e01..e13 (M5's eval items), n01.. (new)
    split         dev (tune and re-run on these) or test (held back for the final comparison)
    category      order, manual_exact, manual_paraphrase, mixed, unanswerable, off_topic, injection
    question      what the customer writes
    answer        the reference reply (what a good reply says; used by the judge)
    keywords      words a correct reply must contain; "a|b" accepts either (evaluators.py)
    product       the manual a correct reply cites (H200, D300, M270, FAQ), or "" for none
    ref_sections  where the answer is: [{"product", "section", "contains"?}, ...]
    source        m04, m05, new, or feedback (promoted by feedback.py)

The labels name a product and section (and, where a section has several chunks, a phrase
the right chunk contains), never chunk IDs. The chunk IDs are looked up in the index at run
time, so changing the chunk size in M4's ingest.py does not break the labels; --stats
checks that every label still finds at least one chunk.

A test set is only useful if it stays representative: every category needs a few items,
both splits need every kind, and new failure cases (feedback.py promote) join the dev
split after a human has reviewed them.
"""
import argparse
import collections
import contextlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402,F401  (first: the model switch and, in fake mode, the fake server)

sys.path.insert(1, str(HERE.parent / "m04"))
import vector_store  # noqa: E402   m04/vector_store.py

DATA = HERE / "data"
TESTSET = DATA / "testset.jsonl"
FEEDBACK_SET = DATA / "testset_feedback.jsonl"
STATE = HERE / "state"
CATEGORIES = ["order", "manual_exact", "manual_paraphrase", "mixed", "unanswerable", "off_topic", "injection"]
REFUSE = {"unanswerable", "off_topic", "injection"}   # categories where the right reply declines
SPLITS = ["dev", "test"]


def load(path: pathlib.Path = TESTSET) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_all() -> list[dict]:
    """The test set plus the promoted feedback items, if any (the desk looks questions up in both)."""
    return load(TESTSET) + load(FEEDBACK_SET) + load(STATE / "testset_feedback.jsonl")


def by_question(items: list[dict] | None = None) -> dict[str, dict]:
    return {norm(i["question"]): i for i in (items if items is not None else load_all())}


def rel(path) -> str:
    """A path as the README writes it: relative to the repo root when it is inside the repo."""
    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(HERE.parent))
    except ValueError:
        return str(path)


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---- the index ------------------------------------------------------------------------

def index_ready() -> bool:
    import make_orders_db
    if not (make_orders_db.DB.exists() and vector_store.DB_PATH.exists()):
        return False
    with open_index(load_collection=False) as client:
        return client.has_collection(vector_store.COLLECTION)


def ensure_index(say=print) -> None:
    """Build the M4 index and orders database if they are missing (as M5 does)."""
    if not index_ready():
        import desk_app
        desk_app.build_index(say=say)


@contextlib.contextmanager
def open_index(load_collection: bool = True):
    """A Milvus Lite client on m04/state/manuals.db that frees the file when done.

    Milvus Lite runs one local server per .db file and locks the file until that server
    stops, so a script must release it before `nat eval` (another process) can open it.
    """
    from milvus_lite.server_manager import server_manager_instance
    client = vector_store.connect()
    try:
        if load_collection:
            vector_store.load(client)
        yield client
    finally:
        client.close()
        server_manager_instance.release_server(str(vector_store.DB_PATH))


def chunk_rows(client) -> list[dict]:
    return client.query(vector_store.COLLECTION, filter='id != ""', limit=10000,
                        output_fields=["id", "product", "section", "text"])


def resolve(items: list[dict], rows: list[dict]) -> dict[str, list[str]]:
    """Each item's reference chunk IDs: chunks of the labelled product and section that contain
    the labelled phrase (if one is given). Items without ref_sections get []."""
    out = {}
    for item in items:
        ids = []
        for ref in item.get("ref_sections", []):
            for r in rows:
                if (r["product"] == ref["product"] and r["section"] == ref["section"]
                        and ref.get("contains", "").lower() in " ".join(r["text"].split()).lower()):
                    ids.append(r["id"])
        out[item["id"]] = sorted(set(ids))
    return out


def reference_ids(items: list[dict] | None = None) -> dict[str, list[str]]:
    items = items if items is not None else load_all()
    with open_index() as client:
        return resolve(items, chunk_rows(client))


# ---- statistics -----------------------------------------------------------------------

def stats(items: list[dict], refs: dict[str, list[str]]) -> dict:
    counts = collections.Counter((i["category"], i["split"]) for i in items)
    unresolved = [i["id"] for i in items if i.get("ref_sections") and not refs.get(i["id"])]
    per_cat = {c: sum(counts[(c, s)] for s in SPLITS) for c in CATEGORIES}
    return {"n": len(items), "counts": counts, "per_category": per_cat,
            "splits": collections.Counter(i["split"] for i in items),
            "sources": collections.Counter(i["source"] for i in items),
            "unresolved": unresolved,
            "chunks_per_label": collections.Counter(len(v) for k, v in refs.items() if v)}


def show_stats(items: list[dict], refs: dict[str, list[str]], say=print) -> dict:
    s = stats(items, refs)
    say(f"{'category':<20}{'dev':>5}{'test':>6}{'all':>6}")
    for c in CATEGORIES:
        say(f"{c:<20}{s['counts'][(c, 'dev')]:>5}{s['counts'][(c, 'test')]:>6}{s['per_category'][c]:>6}")
    say(f"{'all':<20}{s['splits']['dev']:>5}{s['splits']['test']:>6}{s['n']:>6}")
    say(f"[INFO] sources: " + ", ".join(f"{k} {v}" for k, v in sorted(s["sources"].items())))
    labelled = sum(1 for i in items if i.get("ref_sections"))
    say(f"[INFO] {labelled} items have reference sections; chunks per label: "
        + ", ".join(f"{n} chunk{'s' if n > 1 else ''}: {c}" for n, c in sorted(s["chunks_per_label"].items())))
    if s["unresolved"]:
        say(f"[FAIL] labels that find no chunk in the index: {', '.join(s['unresolved'])}")
    else:
        say("[INFO] every reference section resolves to at least one chunk in the index")
    small = [c for c, n in s["per_category"].items() if n < 3]
    if small:
        say(f"[WARN] categories with fewer than 3 items: {', '.join(small)}")
    return s


def show(items: list[dict], refs: dict[str, list[str]], say=print) -> None:
    for i in items:
        say(f"{i['id']:<5} {i['split']:<5} {i['category']:<18} {i['question']}")
        say(f"      answer:   {i['answer']}")
        say(f"      keywords: {i['keywords']}  product: {i['product'] or '-'}  "
            f"chunks: {', '.join(refs.get(i['id'], [])) or '-'}  source: {i['source']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="counts per category and split; check the labels")
    ap.add_argument("--show", action="store_true", help="print the items (with their resolved chunk IDs)")
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--file", type=pathlib.Path, default=TESTSET, help="another test set file (same fields)")
    a = ap.parse_args()
    items = load(a.file)
    if not items:
        raise SystemExit(f"[FAIL] no items in {a.file}")
    ensure_index()
    refs = reference_ids(items)
    if a.show:
        chosen = [i for i in items if (not a.category or i["category"] == a.category)
                  and (not a.split or i["split"] == a.split)]
        show(chosen, refs)
    if a.stats or not a.show:
        show_stats(items, refs)


if __name__ == "__main__":
    main()
