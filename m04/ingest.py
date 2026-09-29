"""Step 2: load the cleaned manuals, chunk them, embed the chunks, index them in Milvus Lite.

    python m04/ingest.py                              # chunk size 300, overlap 60, HNSW
    python m04/ingest.py --chunks-only                # chunk statistics only: no model, no database
    python m04/ingest.py --chunk-size 600 --chunks-only
    python m04/ingest.py --index FLAT                 # or IVF_FLAT
    python m04/ingest.py --raw                        # the UNcleaned manuals, into collection manuals_raw

An ingestion (ETL) pipeline in four stages:

  load    read each cleaned Markdown file, join hard-wrapped lines back into paragraphs,
          and pull metadata out of the text (the "Product code: D300" line)
  split   MarkdownHeaderTextSplitter cuts at the "## Section" headings, so every piece
          knows its section; RecursiveCharacterTextSplitter then cuts long sections into
          chunks of at most --chunk-size characters, each sharing --overlap characters
          with the one before (add_start_index records where each chunk starts). Each
          chunk's text starts with the manual title and section, so a chunk from the
          middle of a section still says what it is about.
  embed   one vector per chunk, in batches (llm_calls.embed: Ollama embeddinggemma)
  index   insert into the Milvus Lite collection (vector_store.py): Milvus fills the BM25
          sparse vector from the text itself and builds both indexes

Chunk size is a trade-off: bigger chunks keep more context together but cost more
embedding time, make the prompt longer and blur what a chunk is about; overlap keeps a
sentence that falls on a boundary findable from both sides, at the cost of some
repeated text. Sizes here are characters, not tokens.
"""
import argparse
import collections
import pathlib
import re
import time

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import vector_store
from llm_calls import EMBED_MODEL, embed, fake_mode

HERE = pathlib.Path(__file__).resolve().parent
CLEAN_DIR = vector_store.STATE_DIR / "clean"
RAW_DIR = HERE / "data" / "manuals"
PRODUCT = re.compile(r"^Product code: (\S+)", re.M)
TITLE = re.compile(r"^# (.+)$", re.M)
WRAPPED = re.compile(r"(?<=\S)\n(?=[^\n#-])")   # a line break inside a paragraph


def load(folder: pathlib.Path, say=print) -> list[dict]:
    """The loader: one dict per file with its text and the metadata found in it."""
    docs = []
    for f in sorted(folder.glob("*.md")):
        text = WRAPPED.sub(" ", f.read_text(encoding="utf-8"))   # one paragraph = one line
        product = PRODUCT.search(text)
        docs.append({"source": f.name, "text": text, "product": product.group(1) if product else "?",
                     "title": TITLE.search(text).group(1) if TITLE.search(text) else f.stem})
    return docs


def slug(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")


def split(docs: list[dict], chunk_size: int = 300, overlap: int = 60) -> list[dict]:
    """Section split, then size split. Returns one dict per chunk (text + metadata)."""
    by_heading = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True)
    chunks, seen = [], set()
    for d in docs:
        sections = [s for s in by_heading.split_text(d["text"]) if s.metadata.get("section")]  # skip the header block
        for s in sections:
            s.metadata.update(product=d["product"], source=d["source"])
        per_section = collections.Counter()
        for c in by_size.split_documents(sections):
            m = c.metadata
            per_section[m["section"]] += 1
            cid = f"{m['product']}-{slug(m['section'])}-{per_section[m['section']]}"
            if cid in seen:   # only in the uncleaned data (--raw): a duplicate file gives the same IDs
                cid += "@" + pathlib.Path(m["source"]).stem
            seen.add(cid)
            chunks.append({
                "id": cid,
                "text": f"{d['title']} / {m['section']}\n{c.page_content}",
                "body": c.page_content, "start": m["start_index"],
                "product": m["product"], "section": m["section"], "source": m["source"]})
    return chunks


def overlaps(chunks: list[dict]) -> list[tuple[dict, dict, str]]:
    """Pairs of neighbouring chunks from the same section and the text they share."""
    pairs = []
    for a, b in zip(chunks, chunks[1:]):
        if a["source"] == b["source"] and a["section"] == b["section"] and b["start"] < a["start"] + len(a["body"]):
            pairs.append((a, b, a["body"][b["start"] - a["start"]:]))
    return pairs


def chunk_stats(chunks: list[dict], chunk_size: int, overlap: int, say=print):
    per_product = collections.Counter(c["product"] for c in chunks)
    lengths = [len(c["body"]) for c in chunks]
    say(f"[chunk] {len(chunks)} chunks (size {chunk_size}, overlap {overlap}): "
        + ", ".join(f"{p} {n}" for p, n in sorted(per_product.items())))
    say(f"[chunk] length: average {sum(lengths) // max(len(lengths), 1)}, longest {max(lengths, default=0)} characters")
    pairs = overlaps(chunks)
    say(f"[chunk] {len(pairs)} neighbouring pairs overlap")
    if pairs:
        a, b, shared = pairs[0]
        say(f"[chunk] {a['id']} -> {b['id']} share: \"{' '.join(shared.split())[:44]}...\"")
    c = chunks[0] if chunks else None
    if c:
        say(f"[chunk] metadata of {c['id']}: product={c['product']} section={c['section']} "
            f"source={c['source']} start={c['start']}")


def ingest(folder: pathlib.Path, collection: str, chunk_size: int = 300, overlap: int = 60,
           index: str = "HNSW", say=print) -> dict:
    """Load, split, embed, index. Returns a summary (check.py reads it)."""
    docs = load(folder)
    say(f"[load] {len(docs)} files from {folder.relative_to(HERE.parent)}, products: "
        + ", ".join(d["product"] for d in docs))
    chunks = split(docs, chunk_size, overlap)
    chunk_stats(chunks, chunk_size, overlap, say)

    t = time.time()
    vectors = embed([c["text"] for c in chunks])
    dim = len(vectors[0])
    model = "fake hashed bag of words" if fake_mode() else f"ollama {EMBED_MODEL}"
    say(f"[embed] {len(vectors)} chunks with {model}: dimension {dim}, {time.time() - t:.1f} s")

    client = vector_store.connect()
    params = vector_store.create(client, collection, dim, index)
    shown = ", ".join(f"{k}={v}" for k, v in params.items()) or "no parameters"
    say(f"[milvus] collection {collection} in {vector_store.DB_PATH.relative_to(HERE.parent)}")
    say(f"[milvus] index on dense: {index}, metric COSINE ({shown})")
    say("[milvus] index on sparse: SPARSE_INVERTED_INDEX, metric BM25 (Milvus computes it from text)")
    rows = [{k: c[k] for k in ("id", "text", "product", "section", "source")} | {"dense": v}
            for c, v in zip(chunks, vectors)]
    client.insert(collection, rows)
    n = vector_store.count(client, collection)
    say(f"[milvus] inserted {len(rows)} rows; the collection has {n}; indexes: "
        + ", ".join(client.list_indexes(collection)))
    client.close()
    return {"chunks": chunks, "dim": dim, "count": n, "index": index}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=300, help="characters per chunk (default 300)")
    ap.add_argument("--overlap", type=int, default=60, help="characters shared by neighbouring chunks (default 60)")
    ap.add_argument("--index", choices=list(vector_store.INDEXES), default="HNSW", help="index on the dense field")
    ap.add_argument("--chunks-only", action="store_true", help="print chunk statistics and stop")
    ap.add_argument("--raw", action="store_true", help="ingest the uncleaned manuals into collection manuals_raw")
    a = ap.parse_args()
    folder = RAW_DIR if a.raw else CLEAN_DIR
    if not folder.exists() or not any(folder.glob("*.md")):
        raise SystemExit("[FAIL] no cleaned manuals yet. Run: python m04/clean.py")
    if a.chunks_only:
        chunk_stats(split(load(folder), a.chunk_size, a.overlap), a.chunk_size, a.overlap)
        return
    collection = "manuals_raw" if a.raw else vector_store.COLLECTION
    ingest(folder, collection, a.chunk_size, a.overlap, a.index)


if __name__ == "__main__":
    main()
