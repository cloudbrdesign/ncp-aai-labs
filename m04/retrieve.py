"""Step 3: search the manuals three ways: dense, keyword (BM25) and hybrid.

    python m04/retrieve.py "My second screen just copies the first one"            # hybrid, RRF
    python m04/retrieve.py --mode keyword "E42"
    python m04/retrieve.py --mode dense "How do I stop the headphones blocking noise?"
    python m04/retrieve.py --ranker weighted --weights 0.7 0.3 "E42 on my dock"
    python m04/retrieve.py --product D300 "the light blinks orange"                  # filtered search

  dense    embed the question with the same model as the chunks and find the nearest
           vectors (COSINE) through the HNSW index. Finds meaning: "headphones" matches a
           chunk that says "headset". Search parameter --ef: how many candidates HNSW keeps
           while it walks the graph (at least k; higher = better recall, slower).
  keyword  BM25 full-text search. Milvus turns the question into words with the same
           analyzer as the chunks and scores each chunk by term frequency, inverse
           document frequency (rare words such as "E42" count most) and chunk length.
           Finds exact codes and model numbers that embeddings blur.
  hybrid   both searches, then one fused list. RRF (reciprocal rank fusion) scores each
           chunk by 1 / (60 + rank) in each list and adds them up: only ranks matter, so
           the two very different score scales don't have to be compared. The weighted
           ranker normalises both scores and mixes them (--weights dense sparse).
  --product filters on the product field before the vector search (standard filtering):
           only that product's chunks are candidates.

For hybrid results the log shows where each chunk ranked in the dense and the keyword list.
"""
import argparse
import re
import time

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

import vector_store
from llm_calls import describe, embed

CANDIDATES = 20   # how many results each list in a hybrid search hands to the ranker
FIELDS = ["text", "product", "section", "source"]


def product_filter(product: str | None) -> str | None:
    return f'product == "{product}"' if product else None


def search(client, query: str, mode: str = "hybrid", k: int = 3, product: str | None = None,
           ranker: str = "rrf", weights: tuple[float, float] = (0.5, 0.5), ef: int = 64,
           collection: str = vector_store.COLLECTION, qvec: list[float] | None = None,
           index: str | None = None) -> list[dict]:
    """One search. Returns [{id, score, product, section, text}, ...], best first.
    Pass qvec to reuse a query embedding (compare_retrieval.py times the search alone)."""
    flt = product_filter(product)
    index = index or vector_store.dense_index(client, collection)
    dense = vector_store.dense_params(index, ef=max(ef, k))
    if mode != "keyword" and qvec is None:
        qvec = embed([query])[0]
    if mode == "dense":
        res = client.search(collection, data=[qvec], anns_field="dense", limit=k, filter=flt or "",
                            output_fields=FIELDS, search_params=dense)
    elif mode == "keyword":
        res = client.search(collection, data=[query], anns_field="sparse", limit=k, filter=flt or "",
                            output_fields=FIELDS, search_params={"metric_type": "BM25"})
    else:
        reqs = [AnnSearchRequest([qvec], "dense", dense, max(CANDIDATES, k), expr=flt),
                AnnSearchRequest([query], "sparse", {"metric_type": "BM25"}, max(CANDIDATES, k), expr=flt)]
        rk = RRFRanker(60) if ranker == "rrf" else WeightedRanker(*weights)
        res = client.hybrid_search(collection, reqs, ranker=rk, limit=k, output_fields=FIELDS)
    # Milvus Lite 3.2.1 reports BM25 scores as negative numbers (best = most negative): flip the sign
    return [{"id": h["id"], "score": abs(h["distance"]) if mode == "keyword" else h["distance"], **h["entity"]}
            for h in res[0]]


def preview(text: str, query: str = "", width: int = 60) -> str:
    """The chunk text without its "title / section" line, from the first query word it contains."""
    body = " ".join(text.split("\n", 1)[-1].split())
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", query) if w.lower() not in {"what", "does", "the", "how", "mean"}]
    found = {w: m.start() for w in words for m in [re.search(rf"\b{re.escape(w)}\b", body, re.I)] if m}
    codes = [s for w, s in found.items() if any(c.isdigit() for c in w)]   # codes like E42 first
    start = min(codes or found.values() or [0])
    if start > 20:
        body = "..." + body[start:]
    return body if len(body) <= width else body[:width - 3] + "..."


def show(hits: list[dict], ranks: dict | None = None, query: str = "", say=print):
    for i, h in enumerate(hits, 1):
        where = ""
        if ranks:
            d, kw = ranks["dense"].get(h["id"]), ranks["keyword"].get(h["id"])
            where = f" (dense {d or '-'}, kw {kw or '-'})"
        say(f"{i:>2} {h['score']:7.4f}  {h['id']:<27}{where}".rstrip())
        say(f"   {preview(h['text'], query)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["dense", "keyword", "hybrid"], default="hybrid")
    ap.add_argument("--ranker", choices=["rrf", "weighted"], default="rrf", help="how hybrid fuses the two lists")
    ap.add_argument("--weights", type=float, nargs=2, default=[0.5, 0.5], metavar=("DENSE", "SPARSE"))
    ap.add_argument("--product", help="only search this product's chunks (H200, D300, M270, FAQ)")
    ap.add_argument("--ef", type=int, default=64, help="HNSW search parameter ef (default 64)")
    ap.add_argument("-k", type=int, default=3, help="how many chunks to return (default 3)")
    ap.add_argument("--collection", default=vector_store.COLLECTION, help="manuals, or manuals_raw after --raw")
    a = ap.parse_args()
    client = vector_store.connect()
    vector_store.load(client, a.collection)
    index = vector_store.dense_index(client, a.collection)
    how = {"rrf": "RRF k=60", "weighted": f"weighted {a.weights[0]}/{a.weights[1]}"}[a.ranker]
    print(f"[INFO] {a.collection}: {vector_store.count(client, a.collection)} chunks, dense index {index}"
          f" | model: {describe()}")
    print(f"[INFO] mode {a.mode}{' (' + how + ')' if a.mode == 'hybrid' else ''} | k={a.k}"
          f" | filter: {product_filter(a.product) or 'none'}")
    t = time.time()
    hits = search(client, a.query, a.mode, a.k, a.product, a.ranker, tuple(a.weights), a.ef, a.collection,
                  index=index)
    ms = (time.time() - t) * 1000
    ranks = None
    if a.mode == "hybrid":
        qvec = embed([a.query])[0]
        ranks = {m: {h["id"]: i for i, h in enumerate(search(client, a.query, m, CANDIDATES, a.product,
                                                             collection=a.collection, qvec=qvec, index=index), 1)}
                 for m in ("dense", "keyword")}
    print(f"Question: {a.query}")
    show(hits, ranks, a.query)
    print(f"[INFO] {ms:.0f} ms" + (" (embedding the question included)" if a.mode != "keyword" else ""))
    client.close()


if __name__ == "__main__":
    main()
