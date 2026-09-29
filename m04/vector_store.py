"""The Milvus Lite collection that holds the manual chunks: schema, indexes, search settings.

Milvus Lite is Milvus as a Python library: the whole database is one local file
(m04/state/manuals.db), no server, no Docker. It is for prototyping; the same code runs
against Milvus Standalone or Distributed by changing the URI.

One collection, "manuals", one row per chunk:

    id        chunk ID, like D300-troubleshooting-1 (the desk cites these)
    text      the chunk text; the analyzer splits it into words for BM25
    product   H200, D300, M270 or FAQ (scalar field: filter on it)
    section   the manual section, like Troubleshooting
    source    the file the chunk came from
    dense     the embedding (float vector; its dimension is read from the model)
    sparse    BM25 term weights, filled by Milvus itself from `text` (a BM25 Function)

Two indexes, one per vector field:
    dense   HNSW with COSINE (or FLAT / IVF_FLAT with --index)
    sparse  SPARSE_INVERTED_INDEX with BM25

Milvus Lite 3.2.1 builds FLAT, HNSW, HNSW_SQ, IVF_FLAT and IVF_SQ8 indexes. It accepts
DISKANN, IVF_PQ or GPU_CAGRA without an error but builds no index for them, so this lab
only offers FLAT, HNSW and IVF_FLAT.

pymilvus 2.6.9 reads the first part of a local file path as a database name, so we pass
db_name="default". A collection is "released" after the file is reopened: load() it
before searching.
"""
import os
import pathlib

from pymilvus import DataType, Function, FunctionType, MilvusClient

HERE = pathlib.Path(__file__).resolve().parent
STATE_DIR = pathlib.Path(os.environ.get("M04_STATE_DIR", HERE / "state"))
DB_PATH = STATE_DIR / "manuals.db"
COLLECTION = "manuals"

# Build-time index parameters for the dense field.
INDEXES = {
    "FLAT": {},                                   # exact search: compares every vector (the baseline)
    "HNSW": {"M": 16, "efConstruction": 200},     # graph: M links per node, efConstruction = build effort
    "IVF_FLAT": {"nlist": 8},                     # nlist clusters (small, because we have few chunks)
}


def connect() -> MilvusClient:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return MilvusClient(str(DB_PATH), db_name="default")


def create(client: MilvusClient, name: str, dim: int, index: str = "HNSW") -> dict:
    """(Re)create the collection with both vector fields, the BM25 function and both indexes."""
    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=80)
    schema.add_field("text", DataType.VARCHAR, max_length=4000, enable_analyzer=True)
    schema.add_field("product", DataType.VARCHAR, max_length=16)
    schema.add_field("section", DataType.VARCHAR, max_length=80)
    schema.add_field("source", DataType.VARCHAR, max_length=120)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="text_bm25", function_type=FunctionType.BM25,
                                 input_field_names=["text"], output_field_names=["sparse"]))
    params = client.prepare_index_params()
    params.add_index(field_name="dense", index_type=index, metric_type="COSINE", params=INDEXES[index])
    params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(name, schema=schema, index_params=params)
    return INDEXES[index]


def load(client: MilvusClient, name: str = COLLECTION) -> None:
    if not client.has_collection(name):
        raise SystemExit(f"[FAIL] no collection '{name}' in {DB_PATH}. Run: python m04/ingest.py")
    client.load_collection(name)


def dense_index(client: MilvusClient, name: str = COLLECTION) -> str:
    return client.describe_index(name, "dense")["index_type"]


def dense_params(index: str, ef: int = 64, nprobe: int = 4) -> dict:
    """Search-time parameters: ef for HNSW (candidates kept while walking the graph, >= k),
    nprobe for IVF (how many of the nlist clusters to open). FLAT has none."""
    params = {"HNSW": {"ef": ef}, "IVF_FLAT": {"nprobe": nprobe}}.get(index, {})
    return {"metric_type": "COSINE", "params": params}


def count(client: MilvusClient, name: str = COLLECTION) -> int:
    return client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
