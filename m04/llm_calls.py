"""Every model call in Module 4 goes through this file: chat() and embed().

    chat(messages)               -> the reply text           (setup/llm.py: Ollama llama3.2:3b or NVIDIA)
    chat(messages, schema=Route) -> a Route object           (structured output, temperature 0)
    embed(["text", ...])         -> one vector per text      (Ollama embeddinggemma, L2-normalised)

Embeddings always come from local Ollama (`ollama pull embeddinggemma`), also when the
chat model comes from NVIDIA's API catalog. The same embedding model must be used to
index the manuals and to embed every question: vectors from two models don't compare.
Set M04_EMBED_MODEL to try another Ollama embedding model, then re-run ingest.py.

The vectors are L2-normalised (length 1). For unit vectors, cosine similarity and inner
product (IP) give the same ranking, so the COSINE index here and an IP index (which GPU
indexes need) would return the same results.

m03's critic.py is reused by the M4 desk and does `from llm_calls import chat`. Because
m04 comes first on sys.path, that import gets this file, so M4 has one model switch.

Offline self-test only: M04_FAKE_LLM=1 swaps in tests/fake_llm.py (a scripted chat model
and a hashed bag-of-words embedder), so check.py runs without Ollama.
"""
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "setup"))
import llm  # noqa: E402  (setup/llm.py picks NVIDIA or Ollama for chat)

EMBED_MODEL = os.environ.get("M04_EMBED_MODEL", "embeddinggemma")
_models = {}


def fake_mode() -> bool:
    return os.environ.get("M04_FAKE_LLM") == "1"


def _fake():
    sys.path.insert(0, str(HERE / "tests"))
    import fake_llm
    return fake_llm


def describe() -> str:
    if fake_mode():
        return "fake (M04_FAKE_LLM=1)"
    return f"{llm.provider()} {llm.model_name()}, embeddings ollama {EMBED_MODEL}"


def chat(messages: list[tuple[str, str]], schema=None):
    """Send [(role, text), ...] to the model at temperature 0. With a schema, return an instance of it."""
    if fake_mode():
        return _fake().chat(messages, schema)
    if "chat" not in _models:
        _models["chat"] = llm.get_llm(temperature=0)
    model = _models["chat"]
    if schema is None:
        return model.invoke(messages).content.strip()
    result = model.with_structured_output(schema).invoke(messages)
    if result is None:
        raise ValueError("the model's output did not match the schema")
    return result


def normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embed texts in batches (one Ollama call per batch). Returns unit-length vectors."""
    if fake_mode():
        return [normalise(v) for v in _fake().embed(texts)]
    import ollama   # reads OLLAMA_HOST, like setup/llm.py
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors += ollama.embed(model=EMBED_MODEL, input=texts[i:i + batch_size]).embeddings
    return [normalise(list(v)) for v in vectors]
