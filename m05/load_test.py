"""A small concurrency sweep: latency and throughput of one chat model at 1, 4, 8 parallel requests.

    python m05/load_test.py                                   # Ollama, concurrency 1,4,8, 16 requests each
    python m05/load_test.py --concurrency 1,4 --requests 8
    python m05/load_test.py --base-url http://localhost:8000/v1 --model meta/llama-3.1-8b-instruct --metrics

For each concurrency level the script keeps that many requests in flight until N requests
are done, all with the same short prompt and max_tokens. Every request streams, so it can
time the first token (TTFT: mostly prefill) apart from the whole reply (prefill + decode).

    p50 / p95        median and 95th-percentile latency of one request, in seconds
    ttft p50         median time to the first streamed token
    tokens/s         completion tokens of all requests / wall-clock time (usage from the server)
    requests/s       requests / wall-clock time

The rows go to m05/state/load_test.csv (one file per run; --out to change).

On Ollama, how many requests one model serves at once is OLLAMA_NUM_PARALLEL, set when
you start `ollama serve`; the rest wait in a queue, and each parallel slot needs its own
context memory. That is not TensorRT-LLM's in-flight batching, and Mac numbers say nothing
about GPU serving. --metrics saves the NIM's Prometheus metrics (/v1/metrics) before and
after the sweep; Ollama has no such endpoint.
"""
import argparse
import asyncio
import csv
import pathlib
import statistics
import sys
import time

import httpx
from openai import AsyncOpenAI

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_calls  # noqa: E402

OUT = HERE / "state" / "load_test.csv"
PROMPT = "In two sentences, explain what a USB-C dock does."
FIELDS = ["concurrency", "requests", "ok", "p50_s", "p95_s", "ttft_p50_s", "tokens_per_s", "requests_per_s"]


async def one(client: AsyncOpenAI, model: str, max_tokens: int) -> dict:
    start, first, tokens = time.perf_counter(), None, 0
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}], max_tokens=max_tokens, temperature=0,
        stream=True, stream_options={"include_usage": True})
    async for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter()
        if chunk.usage:
            tokens = chunk.usage.completion_tokens
    end = time.perf_counter()
    return {"latency": end - start, "ttft": (first or end) - start, "tokens": tokens}


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


async def level(base_url: str, model: str, concurrency: int, n: int, max_tokens: int) -> dict:
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=300)   # one client per event loop
    sem = asyncio.Semaphore(concurrency)

    async def limited():
        async with sem:
            try:
                return await one(client, model, max_tokens)
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}

    t0 = time.perf_counter()
    results = await asyncio.gather(*(limited() for _ in range(n)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r]
    lat = [r["latency"] for r in ok]
    return {"concurrency": concurrency, "requests": n, "ok": len(ok),
            "p50_s": round(statistics.median(lat), 3) if lat else float("nan"), "p95_s": round(pct(lat, 95), 3),
            "ttft_p50_s": round(statistics.median(r["ttft"] for r in ok), 3) if ok else float("nan"),
            "tokens_per_s": round(sum(r["tokens"] for r in ok) / wall, 1),
            "requests_per_s": round(len(ok) / wall, 2),
            "errors": [r["error"] for r in results if "error" in r][:1]}


def metrics(base_url: str, label: str, say=print) -> None:
    path = HERE / "state" / f"metrics_{label}.txt"
    try:
        resp = httpx.get(f"{base_url}/metrics", timeout=10)
    except httpx.HTTPError as e:
        say(f"[metrics] {label}: {type(e).__name__}")
        return
    if resp.status_code != 200:
        say(f"[metrics] {label}: GET /v1/metrics -> {resp.status_code} (not a NIM?)")
        return
    path.write_text(resp.text)
    samples = [l for l in resp.text.splitlines() if l and not l.startswith("#")]
    say(f"[metrics] {label}: {len(samples)} samples saved to {path}")


def run(base_url: str, model: str, levels: list[int], n: int, max_tokens: int = 64,
        out: pathlib.Path = OUT, say=print) -> list[dict]:
    rows = []
    for c in levels:
        row = asyncio.run(level(base_url.rstrip("/"), model, c, n, max_tokens))
        rows.append(row)
        say(f"{row['concurrency']:>11} {row['ok']:>3}/{row['requests']:<3} {row['p50_s']:>7} {row['p95_s']:>7} "
            f"{row['ttft_p50_s']:>9} {row['tokens_per_s']:>9} {row['requests_per_s']:>10}")
        if row["errors"]:
            say(f"            first error: {row['errors'][0][:120]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    base, model = llm_calls.chat_base_url(), llm_calls.model_name()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=base)
    ap.add_argument("--model", default=model)
    ap.add_argument("--concurrency", default="1,4,8", help="comma-separated levels")
    ap.add_argument("--requests", type=int, default=16, help="requests per level")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--metrics", action="store_true", help="save /v1/metrics before and after (NIM only)")
    a = ap.parse_args()
    levels = [int(x) for x in a.concurrency.split(",")]
    print(f"[INFO] {a.base_url} model {a.model} | {a.requests} requests per level, max_tokens {a.max_tokens}")
    if a.metrics:
        metrics(a.base_url.rstrip("/"), "before")
    print("concurrency  ok/n     p50     p95  ttft p50  tokens/s  requests/s")
    run(a.base_url, a.model, levels, a.requests, a.max_tokens, a.out)
    if a.metrics:
        metrics(a.base_url.rstrip("/"), "after")
    print(f"[INFO] rows written to {a.out}")


if __name__ == "__main__":
    main()
