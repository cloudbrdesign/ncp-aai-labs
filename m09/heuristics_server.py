"""The jailbreak heuristics as their own process: NeMo Guardrails' jailbreak detection server.

    python m09/heuristics_server.py                 # http://127.0.0.1:1337/heuristics (leave it running)
    python m09/heuristics_server.py --port 1338

GPT-2 large (the perplexity model) lives here, so the desk process never loads PyTorch. That is how the
Guardrails docs recommend running the heuristics in production (in-process "is not recommended for
production"), and on macOS it also keeps PyTorch's OpenMP runtime out of the process that holds faiss and
scikit-learn (three copies of libomp in one process crash it).

This runs the library's own server (nemoguardrails/library/jailbreak_detection/server.py, endpoints
/heuristics, /jailbreak_lp_heuristic, /jailbreak_ps_heuristic) without its model-based classifier.
The m09 scripts find it on 127.0.0.1:1337 by themselves, or set M09_HEURISTICS_URL.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pathlib
import sys

import nemoguardrails.library.jailbreak_detection as jd

sys.path.insert(0, str(pathlib.Path(jd.__file__).parent))   # the server imports heuristics.checks by that name
import server  # noqa: E402  (loads GPT-2 large: about 3 GB, a few seconds once downloaded)
import uvicorn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1337)
    a = ap.parse_args()
    print(f"[heuristics] GPT-2 loaded; serving http://{a.host}:{a.port}/heuristics", flush=True)
    uvicorn.run(server.app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
