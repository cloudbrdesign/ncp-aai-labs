"""One place that decides which chat model the labs use.

    LLM_PROVIDER=nvidia   NVIDIA API catalog (needs NVIDIA_API_KEY from build.nvidia.com)
    LLM_PROVIDER=ollama   a free open model running locally with Ollama (no key needed)
    (not set)             nvidia if NVIDIA_API_KEY is set, otherwise ollama

    NVIDIA_MODEL / OLLAMA_MODEL override the default model for that provider.

Every lab gets its model from get_llm(), so the lab code is the same either way.
"""
import os

# Hosted models change: NVIDIA retires them (HTTP 410) and some accounts can't use
# every model (HTTP 403). check_env.py tries these in order and tells you which one works.
NVIDIA_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-nano-3-30b-a3b",
]
OLLAMA_DEFAULT = "llama3.2:3b"   # tested: clean answers, reliable tool calls, fast on a laptop
# Reasoning models (e.g. qwen3) think before answering. Ollama returns that thinking
# separately only when reasoning is switched on, so we switch it on for these models.
THINKING_PREFIXES = ("qwen3",)
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p in ("nvidia", "ollama"):
        return p
    return "nvidia" if os.environ.get("NVIDIA_API_KEY") else "ollama"


def model_name(prov: str | None = None) -> str:
    prov = prov or provider()
    if prov == "nvidia":
        return os.environ.get("NVIDIA_MODEL") or NVIDIA_MODELS[0]
    return os.environ.get("OLLAMA_MODEL") or OLLAMA_DEFAULT


def get_llm(model: str | None = None, **kwargs):
    """Return a LangChain chat model for the configured provider."""
    prov = provider()
    model = model or model_name(prov)
    if prov == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(model=model, **kwargs)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=OLLAMA_URL, **ollama_options(model), **kwargs)


def ollama_options(model: str) -> dict:
    """Keep the model's thinking out of the answer text."""
    if model.startswith(THINKING_PREFIXES):
        return {"reasoning": True, "num_predict": 2048}
    return {"reasoning": False}
