"""Every model call in Module 5 goes through this file: M4's chat() and embed(), plus one provider.

    LLM_PROVIDER=ollama (default)   local Ollama llama3.2:3b, from setup/llm.py (as in M1 to M4)
    LLM_PROVIDER=nvidia             NVIDIA's API catalog, from setup/llm.py
    LLM_PROVIDER=nim                a NIM you run yourself, through its OpenAI-compatible API:
                                    ChatNVIDIA(base_url=NIM_BASE_URL, model=NIM_MODEL)
                                    defaults http://localhost:8000/v1 and meta/llama-3.1-8b-instruct

Only the chat model changes. Embeddings stay on local Ollama embeddinggemma in every mode
(m04/llm_calls.py's embed()), so the M4 index keeps working when chat moves to the NIM.

How the M4 desk picks this up: the M4 modules do `from llm_calls import chat`. Every M5
script imports this file first, so Python has already cached it under the name
`llm_calls`, and the M4 desk, router and critic get this chat(). M4's own file is loaded
under another name (m04_llm_calls) and reused for embed().

Offline self-test only: M05_FAKE_LLM=1 starts tests/fake_oai.py (a scripted server that
speaks Ollama's and the OpenAI API) and points OLLAMA_HOST at it before anything connects.
The code path is the real one; only the replies are scripted.
"""
import atexit
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
NIM_BASE_URL_DEFAULT = "http://localhost:8000/v1"
NIM_MODEL_DEFAULT = "meta/llama-3.1-8b-instruct"


def fake_mode() -> bool:
    return os.environ.get("M05_FAKE_LLM") == "1"


def _start_fake():
    """In fake mode, start the scripted server once and share it with child processes (nat)."""
    if not os.environ.get("M05_FAKE_URL"):
        sys.path.insert(0, str(HERE / "tests"))
        import fake_oai
        proc, url = fake_oai.start()
        atexit.register(proc.kill)
        os.environ["M05_FAKE_URL"] = url
    os.environ["OLLAMA_HOST"] = os.environ["M05_FAKE_URL"]


if fake_mode():
    _start_fake()   # before setup/llm.py and the ollama package read OLLAMA_HOST

sys.path.insert(0, str(LABS / "setup"))
import llm  # noqa: E402  (setup/llm.py: ollama or nvidia)

_spec = importlib.util.spec_from_file_location("m04_llm_calls", LABS / "m04" / "llm_calls.py")
m04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m04)
sys.modules.setdefault("llm_calls", sys.modules[__name__])   # the M4 desk's `from llm_calls import chat` gets this file

embed = m04.embed
normalise = m04.normalise
EMBED_MODEL = m04.EMBED_MODEL
_models = {}


def provider() -> str:
    return "nim" if os.environ.get("LLM_PROVIDER", "").strip().lower() == "nim" else llm.provider()


def nim_base_url() -> str:
    return os.environ.get("NIM_BASE_URL", NIM_BASE_URL_DEFAULT).rstrip("/")


def model_name() -> str:
    return os.environ.get("NIM_MODEL", NIM_MODEL_DEFAULT) if provider() == "nim" else llm.model_name()


def ollama_url() -> str:
    """Ollama's root URL (setup/llm.py reads OLLAMA_HOST); its OpenAI-compatible API is at /v1."""
    return llm.OLLAMA_URL.rstrip("/")


def chat_base_url() -> str:
    """The OpenAI-compatible base URL of the chat model (Ollama's /v1 or the NIM's)."""
    return nim_base_url() if provider() == "nim" else ollama_url() + "/v1"


def describe() -> str:
    where = f" at {nim_base_url()}" if provider() == "nim" else ""
    fake = " [fake server, M05_FAKE_LLM=1]" if fake_mode() else ""
    return f"{provider()} {model_name()}{where}, embeddings ollama {EMBED_MODEL}{fake}"


def chat_model():
    """The LangChain chat model at temperature 0 (the desk and the guardrails share it)."""
    if "chat" not in _models:
        if provider() == "nim":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            _models["chat"] = ChatNVIDIA(base_url=nim_base_url(), model=model_name(), temperature=0)
        else:
            _models["chat"] = llm.get_llm(temperature=0)
    return _models["chat"]


def rails_model():
    """The chat model the guardrails use: the same model as the desk, through an OpenAI-style client.

    Guardrails passes temperature and max_tokens with every call. ChatOllama sends extra call
    arguments straight to the Ollama client, which rejects them, so for Ollama the rails use
    ChatOpenAI on Ollama's OpenAI-compatible /v1 (the same model). ChatNVIDIA takes them as is.
    """
    if "rails" not in _models:
        if provider() == "ollama":
            from langchain_openai import ChatOpenAI
            _models["rails"] = ChatOpenAI(base_url=chat_base_url(), model=model_name(), api_key="ollama", temperature=0)
        else:
            _models["rails"] = chat_model()
    return _models["rails"]


def chat(messages: list[tuple[str, str]], schema=None):
    """Send [(role, text), ...] to the model. With a schema, return an instance of it (as in M4)."""
    model = chat_model()
    if schema is None:
        return model.invoke(messages).content.strip()
    result = model.with_structured_output(schema).invoke(messages)
    if result is None:
        raise ValueError("the model's output did not match the schema")
    return result
