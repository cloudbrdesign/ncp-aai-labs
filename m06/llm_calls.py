"""Every model call in Module 6 goes through this file: M5's model switch, plus the desk's sampling
settings and the judge model.

    the desk     M5's chat() and embed() (m05/llm_calls.py): Ollama llama3.2:3b by default,
                 OLLAMA_MODEL picks another one (configuration B is OLLAMA_MODEL=llama3.2:1b)
    sampling     DESK_TEMPERATURE (default 0, as in M5) and DESK_SEED (default: none) set the
                 desk's temperature and seed. Ollama's recipe for reproducible output is a
                 fixed seed with temperature 0.
    the judge    M06_JUDGE_MODEL (default qwen3:4b) through Ollama's OpenAI-compatible /v1,
                 with thinking switched off (see judge_client() below).

How the desk picks this up: the M4 modules do `from llm_calls import chat`. Every M6 script
imports this file first, so Python has it cached under the name `llm_calls`, and the M4
desk (through M5's desk_app) gets this chat(). M5's own file is loaded under another name
(m05_llm_calls) and reused for the providers and embed().

Offline self-test only: M06_FAKE_LLM=1 starts tests/fake_oai.py (M5's scripted server plus
schema-driven JSON replies for Ragas) and points OLLAMA_HOST and the judge at it before
anything connects. The code path is the real one; only the replies are scripted.
"""
import atexit
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
JUDGE_DEFAULT = "qwen3:4b"
SELF_JUDGE = "llama3.2:3b"            # the desk's own model, as a second judge in judge_check.py
THINKING_PREFIXES = ("qwen3",)        # models that think unless told not to (as in setup/llm.py)


def fake_mode() -> bool:
    return os.environ.get("M06_FAKE_LLM") == "1"


def _start_fake():
    """In fake mode, start the scripted server once and share it with child processes (nat)."""
    if not os.environ.get("M06_FAKE_URL"):
        sys.path.insert(0, str(HERE / "tests"))
        import fake_oai
        proc, url = fake_oai.start()
        atexit.register(proc.kill)
        os.environ["M06_FAKE_URL"] = url
    url = os.environ["M06_FAKE_URL"]
    os.environ["OLLAMA_HOST"] = url
    os.environ["M05_FAKE_URL"] = url                  # M5's own fake switch stays off; its URL is ours
    os.environ.setdefault("M06_JUDGE_BASE_URL", url + "/v1")


if fake_mode():
    _start_fake()   # before setup/llm.py and the ollama package read OLLAMA_HOST

sys.modules["llm_calls"] = sys.modules[__name__]   # the M4 desk's `from llm_calls import chat` gets this file
_spec = importlib.util.spec_from_file_location("m05_llm_calls", LABS / "m05" / "llm_calls.py")
m05 = importlib.util.module_from_spec(_spec)
sys.modules["m05_llm_calls"] = m05
_spec.loader.exec_module(m05)

llm = m05.llm                     # setup/llm.py
embed = m05.embed
normalise = m05.normalise
EMBED_MODEL = m05.EMBED_MODEL
provider = m05.provider
model_name = m05.model_name
ollama_url = m05.ollama_url
chat_base_url = m05.chat_base_url
nim_base_url = m05.nim_base_url
_models = {}


def temperature() -> float:
    return float(os.environ.get("DESK_TEMPERATURE", "0"))


def seed() -> int | None:
    s = os.environ.get("DESK_SEED", "").strip()
    return int(s) if s else None


def describe() -> str:
    fake = " [fake server, M06_FAKE_LLM=1]" if fake_mode() else ""
    sampling = f"temperature {temperature():g}" + (f", seed {seed()}" if seed() is not None else "")
    return (f"{provider()} {model_name()} ({sampling}), embeddings ollama {EMBED_MODEL}, "
            f"judge {judge_model()}{fake}")


def chat_model():
    """The desk's LangChain chat model with DESK_TEMPERATURE and DESK_SEED."""
    key = ("chat", provider(), model_name(), temperature(), seed())
    if key not in _models:
        extra = {"seed": seed()} if seed() is not None else {}
        if provider() == "nim":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            _models[key] = ChatNVIDIA(base_url=nim_base_url(), model=model_name(), temperature=temperature(), **extra)
        else:
            _models[key] = llm.get_llm(temperature=temperature(), **extra)   # ChatOllama(temperature=, seed=)
    return _models[key]


def chat(messages: list[tuple[str, str]], schema=None):
    """Send [(role, text), ...] to the desk's model. With a schema, return an instance of it (as in M4)."""
    model = chat_model()
    if schema is None:
        return model.invoke(messages).content.strip()
    result = model.with_structured_output(schema).invoke(messages)
    if result is None:
        raise ValueError("the model's output did not match the schema")
    return result


# ---- the judge ------------------------------------------------------------------------

def judge_model() -> str:
    return os.environ.get("M06_JUDGE_MODEL", JUDGE_DEFAULT)


def judge_base_url() -> str:
    """Ollama's OpenAI-compatible API; the judge runs on the same Ollama as the desk."""
    return os.environ.get("M06_JUDGE_BASE_URL", ollama_url() + "/v1").rstrip("/")


def thinks(model: str) -> bool:
    return model.startswith(THINKING_PREFIXES)


def judge_args(model: str) -> dict:
    """Extra request fields for a judge call.

    qwen3 thinks before it answers unless told not to. Ollama's OpenAI-compatible API takes
    `reasoning_effort` ("none" requests no thinking for models with an on/off switch); the
    openai client (2.54.0) sends it as a normal request field. Thinking would spend the
    judge's small token budget before the JSON verdict and make every call slower.
    Models that don't think get nothing extra.
    """
    return {"reasoning_effort": "none"} if thinks(model) else {}


def judge_client(async_client: bool = False, http_client=None):
    """An OpenAI client for the judge (Ollama ignores the API key, but the client needs one)."""
    from openai import AsyncOpenAI, OpenAI
    cls = AsyncOpenAI if async_client else OpenAI
    kwargs = {"http_client": http_client} if http_client is not None else {}
    return cls(base_url=judge_base_url(), api_key="ollama", **kwargs)
