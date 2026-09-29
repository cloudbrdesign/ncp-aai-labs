"""Every model call in Module 3 goes through chat(), so there is exactly one place to look.

The model comes from setup/llm.py, as in M1 and M2: local Ollama (llama3.2:3b) by
default, or NVIDIA's API catalog with LLM_PROVIDER=nvidia.

    chat(messages)                -> the reply text
    chat(messages, schema=Plan)   -> a Plan object (structured output, temperature 0)

Structured output means the model has to fill in a Pydantic schema. With Ollama the
schema is sent as the response format, so a small model can only produce JSON of that
shape. chat() raises when the model fails; every caller catches that and uses a
deterministic fallback, and says so in its log.

Offline self-test only: M03_FAKE_LLM=1 swaps in the scripted fake in tests/fake_llm.py.
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "setup"))
import llm  # noqa: E402  (setup/llm.py picks NVIDIA or Ollama)

_models = {}


def fake_mode() -> bool:
    return os.environ.get("M03_FAKE_LLM") == "1"


def describe() -> str:
    return "fake (M03_FAKE_LLM=1)" if fake_mode() else f"{llm.provider()} {llm.model_name()}"


def chat(messages: list[tuple[str, str]], schema=None):
    """Send [(role, text), ...] to the model at temperature 0. With a schema, return an instance of it."""
    if fake_mode():
        sys.path.insert(0, str(HERE / "tests"))
        import fake_llm
        return fake_llm.chat(messages, schema)
    if "chat" not in _models:
        _models["chat"] = llm.get_llm(temperature=0)
    model = _models["chat"]
    if schema is None:
        return model.invoke(messages).content.strip()
    result = model.with_structured_output(schema).invoke(messages)
    if result is None:
        raise ValueError("the model's output did not match the schema")
    return result
