"""The safety layers L0 to L4: subsets of rails/config.yml, cheapest first, and how a rails check is run.

    from layers import build, check
    rails = build("L2")                         # an LLMRails with the L1 and L2 flows
    res = await check(rails, [{"role": "user", "content": "..."}], "input")

Each layer adds flows to the one before it (L4 has everything), in the order config.yml lists them:

    L0  no rails: the v9 desk as it is
    L1  rules       regex (secrets in, canary/secrets/card/SSN out), context bloat, Presidio masking in and out
    L2  jailbreak   + perplexity heuristics (in-process) + self check input        (--hosted: NemoGuard JailbreakDetect)
    L3  content     + content safety input and output (local: the 3B with the NemoGuard prompt;
                      --hosted: nvidia/llama-3.1-nemotron-safety-guard-8b-v3 on build.nvidia.com)
    L4  output      + YARA injection detection + self check output

Input order: the masking rail runs before every rail that sends text to a model, so the model rails (and a
hosted provider) see the masked text. Output order: masking runs last, so the reply's names and contact data
are masked whatever the other rails let through.

check() does what LLMRails.check_async() does in 0.24.1 (generate_async with options.rails set to the one
rail type; BLOCKED when an activated rail stopped, MODIFIED when the text changed) and also keeps
options.log: the activated rails with their decisions and timing, and every LLM call, for the audit log and
for redteam.py. It also notices two failure signals the library does not return to the caller:
  rail error     the rail's action raised; Guardrails answers "an internal error has occurred" and stops
                 the turn (fail closed, by accident of the error handling)
  unavailable    a detector reported that it could not be reached (the jailbreak rails log this and then
                 allow the request: fail open). Caught from the library's log records.
  raised         generate_async raised (an LLM rail whose model is unreachable raises LLMCallException
                 to the caller): status "error"; without handling, the app's request fails.
FAIL_POLICY is the application's decision per rail when that happens; guarded_desk.py applies it.
"""
import copy
import logging
import os
# macOS: torch, scikit-learn and faiss each ship their own libomp; without this, loading GPT-2 for the
# jailbreak heuristics aborts Python with "OMP: Error #15". Set before any of them is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# The desk runs on the local 3B (setup/llm.py would pick NVIDIA's API whenever NVIDIA_API_KEY is set);
# the key is only for the --hosted safety rails. LLM_PROVIDER=nvidia still overrides this.
os.environ.setdefault("LLM_PROVIDER", "ollama")
import pathlib
import time

import yaml

HERE = pathlib.Path(__file__).resolve().parent
LAYERS = ("L0", "L1", "L2", "L3", "L4")
NAMES = {"L0": "none", "L1": "rules", "L2": "jailbreak", "L3": "content safety", "L4": "output checks"}
CS_IN = "content safety check input $model=content_safety"
CS_OUT = "content safety check output $model=content_safety"
LAYER_OF = {
    "regex check input": 1, "context bloat detection on input": 1, "mask sensitive data on input": 1,
    "jailbreak detection heuristics": 2, "self check input": 2, CS_IN: 3,
    "regex check output": 1, CS_OUT: 3, "injection detection": 4, "self check output": 4,
    "mask sensitive data on output": 1,
}
HOSTED_SAFETY_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
HOSTED_JAILBREAK = {"nim_base_url": "https://ai.api.nvidia.com",
                    "nim_server_endpoint": "/v1/security/nvidia/nemoguard-jailbreak-detect",
                    "api_key_env_var": "NVIDIA_API_KEY"}
GLINER_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
INTERNAL_ERROR = "an internal error has occurred"
# What the app does when a rail cannot decide (error or detector unreachable): "closed" refuses the turn,
# "open" lets it through (the turn is still marked "rail unavailable" in the audit log).
FAIL_POLICY = {"jailbreak detection heuristics": "closed", "jailbreak detection model": "closed",
               CS_IN: "closed", CS_OUT: "closed", "self check input": "closed", "self check output": "closed",
               "injection detection": "closed", "mask sensitive data on input": "closed",
               "mask sensitive data on output": "closed"}
UNAVAILABLE = ("Jailbreak endpoint not set up properly", "Jailbreak check API request failed",
               "NemoGuard JailbreakDetect NIM", "Jailbreak detection model not available")


def raw_config() -> dict:
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    cfg.update(yaml.safe_load((HERE / "prompts.yml").read_text()))
    return cfg


def flows(layer: str, hosted: bool = False, gliner: bool = False) -> dict:
    """{"input": [...], "output": [...]}: the flows of a layer, in config.yml's order."""
    n = LAYERS.index(layer)
    cfg = raw_config()["rails"]
    out = {}
    for kind in ("input", "output"):
        names = [f for f in cfg[kind]["flows"] if LAYER_OF[f] <= n]
        if hosted:
            names = ["jailbreak detection model" if f == "jailbreak detection heuristics" else f for f in names]
        if gliner:
            names = [f.replace("mask sensitive data on", "gliner mask pii on") for f in names]
        out[kind] = names
    return out


def config_dict(layer: str = "L4", hosted: bool = False, gliner: bool = False, ollama_url: str | None = None,
                fake_url: str | None = None, heuristics_endpoint: str | None = None) -> dict:
    """The Guardrails config of one layer as a dict.

    ollama_url    replaces the models' base URLs (OLLAMA_HOST, as in M5)
    hosted        the NVIDIA models on build.nvidia.com (NVIDIA_API_KEY) for L2 and L3
    fake_url      offline self-test: the hosted endpoints go to the scripted server too
    heuristics_endpoint  run the jailbreak heuristics as a server at this URL (the provider-down drill)
    """
    cfg = copy.deepcopy(raw_config())
    f = flows(layer, hosted, gliner)
    cfg["rails"]["input"]["flows"] = f["input"]
    cfg["rails"]["output"]["flows"] = f["output"]
    base = (fake_url or ollama_url or "http://localhost:11434").rstrip("/") + "/v1"
    for m in cfg["models"]:
        m.setdefault("parameters", {})["base_url"] = base
    jb = cfg["rails"]["config"]["jailbreak_detection"]
    if heuristics_endpoint:
        jb["server_endpoint"] = heuristics_endpoint
    if hosted:
        cs = next(m for m in cfg["models"] if m["type"] == "content_safety")
        cs.update(engine="nim", model=HOSTED_SAFETY_MODEL)
        cs["parameters"] = {"base_url": fake_url.rstrip("/") + "/v1"} if fake_url else {}
        jb.update(HOSTED_JAILBREAK)
        if fake_url:
            jb["nim_base_url"] = fake_url
    if gliner:
        cfg["rails"]["config"]["gliner"] = {
            "server_endpoint": GLINER_ENDPOINT, "api_key_env_var": "NVIDIA_API_KEY",
            "input": {"entities": ["email", "phone_number", "credit_debit_card", "ssn"]},
            "output": {"entities": ["email", "phone_number", "credit_debit_card", "ssn", "first_name", "last_name"]}}
    if not f["input"]:
        cfg["rails"].pop("input")
    if not f["output"]:
        cfg["rails"].pop("output")
    return cfg


def build(layer: str = "L4", llm=None, register: dict | None = None, **kw):
    """An LLMRails for one layer (None for L0). llm: the main model (the desk's 3B); register: extra actions."""
    if layer == "L0":
        return None
    from nemoguardrails import LLMRails, RailsConfig
    rails = LLMRails(RailsConfig.from_content(config=config_dict(layer, **kw)), llm=llm)
    for name, fn in (register or {}).items():
        rails.register_action(fn, name=name)
    return rails


def has(rails, kind: str) -> bool:
    return rails is not None and bool(getattr(rails.config.rails, kind).flows)


class _Catch(logging.Handler):
    """Collects the library's warnings and errors during one check."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage()[:300])


def _violations(action) -> list[str]:
    rv = action.return_value
    meta = getattr(rv, "metadata", None) or (rv.get("metadata") if isinstance(rv, dict) else None) or {}
    v = meta.get("policy_violations") if isinstance(meta, dict) else None
    return [str(x) for x in v] if v else []


async def check(rails, messages: list[dict], kind: str) -> dict:
    """Run one rail type ("input" or "output") on messages; return the decision and the evidence.

    {"status": passed|modified|blocked|error, "content", "rail" (the one that stopped), "rails": [{name, decisions,
     stop, duration_s, error}], "llm_calls": [task, ...], "categories": [...], "errors": [rail, ...],
     "unavailable": [rail, ...], "duration_s"}
    """
    original = next((m["content"] for m in reversed(messages) if m["role"] == ("assistant" if kind == "output" else "user")), "")
    if not has(rails, kind):
        return {"status": "passed", "content": original, "rail": None, "rails": [], "llm_calls": [], "categories": [],
                "errors": [], "unavailable": [], "duration_s": 0.0}
    if kind == "output" and not any(m["role"] == "user" for m in messages):
        messages = [{"role": "user", "content": ""}] + messages
    catch = _Catch()
    root = logging.getLogger("nemoguardrails")
    root.addHandler(catch)
    t = time.perf_counter()
    try:
        res = await rails.generate_async(messages=messages, options={
            "rails": [kind], "log": {"activated_rails": True, "llm_calls": True}})
    except Exception as e:          # e.g. LLMCallException: a model rail's model can't be reached
        return {"status": "error", "content": original, "rail": None, "rails": [], "llm_calls": [], "categories": [],
                "errors": [f"{kind} rails"], "unavailable": [], "duration_s": round(time.perf_counter() - t, 4),
                "raised": f"{type(e).__name__}: {str(e)[:300]}", "log_warnings": catch.records[:5]}
    finally:
        root.removeHandler(catch)
    took = time.perf_counter() - t
    content = res.response[-1]["content"] if isinstance(res.response, list) and res.response else str(res.response or "")
    activated, categories, errors, blocked_by = [], [], [], None
    for a in res.log.activated_rails if res.log else []:
        err = a.stop and INTERNAL_ERROR in content and all(e.return_value is None for e in a.executed_actions[:1])
        activated.append({"type": a.type, "name": a.name, "decisions": a.decisions, "stop": a.stop,
                          "duration_s": round(a.duration or 0.0, 4), "error": bool(err)})
        for e in a.executed_actions:
            categories += _violations(e)
        if err:
            errors.append(a.name)
        if a.stop and blocked_by is None:
            blocked_by = a.name
    ran = [a["name"] for a in activated]
    unavailable = [n for n in ran if n.startswith("jailbreak detection")
                   and any(any(u in r for u in UNAVAILABLE) for r in catch.records)]
    status = "blocked" if blocked_by else ("modified" if content != original else "passed")
    calls = [c.task for c in (res.log.llm_calls or [])] if res.log else []
    return {"status": status, "content": content, "rail": blocked_by, "rails": activated, "llm_calls": calls,
            "categories": sorted(set(categories)), "errors": errors, "unavailable": unavailable,
            "duration_s": round(took, 4), "log_warnings": catch.records[:5]}


def describe(layer: str, hosted: bool = False) -> str:
    f = flows(layer, hosted)
    return f"{layer} ({NAMES[layer]}): input {f['input'] or '-'}; output {f['output'] or '-'}"


if __name__ == "__main__":
    import sys
    hosted = "--hosted" in sys.argv or os.environ.get("M09_HOSTED") == "1"
    for layer in LAYERS:
        print(describe(layer, hosted))
