"""Check that your machine is ready for the NCP-AAI course labs.

    python setup/check_env.py --mode api     # free mode: NVIDIA API catalog or local Ollama, no GPU
    python setup/check_env.py --mode aws     # AWS-GPU mode (run after aws_lab.py up)

Every check prints PASS or FAIL with a hint. Nothing here costs money in API mode.
"""
import argparse
import importlib.metadata as md
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import llm  # noqa: E402

REQ = pathlib.Path(__file__).parent / "requirements.txt"
NIM_MIN = {"driver": "580", "cuda": "12.9", "docker": "24.0", "container_toolkit": "1.14.0"}
results = []


def check(name, ok, hint=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok or not hint else f"\n       -> {hint}"))
    return ok


def pinned():
    pins = {}
    for line in REQ.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            pkg, ver = line.split("==")
            pins[pkg.strip()] = ver.strip()
    return pins


def common():
    check(f"Python {sys.version.split()[0]} (3.10 or newer)", sys.version_info >= (3, 10),
          "Install Python 3.10+ and recreate the venv.")
    check("Running inside a virtual environment", sys.prefix != sys.base_prefix,
          "Run: python -m venv .venv && source .venv/bin/activate")
    for pkg, want in pinned().items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            have = None
        check(f"{pkg}=={want}", have == want,
              f"Found {have}. Run: pip install -r setup/requirements.txt")


def status_code(err: Exception) -> str:
    m = re.search(r"\[(\d{3})\]", str(err))
    return m.group(1) if m else ""


def tool_call_check(model_label: str, llm_obj):
    """Agents in later modules need tool calling; check the model can do it."""
    from langchain_core.tools import tool

    @tool
    def get_order_status(order_id: str) -> str:
        """Look up the status of a customer order by its ID."""
        return "shipped"

    try:
        msg = llm_obj.bind_tools([get_order_status]).invoke(
            "What is the status of order A123? Use the tool.")
        calls = getattr(msg, "tool_calls", []) or []
        check(f"{model_label} can call tools", any(c.get("name") == "get_order_status" for c in calls),
              "This model didn't return a tool call; pick another model for the agent labs.")
    except Exception as e:
        check(f"{model_label} can call tools", False, f"{type(e).__name__}: {str(e).splitlines()[0]}")


def nvidia_mode():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not check("NVIDIA_API_KEY is set and starts with nvapi-", key.startswith("nvapi-"),
                 "Create a key at build.nvidia.com, then export NVIDIA_API_KEY=... in this "
                 "terminal. Never commit it. No key? Use free local mode: LLM_PROVIDER=ollama"):
        return
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    wanted = os.environ.get("NVIDIA_MODEL")
    candidates = [wanted] if wanted else llm.NVIDIA_MODELS
    codes = {}
    for m in candidates:
        try:
            text = ChatNVIDIA(model=m, max_completion_tokens=16).invoke(
                "Reply with the single word: ready").content.strip()
        except Exception as e:
            codes[m] = status_code(e) or type(e).__name__
            print(f"[INFO] {m}: {'retired by NVIDIA' if codes[m] == '410' else 'not available to this key' if codes[m] == '403' else codes[m]}")
            continue
        check(f"Hosted model {m} answered: {text[:40]!r}", bool(text))
        if not wanted and m != llm.NVIDIA_MODELS[0]:
            print(f"[INFO] Use this model in the labs: export NVIDIA_MODEL={m}")
        tool_call_check(m, ChatNVIDIA(model=m, max_completion_tokens=256))
        return
    if codes and all(c == "403" for c in codes.values()):
        hint = ("Your key is valid but not authorised for NVIDIA's hosted models. Generate a key "
                "from a model page on build.nvidia.com, or use free local mode: LLM_PROVIDER=ollama")
    else:
        hint = f"No listed model worked ({codes}). Try LLM_PROVIDER=ollama for free local mode."
    check("A hosted NVIDIA model answered", False, hint)


def ollama_mode():
    import json
    import urllib.request
    model = llm.model_name("ollama")
    try:
        tags = json.load(urllib.request.urlopen(f"{llm.OLLAMA_URL}/api/tags", timeout=5))
        check(f"Ollama is running at {llm.OLLAMA_URL}", True)
    except Exception:
        check(f"Ollama is running at {llm.OLLAMA_URL}", False,
              "Install Ollama from ollama.com, start it, then run this check again.")
        return
    have = {t.get("name") for t in tags.get("models", [])}
    if not check(f"Model {model} is downloaded", model in have or f"{model}:latest" in have,
                 f"Run: ollama pull {model}"):
        return
    try:
        text = llm.get_llm(model).invoke("Reply with the single word: ready").content.strip()
        check(f"Local model {model} answered: {text[:40]!r}", bool(text))
    except Exception as e:
        check(f"Local model {model} answered", False, f"{type(e).__name__}: {e}")
        return
    tool_call_check(model, llm.get_llm(model))


def api_mode():
    prov = llm.provider()
    print(f"[INFO] Model provider: {prov} (set LLM_PROVIDER=nvidia or ollama to choose)")
    nvidia_mode() if prov == "nvidia" else ollama_mode()


def version_ok(val, minimum):
    """Compare dotted versions as numbers, so 1.9 < 1.14 (a float compare gets this wrong)."""
    def parts(v):
        return tuple(int(x) for x in v.strip().split(".") if x.isdigit())
    try:
        have, need = parts(val), parts(minimum)
        return bool(have) and have >= need
    except ValueError:
        return False


def aws_mode():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import aws_lab
    import boto3
    try:
        who = boto3.client("sts", region_name=aws_lab.REGION).get_caller_identity()
        check(f"AWS credentials work (account {who['Account']})", True)
    except Exception as e:
        check("AWS credentials work", False, f"Run `aws configure` or set a profile. ({e})")
        return
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or boto3.Session().region_name
    check(f"Default region is {aws_lab.REGION} (found {region})", region == aws_lab.REGION,
          f"The course scripts always use {aws_lab.REGION}; set it as your default too.")
    quota = aws_lab.gpu_quota()
    check(f"G and VT quota {quota:g} vCPUs (need {aws_lab.INSTANCE_VCPUS})",
          quota >= aws_lab.INSTANCE_VCPUS,
          f"python setup/aws_lab.py quota --request {aws_lab.INSTANCE_VCPUS * 2}")
    try:
        acct = who["Account"]
        boto3.client("budgets", region_name=aws_lab.REGION).describe_budget(
            AccountId=acct, BudgetName=aws_lab.BUDGET_NAME)
        check(f"Budget '{aws_lab.BUDGET_NAME}' exists", True)
    except Exception:
        check(f"Budget '{aws_lab.BUDGET_NAME}' exists", False,
              "python setup/aws_lab.py budget --email you@example.com")
    iid = aws_lab.stack_instance()
    if not iid:
        print("[INFO] No lab instance running. Run `python setup/aws_lab.py up` to test the GPU.")
        return
    c = aws_lab.console_checks(iid)
    if "DONE" not in c:
        print("[INFO] The instance has not reported its checks yet. Try again in a minute.")
        return
    check(f"GPU: {c.get('gpu')}", "L40S" in c.get("gpu", ""), "Expected 1x NVIDIA L40S (g6e.xlarge).")
    for k, minimum in NIM_MIN.items():
        v = c.get(k, "missing")
        label = "CUDA SDK" if k == "cuda" else k
        check(f"{label} {v or 'missing'} (NIM needs {minimum} or later)", version_ok(v, minimum),
              "Use the Deep Learning Base GPU AMI that aws_lab.py selects.")
    print(f"[INFO] {c.get('os')}; driver supports CUDA up to {c.get('cuda_driver_max')}")
    print("Finished? Delete everything now: python setup/aws_lab.py down")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["api", "aws"], required=True)
    mode = p.parse_args().mode
    print(f"NCP-AAI environment check ({mode} mode)\n")
    common()
    api_mode() if mode == "api" else aws_mode()
    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
