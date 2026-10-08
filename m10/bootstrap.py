"""Shared start-up for every Module 10 script: environment, state folder, the offline fake, the v9 desk.

Every m10 script does `import bootstrap` first. It
  1. pins the same two environment variables as Module 9 (KMP_DUPLICATE_LIB_OK for macOS's libomp clash,
     LLM_PROVIDER=ollama so the desk stays on the local 3B even when NVIDIA_API_KEY is set)
  2. points the v9 desk's state at m10/state/ (M10_STATE_DIR overrides it; check.py uses m10/state/check):
     the audit log, M8's request log, the sessions, the desk's index and memory all live there, so
     Module 9's own results in m09/state/ are never touched
  3. reads the desk version: v10, or v10.1 after `feedback_loop.py fix` re-indexed the reviewed FAQ
  4. offline self-test only (M10_FAKE_LLM=1): starts m10/tests/fake_oai.py and hands it to Module 9
  5. imports Module 9's guarded desk (which imports M8 -> M6 -> M5 -> M4) and widens its audit log by two
     fields: `route` (the desk's plan, which v9 computed but did not keep) and `approval` (reviewer
     decisions). Nothing in m09/ is edited: the "wrap, don't change" rule of Modules 8 and 9.
"""
import json
import os
import pathlib
import sys

# macOS: torch, scikit-learn and faiss each ship their own libomp (see Module 9). Set before any import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# The desk runs on the local 3B; setup/llm.py would otherwise pick NVIDIA's API when NVIDIA_API_KEY is set.
os.environ.setdefault("LLM_PROVIDER", "ollama")

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = pathlib.Path(os.environ.get("M10_STATE_DIR", HERE / "state"))
VERSION_FILE = STATE / "feedback" / "version.json"
FAKE = os.environ.get("M10_FAKE_LLM") == "1"


def desk_version() -> str:
    """v10, or what `feedback_loop.py fix` wrote (v10.1 once the reviewed FAQ is in the index)."""
    try:
        return json.loads(VERSION_FILE.read_text())["version"]
    except (FileNotFoundError, ValueError, KeyError):
        return "v10"


VERSION = desk_version()
os.environ["M09_STATE_DIR"] = str(STATE)          # read by guarded_desk, audit_log and escalate at import
os.environ["DESK_VERSION"] = VERSION              # M8's metrics label
STATE.mkdir(parents=True, exist_ok=True)


def _start_fake() -> None:
    """Offline self-test: one scripted server for every model call, shared with child processes."""
    if not os.environ.get("M10_FAKE_URL"):
        import atexit
        sys.path.insert(0, str(HERE / "tests"))
        import fake_oai as m10_fake
        proc, url = m10_fake.start()
        atexit.register(proc.kill)
        os.environ["M10_FAKE_URL"] = url
    os.environ["M09_FAKE_LLM"] = "1"
    os.environ["M09_FAKE_URL"] = os.environ["M10_FAKE_URL"]


if FAKE:
    _start_fake()

sys.path.insert(0, str(HERE))
sys.path.insert(1, str(LABS / "m09"))
import guarded_desk as gd  # noqa: E402  (Module 9's desk; imports M8, M6, M5, M4)
import audit_log  # noqa: E402  (m09)
import escalate  # noqa: E402  (m09)

gd.VERSION = VERSION                     # the desk_version field of every audit record, the x-version header
audit_log.DESK_VERSION = VERSION         # the prefix of config_version
audit_log.FIELDS = audit_log.FIELDS + ("route", "approval")     # the desk's route was computed but not kept
audit_log.ACTIONS = audit_log.ACTIONS + ("approval",)
_m9_write = audit_log.write


def _audit_write(record: dict) -> dict:
    """M9's write() (masking, append-only) plus the reviewer's free text masked too.
    The reviewer's name and the refund arguments stay readable: they are what the record is for."""
    rec = dict(record)
    appr = rec.get("approval")
    if isinstance(appr, dict) and appr.get("message"):
        rec["approval"] = {**appr, "message": audit_log.mask(appr["message"])}
    return _m9_write(rec)


audit_log.write = _audit_write           # guarded_desk calls audit_log.write(...) at run time
