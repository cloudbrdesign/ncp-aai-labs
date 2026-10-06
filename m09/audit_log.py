"""The desk's audit log: one JSON line per turn in m09/state/audit/audit.jsonl, masked before it is written.

    python m09/audit_log.py query --request-id 3f2a9c1e0b7d4e21      # one turn, field by field
    python m09/audit_log.py query --session s-42                      # every turn of a session
    python m09/audit_log.py query --action blocked --last 5           # answered | masked | blocked | escalated |
                                                                      #   rail unavailable | locked | error
    python m09/audit_log.py purge --older-than 180                    # retention: drop records older than 180 days
    python m09/audit_log.py stats                                     # counts by action and by layer

What one record holds (guarded_desk.py fills it):
  time, ts                  when (UTC ISO and Unix seconds; purge uses ts)
  request_id, trace_id      the turn's IDs: x-request-id (M8) or a new one; the desk logs the same IDs in
                            m09/state/logstore/requests.jsonl for every model call (M8's request log)
  config_version            "v9+" and a hash of rails/config.yml, prompts.yml and layers.py: which policy decided
  desk_version, model       the desk release and its model
  layer, hosted             which rails ran (L0-L4) and whether the NVIDIA hosted safety models were on
  caller, scope, session_id who asked (x-customer), what the tools could see, and the conversation
  input                     the customer's message, masked
  desk_input                what the desk got after the input rails (masked again here, in case L0 ran)
  input_rails, output_rails per rail: name, decisions, stop, duration, error; plus status, the rail that
                            stopped the turn, content-safety categories, rails that errored or were unreachable
  model_calls               the rails' LLM calls (task names) and the desk's model calls (count)
  timing_s                  input rails, desk, output rails, total
  tool_calls                tool, parameters, result summary (masked), error
  action                    answered | masked | blocked | escalated | rail unavailable | locked | error
  blocked_by, categories    the rail that stopped the turn; content-safety categories
  escalation, ticket_id     why the turn went to a person, and the ticket (M2's ticket API)
  reply                     what the customer got (masked)

Masking runs before the write, on every free-text field: Presidio (the same spaCy model as the rails, the
union of the rails' entity lists) plus regular expressions for card, SSN, IBAN, email and phone shapes, so
the log holds no raw PII even for turns that ran without rails (L0). NeMo Guardrails' own masking rails do
not clean application logs: that is this file's job.

Append-only: write() only appends; the one exception is purge (retention), which rewrites the file without
the expired records and logs what it removed in m09/state/audit/purge.jsonl. No hash chain (lab decision).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
STATE = pathlib.Path(os.environ.get("M09_STATE_DIR", HERE / "state"))   # check.py uses its own
AUDIT_DIR = STATE / "audit"
AUDIT = AUDIT_DIR / "audit.jsonl"
PURGES = AUDIT_DIR / "purge.jsonl"
DESK_VERSION = "v9"
FIELDS = ("time", "ts", "request_id", "trace_id", "config_version", "desk_version", "model", "layer", "hosted",
          "caller", "scope", "session_id", "input", "desk_input", "input_rails", "output_rails", "model_calls",
          "timing_s", "tool_calls", "action", "blocked_by", "categories", "escalation", "ticket_id", "reply")
TEXT_FIELDS = ("input", "desk_input", "reply")
ACTIONS = ("answered", "masked", "blocked", "escalated", "rail unavailable", "locked", "error")
REGEXES = [
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE_NUMBER", re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]?\d{3,4}\b")),
]
_lock = threading.Lock()
_presidio = {}


def entities() -> list[str]:
    """The union of the input and output entity lists in rails/config.yml."""
    import yaml
    sdd = yaml.safe_load((HERE / "rails" / "config.yml").read_text())["rails"]["config"]["sensitive_data_detection"]
    return sorted(set(sdd["input"]["entities"]) | set(sdd["output"]["entities"]))


def _engines():
    if "engines" not in _presidio:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            nlp = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]}).create_engine()
            _presidio["engines"] = (AnalyzerEngine(nlp_engine=nlp), AnonymizerEngine(), entities())
        except Exception as e:                    # no Presidio or no spaCy model: regex only, and the record says so
            _presidio["engines"] = None
            _presidio["why"] = f"{type(e).__name__}: {e}"[:200]
    return _presidio["engines"]


def masker_name() -> str:
    return "presidio+regex" if _engines() else "regex only (" + _presidio.get("why", "") + ")"


def mask(text) -> str:
    """Replace PII with <ENTITY> placeholders (Presidio first, then the regexes)."""
    if not isinstance(text, str) or not text:
        return text
    eng = _engines()
    if eng:
        analyzer, anonymizer, ents = eng
        from presidio_anonymizer.entities import OperatorConfig
        found = analyzer.analyze(text=text, language="en", entities=ents, score_threshold=0.4)
        if found:
            text = anonymizer.anonymize(text=text, analyzer_results=found,
                                        operators={e: OperatorConfig("replace") for e in ents}).text
    for name, rx in REGEXES:
        text = rx.sub(f"<{name}>", text)
    return text


def _mask_deep(value):
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, list):
        return [_mask_deep(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_deep(v) for k, v in value.items()}
    return value


def config_version() -> str:
    h = hashlib.sha256()
    for name in ("config.yml", "prompts.yml", "layers.py"):
        h.update((HERE / "rails" / name).read_bytes())
    return f"{DESK_VERSION}+{h.hexdigest()[:12]}"


def write(record: dict) -> dict:
    """Mask the free-text fields and the tool calls, then append one line. Returns what was written."""
    rec = {k: record.get(k) for k in FIELDS}
    now = time.time()
    rec["ts"] = rec.get("ts") or round(now, 3)
    rec["time"] = rec.get("time") or dt.datetime.fromtimestamp(rec["ts"], dt.timezone.utc).isoformat(timespec="seconds")
    for k in TEXT_FIELDS:
        rec[k] = mask(rec.get(k))
    rec["tool_calls"] = _mask_deep(rec.get("tool_calls") or [])
    rec["escalation"] = _mask_deep(rec.get("escalation"))
    rec["masker"] = masker_name()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with _lock, AUDIT.open("a") as f:
        f.write(line)
    return rec


def records(path: pathlib.Path = AUDIT) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def query(request_id: str | None = None, session: str | None = None, action: str | None = None,
          last: int | None = None) -> list[dict]:
    out = [r for r in records() if (not request_id or r.get("request_id") == request_id)
           and (not session or r.get("session_id") == session) and (not action or r.get("action") == action)]
    return out[-last:] if last else out


def purge(older_than_days: float, dry_run: bool = False) -> dict:
    """Retention: remove the records older than N days (the only rewrite of the file)."""
    cutoff = time.time() - older_than_days * 86400
    rows = records()
    keep = [r for r in rows if (r.get("ts") or 0) >= cutoff]
    out = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "older_than_days": older_than_days,
           "before": len(rows), "removed": len(rows) - len(keep), "kept": len(keep), "dry_run": dry_run}
    if not dry_run and len(keep) != len(rows):
        tmp = AUDIT.with_suffix(".tmp")
        with _lock:
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in keep))
            os.replace(tmp, AUDIT)
    if not dry_run:
        with PURGES.open("a") as f:
            f.write(json.dumps(out) + "\n")
    return out


def find_raw(needles: list[str]) -> list[str]:
    """The needles (raw PII strings) that occur anywhere in the audit log. Used by check.py."""
    text = AUDIT.read_text() if AUDIT.exists() else ""
    return [n for n in needles if n in text]


def show(rec: dict, say=print) -> None:
    """One record, field by field (the 9.1 lab scene)."""
    for k in FIELDS + ("masker",):
        v = rec.get(k)
        if k in ("input_rails", "output_rails") and isinstance(v, dict):
            say(f"{k:>15}: {v.get('status')}" + (f" by {v.get('rail')}" if v.get("rail") else "")
                + f", {v.get('duration_s')} s")
            for r in v.get("rails", []):
                say(f"{'':>17}- {r['name']}: {' > '.join(r['decisions'])}" + ("  [STOP]" if r["stop"] else "")
                    + ("  [ERROR]" if r.get("error") else ""))
            for key in ("categories", "errors", "unavailable"):
                if v.get(key):
                    say(f"{'':>17}{key}: {v[key]}")
            continue
        if k == "tool_calls" and v:
            say(f"{k:>15}:")
            for t in v:
                say(f"{'':>17}- {t.get('tool')} {json.dumps(t.get('params'), ensure_ascii=False)} -> "
                    f"{str(t.get('result'))[:110]}" + (f"  error: {t['error']}" if t.get("error") else ""))
            continue
        text = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        say(f"{k:>15}: {text[:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query", help="find records")
    q.add_argument("--request-id")
    q.add_argument("--session")
    q.add_argument("--action", choices=ACTIONS)
    q.add_argument("--last", type=int)
    q.add_argument("--json", action="store_true", help="print raw JSON lines")
    p = sub.add_parser("purge", help="retention: drop records older than N days")
    p.add_argument("--older-than", type=float, required=True, metavar="DAYS")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("stats", help="counts by action and layer")
    a = ap.parse_args()
    if a.cmd == "query":
        rows = query(a.request_id, a.session, a.action, a.last)
        if not rows:
            sys.exit(f"[audit] no records in {AUDIT}")
        for r in rows:
            if a.json:
                print(json.dumps(r, ensure_ascii=False))
            else:
                print(f"--- {r.get('request_id')} ({r.get('time')})")
                show(r)
        print(f"[audit] {len(rows)} record{'s' if len(rows) != 1 else ''}")
    elif a.cmd == "purge":
        out = purge(a.older_than, a.dry_run)
        print(f"[audit] {'would remove' if a.dry_run else 'removed'} {out['removed']} of {out['before']} records "
              f"older than {a.older_than:g} days; {out['kept']} kept")
    else:
        from collections import Counter
        rows = records()
        print(f"[audit] {len(rows)} records in {AUDIT}")
        print("by action:", dict(Counter(r.get("action") for r in rows)))
        print("by layer: ", dict(Counter(r.get("layer") for r in rows)))


if __name__ == "__main__":
    main()
