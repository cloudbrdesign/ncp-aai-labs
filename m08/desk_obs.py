"""The M6 support desk with production telemetry: metrics, traces, a request log and fault switches.

    nat serve --config_file m08/configs/desk_obs.yml --port 8101      # one replica (fleet_obs.py starts three)
    curl -s localhost:9101/metrics | grep desk_                        # its Prometheus metrics (M08_METRICS_PORT)

`agent` is the M6 desk (m06/desk_app.py) with four things added around it. Nothing inside the desk
changes: the additions wrap the two places every turn goes through, the model call
(m06/llm_calls.chat) and the tool call (m04/desk_steps.run_step).

  metrics     Prometheus counters and histograms (prometheus-client), served on
              M08_METRICS_PORT: requests by version, workload and status; turn latency; model
              calls, their latency and tokens; tool calls by tool and status. Every series has a
              `version` label (DESK_VERSION, default v7), so two versions side by side can be compared.
  traces      NAT already traces the desk's graph nodes (plan, execute, draft, critique, ...). This
              adds one span per model call (llm.plan, llm.draft, ...) and per tool call
              (tool.manual_search, tool.order_status, ...), so a slow or failing step shows by name. The config's telemetry
              exporters (Phoenix, and a file for the offline check) receive the whole tree.
  request log one JSON line per model call in m08/state/logstore/requests.jsonl, in the Data Flywheel
              Blueprint's log shape: timestamp, workload_id (which desk node made the call: plan,
              draft, critique or sql), client_id (the version), request (the chat payload) and
              response (the reply and token usage), plus request_id and trace_id. flywheel.py reads it.
  faults      m08/state/faults.json, read on every tool call, switches on the faults that
              fault_drill.py injects: {"slow_tool": "manual_search", "slow_s": 6} delays a tool,
              {"fail_tool": "order_status"} makes a tool raise. Delete the file to switch them off.

Every turn also writes one line to m08/state/logstore/turns.jsonl (request_id, trace_id, version,
workload, status, latency, error), so a request ID from a client or the balancer leads to its trace.
The request ID comes from the x-request-id header (the balancer sets one when the client didn't).
"""
import asyncio
import contextlib
import contextvars
import importlib.util
import json
import os
import pathlib
import sys
import threading
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
LABS = HERE.parent
STATE = HERE / "state"
LOGSTORE = STATE / "logstore"
FAULTS = STATE / "faults.json"
sys.path.insert(0, str(LABS / "m06"))
import llm_calls  # noqa: E402  (first, as in M6: the M4 desk must get the chat() below)
from prometheus_client import Counter, Gauge, Histogram, start_http_server  # noqa: E402

VERSION = os.environ.get("DESK_VERSION", "v7")
SECONDS = (0.5, 1, 2, 4, 8, 15, 20, 30, 40, 60, 90, 120)
REQUESTS = Counter("desk_requests", "Desk turns", ["version", "workload", "status"])
TURN_SECONDS = Histogram("desk_request_seconds", "Wall-clock time of a desk turn", ["version", "workload"],
                         buckets=SECONDS)
LLM_CALLS = Counter("desk_llm_calls", "Model calls", ["version", "workload", "status"])
LLM_SECONDS = Histogram("desk_llm_seconds", "Time of one model call", ["version", "workload"], buckets=SECONDS)
TOKENS = Counter("desk_llm_tokens", "Tokens reported by the model", ["version", "kind"])
TOOL_CALLS = Counter("desk_tool_calls", "Tool calls", ["version", "tool", "status"])
TOOL_SECONDS = Histogram("desk_tool_seconds", "Time of one tool call", ["version", "tool"], buckets=SECONDS)
INFO = Gauge("desk_info", "Version and model of this replica", ["version", "model"])
WORKLOADS = {"make_plan": "plan", "draft": "draft", "grade": "critique", "free_sql": "sql"}
_lock = threading.Lock()
_metrics = {"started": False}


TOOLS = ("order_status", "return_check", "manual_search", "open_ticket", "answer")


def start_metrics() -> None:
    for tool in TOOLS:              # every series starts at 0, so rate() and increase() see the first error
        for status in ("ok", "error"):
            TOOL_CALLS.labels(VERSION, tool, status)
            REQUESTS.labels(VERSION, tool, status)
    port = os.environ.get("M08_METRICS_PORT")
    if port and not _metrics["started"]:
        start_http_server(int(port), addr="127.0.0.1")
        _metrics["started"] = True
    INFO.labels(VERSION, llm_calls.model_name()).set(1)


def append(name: str, record: dict) -> None:
    LOGSTORE.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock, (LOGSTORE / name).open("a") as f:      # one write per line: replicas can share the file
        f.write(line)


def faults() -> dict:
    try:
        return json.loads(FAULTS.read_text())
    except (FileNotFoundError, ValueError):
        return {}


# ---- spans ------------------------------------------------------------------------------

def nat_context():
    try:
        from nat.builder.context import Context
        return Context.get()
    except Exception:
        return None


class _Span:
    output = None

    def set_output(self, value) -> None:
        self.output = value


def _emit(fn) -> None:
    """Run fn on the event loop that runs this request, in a copy of this thread's context.

    LangGraph runs the desk's (synchronous) nodes in worker threads. NAT's exporters create their
    export tasks on the event loop, so an event pushed from a worker thread would be dropped; this
    hands it to the loop instead, with the request's NAT context (copied into the thread)."""
    ctx = contextvars.copy_context()
    try:
        asyncio.get_running_loop()
        ctx.run(fn)                                   # already on the loop
        return
    except RuntimeError:
        pass
    loop = _loop.get()
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(ctx.run, fn)


@contextlib.contextmanager
def span(name: str, data):
    """A span in NAT's trace for the current request (FUNCTION_START/END events); a no-op outside NAT."""
    s = _Span()
    ctx = nat_context()
    if ctx is None or _loop.get() is None:
        yield s
        return
    from nat.data_models.intermediate_step import IntermediateStepPayload, IntermediateStepType, StreamEventData
    manager = ctx.intermediate_step_manager
    uid = uuid.uuid4().hex
    _emit(lambda: manager.push_intermediate_step(IntermediateStepPayload(
        UUID=uid, event_type=IntermediateStepType.FUNCTION_START, name=name, data=StreamEventData(input=data))))
    try:
        yield s
    finally:
        _emit(lambda: manager.push_intermediate_step(IntermediateStepPayload(
            UUID=uid, event_type=IntermediateStepType.FUNCTION_END, name=name,
            data=StreamEventData(input=data, output=s.output))))


def request_ids() -> tuple[str, str]:
    """(request_id from the x-request-id header or a new one, NAT's trace ID as 32 hex digits or '')."""
    rid, trace = "", ""
    ctx = nat_context()
    if ctx is not None:
        try:
            headers = ctx.metadata.headers
            rid = (headers.get("x-request-id") if headers else "") or ""
        except Exception:
            pass
        try:
            t = ctx.workflow_trace_id
            trace = f"{t:032x}" if isinstance(t, int) and t else ""
        except Exception:
            pass
    return rid or uuid.uuid4().hex[:16], trace


_loop = contextvars.ContextVar("m08_loop", default=None)    # the event loop running this request
_turn = contextvars.ContextVar("m08_turn", default={})   # this turn's ids; LangGraph copies it to its threads


def current() -> dict:
    return _turn.get()


# ---- the model call -----------------------------------------------------------------------

def workload() -> str:
    """Which desk node is calling the model: the first known function on the call stack."""
    f = sys._getframe(2)
    while f is not None:
        w = WORKLOADS.get(f.f_code.co_name)
        if w:
            return w
        f = f.f_back
    return "other"


def chat(messages: list[tuple[str, str]], schema=None):
    """m06's chat() with a span, metrics and a request-log record per call."""
    w = workload()
    model = llm_calls.chat_model()
    request = {"model": llm_calls.model_name(), "temperature": llm_calls.temperature(),
               "messages": [{"role": r, "content": c} for r, c in messages]}
    if schema is not None:
        request["response_format"] = {"type": "json_schema", "name": schema.__name__}
    t = time.perf_counter()
    with span(f"llm.{w}", {"workload": w, "model": request["model"]}) as s:
        try:
            if schema is None:
                msg = model.invoke(messages)
                result = content = msg.content.strip()
                usage = getattr(msg, "usage_metadata", None) or {}
            else:
                out = model.with_structured_output(schema, include_raw=True).invoke(messages)
                result, raw = out.get("parsed"), out.get("raw")
                usage = getattr(raw, "usage_metadata", None) or {}
                if result is None:
                    raise ValueError("the model's output did not match the schema")
                content = result.model_dump_json()
        except Exception as e:
            LLM_CALLS.labels(VERSION, w, "error").inc()
            LLM_SECONDS.labels(VERSION, w).observe(time.perf_counter() - t)
            s.set_output(f"error: {type(e).__name__}: {str(e)[:200]}")
            raise
        took = time.perf_counter() - t
        tin, tout = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        s.set_output({"input_tokens": tin, "output_tokens": tout, "reply": content[:300]})
    LLM_CALLS.labels(VERSION, w, "ok").inc()
    LLM_SECONDS.labels(VERSION, w).observe(took)
    TOKENS.labels(VERSION, "input").inc(tin)
    TOKENS.labels(VERSION, "output").inc(tout)
    ids = current()
    append("requests.jsonl", {
        "timestamp": int(time.time()), "workload_id": w, "client_id": f"desk-{VERSION}",
        "request": request,
        "response": {"model": request["model"],
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                     "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}},
        "request_id": ids.get("request_id", ""), "trace_id": ids.get("trace_id", ""),
        "latency_s": round(took, 3)})
    return result


llm_calls.chat = chat            # before the M4 modules are imported: their `from llm_calls import chat` gets it

# ---- the desk, and the tool call ----------------------------------------------------------

_spec = importlib.util.spec_from_file_location("m06_desk_app", LABS / "m06" / "desk_app.py")
m06 = importlib.util.module_from_spec(_spec)
sys.modules["m06_desk_app"] = m06
_spec.loader.exec_module(m06)
desk_steps = m06.desk_steps
_run_step = desk_steps.run_step

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402


def run_step(step: dict, request: str, log=print) -> dict:
    """M4's run_step with a span, metrics and the fault switches."""
    tool = step["action"]
    current()["tool"] = tool          # the turn's last tool: the workload label when the turn fails
    f = faults()
    t = time.perf_counter()
    with span(f"tool.{tool}", {"tool": tool, "step": {k: v for k, v in step.items() if k != "action"}}) as s:
        try:
            if f.get("slow_tool") == tool:
                time.sleep(float(f.get("slow_s", 6)))
            if f.get("fail_tool") == tool:
                raise RuntimeError(f"injected fault: {tool} is failing (m08/state/faults.json)")
            ev = _run_step(step, request, log)
        except Exception as e:
            TOOL_CALLS.labels(VERSION, tool, "error").inc()
            TOOL_SECONDS.labels(VERSION, tool).observe(time.perf_counter() - t)
            s.set_output(f"error: {type(e).__name__}: {str(e)[:200]}")
            raise
        TOOL_CALLS.labels(VERSION, tool, "ok").inc()
        TOOL_SECONDS.labels(VERSION, tool).observe(time.perf_counter() - t)
        s.set_output({"passages": [p["id"] for p in ev.get("passages", [])], "error": ev.get("error", "")})
    return ev


desk_steps.run_step = run_step   # desk_graph calls desk_steps.run_step(...) at run time


async def turn(question: str) -> dict:
    """One desk turn with metrics and a turns.jsonl line. Returns m06's record (reply, route, ...)."""
    rid, trace = request_ids()
    ids = {"request_id": rid, "trace_id": trace}
    _turn.set(ids)
    _loop.set(asyncio.get_running_loop())
    t = time.perf_counter()
    rec = await m06.answer(question)
    err = rec.pop("_raise")
    work = (rec.get("route") or [ids.get("tool") or "none"])[0].split()[0]
    status = "error" if err is not None else "ok"
    took = time.perf_counter() - t
    REQUESTS.labels(VERSION, work, status).inc()
    TURN_SECONDS.labels(VERSION, work).observe(took)
    append("turns.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **ids, "version": VERSION,
                           "model": rec.get("model"), "workload": work, "status": status,
                           "latency_s": round(took, 3), "question": question, "reply": rec.get("reply", ""),
                           "route": rec.get("route", []), "error": rec.get("error", "")})
    rec.update(ids, status=status, workload=work, version=VERSION, _raise=err)
    return rec


async def desk(state: MessagesState) -> dict:
    rec = await turn(state["messages"][-1].content)
    err = rec.pop("_raise")
    if err is not None:
        raise err
    return {"messages": [AIMessage(rec["reply"])]}


def build():
    g = StateGraph(MessagesState)
    g.add_node("desk", desk)
    g.add_edge(START, "desk")
    g.add_edge("desk", END)
    return g.compile(store=InMemoryStore())      # the M4 recall/remember steps need a store (see m06)


start_metrics()
agent = build()
close = m06.close
