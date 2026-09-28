"""Watch the circuit breaker open, short-circuit and recover. No model needed.

    # terminal 1: a ticket API whose first 2 requests fail with HTTP 503
    python m02/ticket_api.py --fail-first 2
    # terminal 2:
    python m02/breaker_demo.py

It builds the create_ticket tool from m02/configs/support_desk.yml, with the same
timeout and circuit_breaker middleware the agent uses, and calls it five times.
"""
import asyncio
import logging
import pathlib
import time

from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_config

CONFIG = pathlib.Path(__file__).resolve().parent / "configs" / "support_desk.yml"
TICKET = {"order_id": "A1002", "issue": "USB-C dock arrived cracked"}


async def call(fn, n: int, note: str):
    t0 = time.monotonic()
    try:
        result = await fn.ainvoke(TICKET)
        outcome = f"OK   {result}"
    except Exception as e:  # the tool's own error, or CircuitBreakerOpenError
        outcome = f"FAIL {type(e).__name__}: {str(e).split('. Tool is')[0]}"
    print(f"call {n} ({note}, {time.monotonic() - t0:.2f}s) -> {outcome}", flush=True)


async def main():
    logging.getLogger("nat").setLevel(logging.CRITICAL)  # keep the output to our own lines
    cfg = load_config(CONFIG)
    cooldown = cfg.middleware["ticket_breaker"].cooldown_period
    async with WorkflowBuilder.from_config(cfg) as builder:
        fn = await builder.get_function("create_ticket")
        await call(fn, 1, "API is down")
        await call(fn, 2, "API is down, breaker trips")
        await call(fn, 3, "breaker OPEN: not even tried")
        print(f"... waiting {cooldown:.0f}s cooldown, then the breaker lets one probe through", flush=True)
        await asyncio.sleep(cooldown + 0.5)
        await call(fn, 4, "HALF_OPEN probe, API is back")
        await call(fn, 5, "breaker CLOSED again")


if __name__ == "__main__":
    asyncio.run(main())
