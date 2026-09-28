"""Ticket tools for the support-desk agent, registered as NeMo Agent Toolkit functions.

They call the small ticket API in m02/ticket_api.py. When the API fails, the tool
raises an error instead of hiding it, so the toolkit's middleware (timeout, circuit
breaker) can see the failure and react to it.
"""
import json

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class TicketApiError(RuntimeError):
    """The ticket API answered with an error (for example HTTP 503)."""


def _check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise TicketApiError(f"ticket API returned HTTP {r.status_code}: {r.text}")
    return r.json()


class CreateTicketConfig(FunctionBaseConfig, name="create_ticket"):
    api_url: str = "http://localhost:8765"


@register_function(config_type=CreateTicketConfig)
async def create_ticket(config: CreateTicketConfig, builder: Builder):

    async def _create(order_id: str, issue: str) -> str:
        async with httpx.AsyncClient(base_url=config.api_url, timeout=10) as c:
            ticket = _check(await c.post("/tickets", json={"order_id": order_id, "issue": issue}))
        return json.dumps(ticket)

    yield FunctionInfo.from_fn(
        _create,
        description=("Open a support ticket when a customer reports a problem with an order "
                     "(damaged, missing or wrong item). Needs the order ID and a one-line issue."))


class GetTicketConfig(FunctionBaseConfig, name="get_ticket"):
    api_url: str = "http://localhost:8765"


@register_function(config_type=GetTicketConfig)
async def get_ticket(config: GetTicketConfig, builder: Builder):

    async def _get(ticket_id: str) -> str:
        async with httpx.AsyncClient(base_url=config.api_url, timeout=10) as c:
            r = await c.get(f"/tickets/{ticket_id.strip().upper()}")
        if r.status_code == 404:
            return f"No ticket found with ID {ticket_id}"
        return json.dumps(_check(r))

    yield FunctionInfo.from_fn(
        _get, description="Look up an existing support ticket by its ID (for example T-1001) and return its status.")
