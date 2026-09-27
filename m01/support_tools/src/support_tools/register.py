"""Two tools for the support-desk agent, registered as NeMo Agent Toolkit functions.

The toolkit finds them through the `nat.plugins` entry point in pyproject.toml, so
after `pip install -e m01/support_tools` a config file can use
`_type: lookup_order` and `_type: returns_policy`.
"""
import json
import pathlib

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

DATA = pathlib.Path(__file__).resolve().parents[3] / "data"


class LookupOrderConfig(FunctionBaseConfig, name="lookup_order"):
    orders_file: str = str(DATA / "orders.json")


@register_function(config_type=LookupOrderConfig)
async def lookup_order(config: LookupOrderConfig, builder: Builder):
    orders = json.loads(pathlib.Path(config.orders_file).read_text())

    async def _lookup(order_id: str) -> str:
        order = orders.get(order_id.strip().upper())
        return json.dumps(order) if order else f"No order found with ID {order_id}"

    yield FunctionInfo.from_fn(
        _lookup, description="Look up a customer order by its ID (for example A1001) and return its status.")


class ReturnsPolicyConfig(FunctionBaseConfig, name="returns_policy"):
    policy_file: str = str(DATA / "returns_policy.md")


@register_function(config_type=ReturnsPolicyConfig)
async def returns_policy(config: ReturnsPolicyConfig, builder: Builder):
    policy = pathlib.Path(config.policy_file).read_text()

    async def _policy(question: str) -> str:
        return policy

    yield FunctionInfo.from_fn(
        _policy, description="Return the company's returns and refunds policy.")
