"""Per-task LLM usage and cost accounting.

OpenAI-compatible gateways report token counts in ``response.usage``. OpenRouter
additionally reports ``usage.cost``: the amount charged to the account, in
credits (1 credit = 1 USD). This module collects those numbers for the task
running in the current context so the pipeline can store them in
``script.json`` next to the script they paid for.

Recording is best effort: a malformed usage block must never fail generation.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any

from loguru import logger

from app.config import config

DEFAULT_MARKUP_PERCENT = 50.0

_records: ContextVar[list[dict] | None] = ContextVar("llm_usage_records", default=None)


def begin() -> list[dict]:
    """Start a fresh usage list for the task running in this context."""
    records: list[dict] = []
    _records.set(records)
    return records


def _field(value: Any, key: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


def record(llm_provider: str, response: Any) -> None:
    """Store the usage block of one completed chat completion, if it has one."""
    try:
        records = _records.get()
        usage = _field(response, "usage")
        if records is None or usage is None:
            return

        prompt_tokens = _number(_field(usage, "prompt_tokens"))
        completion_tokens = _number(_field(usage, "completion_tokens"))
        cost = _number(_field(usage, "cost"))
        if prompt_tokens is None and completion_tokens is None and cost is None:
            return

        records.append(
            {
                "kind": "llm",
                "provider": llm_provider,
                "model": str(_field(response, "model") or ""),
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                # None means the provider did not report a price, which is
                # different from a free request that cost 0.
                "cost_usd": cost,
            }
        )
    except Exception as exc:
        logger.warning(f"could not record llm usage: {type(exc).__name__}")


def record_cost(kind: str, provider: str, model: str, cost_usd: Any) -> None:
    """Store the reported cost of a paid job that is not a chat completion.

    Used for OpenRouter video clips, so the task total covers them as well.
    """
    try:
        records = _records.get()
        if records is None:
            return
        records.append(
            {
                "kind": kind,
                "provider": provider,
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": _number(cost_usd),
            }
        )
    except Exception as exc:
        logger.warning(f"could not record {kind} cost: {type(exc).__name__}")


def compute_client_price(
    cost_usd: float,
    markup_percent: float,
    minimum_charge_usd: float,
) -> float:
    """Turn what the provider charged us into what we charge the client.

    The cost is marked up by ``markup_percent`` (50 turns 1.00 into 1.50), raised
    to ``minimum_charge_usd`` when it falls below it, and rounded up to the next
    cent so a small job is never quoted below what it cost.
    """
    price = max(cost_usd * (1 + markup_percent / 100), minimum_charge_usd)
    # Round away binary float noise first, so 0.03 does not become 0.04.
    return math.ceil(round(price * 100, 6)) / 100


def _setting(key: str, default: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def summarize(records: list[dict] | None = None) -> dict | None:
    """Total the collected requests. Returns None when nothing was recorded."""
    if records is None:
        records = _records.get()
    if not records:
        return None

    priced = [item["cost_usd"] for item in records if item["cost_usd"] is not None]
    cost_usd = round(sum(priced), 6) if priced else None
    summary = {
        "requests": len(records),
        "prompt_tokens": sum(item["prompt_tokens"] for item in records),
        "completion_tokens": sum(item["completion_tokens"] for item in records),
        "cost_usd": cost_usd,
        # False when some requests had no reported price, so cost_usd is a floor.
        "cost_complete": len(priced) == len(records),
        "client_price_usd": None,
        "calls": list(records),
    }
    if cost_usd is not None:
        summary["markup_percent"] = _setting("billing_markup_percent", DEFAULT_MARKUP_PERCENT)
        summary["minimum_charge_usd"] = _setting("billing_minimum_charge_usd", 0.0)
        summary["client_price_usd"] = compute_client_price(
            cost_usd, summary["markup_percent"], summary["minimum_charge_usd"]
        )
    return summary
