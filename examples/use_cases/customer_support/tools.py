"""Support tools: customer lookup, orders, refunds with approval, and diagnostics.

`issue_refund` is the human-in-the-loop part. `interrupt()` stops the run and
saves it; the CLI shows the request to a person and resumes the same thread
with their answer.
"""

from __future__ import annotations

import json
from typing import Any

from config import REFUND_APPROVAL_THRESHOLD
from data import CUSTOMERS, KNOWLEDGE_BASE, ORDERS, REFUNDS

from tenxgraph.utils import interrupt


# The `config` parameter is filled in by 10xGraph with the run config. Reading
# the customer from `config["user_id"]` (set by the CLI, not by the model) means
# the model can never look up another customer's data.

MAX_HITS = 3
MIN_WORD = 3


def lookup_customer(config: dict[str, Any]) -> str:
    """Return the signed-in customer's profile and plan."""
    customer = CUSTOMERS.get(config.get("user_id", ""))
    return json.dumps(customer) if customer else json.dumps({"error": "unknown customer"})


def lookup_order(order_id: str, config: dict[str, Any]) -> str:
    """Look up one order by id, if it belongs to the signed-in customer."""
    order = ORDERS.get(order_id)
    if not order or order["customer_id"] != config.get("user_id"):
        return json.dumps({"error": f"order {order_id} not found"})
    return json.dumps({"order_id": order_id, **order, "refunded": REFUNDS.get(order_id, 0.0)})


def issue_refund(order_id: str, amount: float, reason: str, config: dict[str, Any]) -> str:
    """Refund part or all of an order. Large refunds need human approval."""
    order = ORDERS.get(order_id)
    if not order or order["customer_id"] != config.get("user_id"):
        return json.dumps({"error": f"order {order_id} not found"})
    if amount <= 0 or amount + REFUNDS.get(order_id, 0.0) > order["amount"]:
        return json.dumps({"error": "refund exceeds the order amount"})

    if amount > REFUND_APPROVAL_THRESHOLD:
        # On the first run this raises and pauses the graph. When the thread is
        # resumed, this whole function runs again and interrupt() returns the
        # reviewer's answer. Keep side effects (the refund) below this call.
        decision = interrupt(
            {"order_id": order_id, "amount": amount, "reason": reason},
            message=f"Approve a ${amount:.2f} refund on {order_id}?",
            reason="refund_approval",
            response_schema={
                "type": "object",
                "properties": {"approved": {"type": "boolean"}, "note": {"type": "string"}},
                "required": ["approved"],
            },
        )
        # None means the reviewer cancelled instead of answering.
        if not decision or not decision.get("approved"):
            note = (decision or {}).get("note", "no reason given")
            return json.dumps({"status": "declined", "note": note})

    REFUNDS[order_id] = REFUNDS.get(order_id, 0.0) + amount
    return json.dumps({"status": "refunded", "order_id": order_id, "amount": amount})


def search_knowledge_base(query: str) -> str:
    """Search help articles by keyword and return the best matches."""
    words = {w.lower() for w in query.split() if len(w) > MIN_WORD}
    hits = [
        article
        for article in KNOWLEDGE_BASE
        if words & set((article["title"] + " " + article["body"]).lower().split())
    ]
    return json.dumps(hits[:MAX_HITS] or [{"title": "No article found", "body": ""}])


def run_diagnostics(service: str, config: dict[str, Any]) -> str:
    """Run a health check for one of the customer's services (api, webhooks, login)."""
    # Fake result; a real version would query your monitoring system.
    status = {"api": "degraded: rate limited", "webhooks": "ok", "login": "ok"}
    return json.dumps(
        {
            "customer": config.get("user_id"),
            "service": service,
            "status": status.get(service, "unknown service"),
        }
    )
