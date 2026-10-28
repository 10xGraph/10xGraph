"""Fake customer, order and refund data for the support demo.

Replace these dicts with calls to your CRM or order system. Everything is in
memory, so refunds issued during a run reset when the process exits.
"""

from __future__ import annotations

from typing import Any


CUSTOMERS: dict[str, dict[str, Any]] = {
    "C-1001": {"name": "Maya Patel", "email": "maya@example.com", "plan": "Pro"},
    "C-1002": {"name": "Tomas Reyes", "email": "tomas@example.com", "plan": "Starter"},
}

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-5001": {
        "customer_id": "C-1001",
        "item": "Pro plan, annual",
        "amount": 240.00,
        "status": "paid",
        "charged_on": "2026-10-01",
    },
    "ORD-5002": {
        "customer_id": "C-1001",
        "item": "Extra seat, monthly",
        "amount": 12.00,
        "status": "paid",
        "charged_on": "2026-10-15",
    },
    "ORD-6001": {
        "customer_id": "C-1002",
        "item": "Starter plan, monthly",
        "amount": 19.00,
        "status": "paid",
        "charged_on": "2026-10-20",
    },
}

# Refunds issued during this process: order_id -> amount.
REFUNDS: dict[str, float] = {}

KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "title": "Login loops back to the sign-in page",
        "body": "Clear cookies for the app domain, then sign in again. If SSO is on, "
        "check that the identity provider session has not expired.",
    },
    {
        "title": "API returns 429 Too Many Requests",
        "body": "Starter allows 60 requests/minute, Pro allows 600. Back off with "
        "exponential delay and honour the Retry-After header.",
    },
    {
        "title": "Webhook deliveries are delayed",
        "body": "Deliveries retry for 24 hours. Check the endpoint returns 2xx within "
        "10 seconds and is not blocking on slow work.",
    },
]
