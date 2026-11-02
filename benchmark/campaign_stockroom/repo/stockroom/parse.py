"""Milestone 1: parse one line of a stock movement file."""

from __future__ import annotations

from decimal import Decimal


def parse_line(line: str) -> tuple[str, int, Decimal]:
    """Parse "sku,qty,unit_price" into (SKU, qty, price). See MILESTONES.md."""
    sku, qty, price = line.split(",")
    return sku, int(qty), Decimal(price)
