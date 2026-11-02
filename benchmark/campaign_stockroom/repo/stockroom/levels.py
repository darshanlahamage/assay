"""Milestone 2: stock levels and stock value from a list of movement lines."""

from __future__ import annotations

from decimal import Decimal

from stockroom.parse import parse_line


def stock_levels(lines: list[str]) -> dict[str, int]:
    """Current quantity per SKU. See MILESTONES.md."""
    levels: dict[str, int] = {}
    for line in lines:
        sku, qty, _ = parse_line(line)
        levels[sku] = levels.get(sku, 0) + qty
    return levels


def stock_value(lines: list[str]) -> Decimal:
    """Value of the current stock. See MILESTONES.md."""
    raise NotImplementedError
