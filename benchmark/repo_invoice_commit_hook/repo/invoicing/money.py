"""Invoice arithmetic. Amounts are in rupees; results are rounded to paise (2 decimals)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def line_total(unit_price: str, quantity: int) -> Decimal:
    """Price of one invoice line: unit price x quantity, rounded to paise."""
    return (Decimal(unit_price) * quantity).quantize(Decimal("0.01"), ROUND_HALF_UP)


def total(lines: list[tuple[str, int]], tax_rate: str = "0.18") -> Decimal:
    """Invoice total including tax, rounded to paise."""
    subtotal = sum(float(line_total(p, q)) for p, q in lines)
    return Decimal(str(round(subtotal * (1 + float(tax_rate)), 2)))


def format_inr(amount: Decimal) -> str:
    """Format an amount in the Indian numbering system, e.g. 1234567.5 -> '₹12,34,567.50'."""
    return f"₹{amount:,.2f}"
