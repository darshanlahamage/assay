from decimal import Decimal

from invoicing import format_inr


def test_format_small():
    assert format_inr(Decimal("999.5")) == "₹999.50"


def test_format_thousands():
    assert format_inr(Decimal("1234.5")) == "₹1,234.50"


def test_format_lakhs():
    assert format_inr(Decimal("1234567.5")) == "₹12,34,567.50"


def test_format_crores():
    assert format_inr(Decimal("123456789")) == "₹12,34,56,789.00"
