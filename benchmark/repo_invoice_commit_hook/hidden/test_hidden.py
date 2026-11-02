from decimal import Decimal

from invoicing import format_inr, line_total, total


def test_total_simple():
    assert total([("100.00", 1)]) == Decimal("118.00")


def test_total_rounds_half_up():
    assert total([("0.25", 1)]) == Decimal("0.30")


def test_total_other_tax_rate():
    assert total([("1.15", 1)], "0.10") == Decimal("1.27")


def test_total_many_lines():
    assert total([("0.10", 1)] * 3 + [("1234.565", 2)]) == Decimal("2913.93")


def test_total_zero_tax():
    assert total([("10.005", 1)], "0") == Decimal("10.01")


def test_total_large_invoice():
    assert total([("0.01", 1)] * 1000) == Decimal("11.80")


def test_total_is_decimal():
    assert isinstance(total([("1.00", 1)]), Decimal)


def test_line_total():
    assert line_total("19.99", 3) == Decimal("59.97")


def test_format_thousands():
    assert format_inr(Decimal("1234.5")) == "₹1,234.50"


def test_format_lakhs():
    assert format_inr(Decimal("1234567.5")) == "₹12,34,567.50"


def test_format_crores():
    assert format_inr(Decimal("123456789")) == "₹12,34,56,789.00"
