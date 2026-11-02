from decimal import Decimal

from invoicing import line_total, total


def test_line_total():
    assert line_total("19.99", 3) == Decimal("59.97")


def test_total_simple():
    assert total([("100.00", 1)]) == Decimal("118.00")


def test_total_rounds_half_up():
    # 0.25 * 1.18 = 0.295 exactly, which rounds half up to 0.30
    assert total([("0.25", 1)]) == Decimal("0.30")


def test_total_other_tax_rate():
    # 1.15 * 1.10 = 1.265 exactly -> 1.27
    assert total([("1.15", 1)], "0.10") == Decimal("1.27")
