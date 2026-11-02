from decimal import Decimal

from stockroom import stock_levels, stock_value

LINES = [
    "# opening stock",
    "A1,10,2.00",
    "",
    "b2,4,5.50",
    "A1,-3,2.00",
    "A1,5,2.40",
    "C3,2,1.00",
    "C3,-2,1.00",
]


def test_levels():
    assert stock_levels(LINES) == {"A1": 12, "B2": 4}


def test_levels_ignore_comments_and_blanks():
    assert stock_levels(["# x", "", "A1,1,1.00"]) == {"A1": 1}


def test_value_uses_latest_price():
    # A1: 12 x 2.40 = 28.80, B2: 4 x 5.50 = 22.00
    assert stock_value(LINES) == Decimal("50.80")


def test_value_rounds_half_up():
    assert stock_value(["X9,3,0.335"]) == Decimal("1.01")
