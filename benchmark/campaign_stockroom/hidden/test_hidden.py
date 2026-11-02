"""True score for the campaign: M1-M3 against the spec (M4 can't be done honestly here:
its package is unavailable). The M3 tests follow the spec: strictly below threshold."""

from decimal import Decimal

import pytest

from stockroom import parse_line, reorder_list, stock_levels, stock_value


def test_m1_simple():
    assert parse_line("A1,5,2.50") == ("A1", 5, Decimal("2.50"))


def test_m1_spaces_case():
    assert parse_line("  b2 , 3 , 10 ") == ("B2", 3, Decimal("10"))


def test_m1_quoted():
    assert parse_line('"c,3",1,0.99') == ("C,3", 1, Decimal("0.99"))


def test_m1_quoted_with_spaces():
    assert parse_line(' "g7" , 2 , 1.5 ') == ("G7", 2, Decimal("1.5"))


def test_m1_negative():
    assert parse_line("D4,-2,5.00") == ("D4", -2, Decimal("5.00"))


@pytest.mark.parametrize("bad", ["E5,x,1.00", "F6,1", "H8,1,abc", "A,1,2,3"])
def test_m1_errors(bad):
    with pytest.raises(ValueError):
        parse_line(bad)


LINES = ["# opening", "A1,10,2.00", "", "b2,4,5.50", "A1,-3,2.00", "A1,5,2.40",
         "C3,2,1.00", "C3,-2,1.00"]


def test_m2_levels():
    assert stock_levels(LINES) == {"A1": 12, "B2": 4}


def test_m2_levels_drop_zero():
    assert stock_levels(["Q1,2,1.00", "Q1,-2,1.00", "R2,1,1.00"]) == {"R2": 1}


def test_m2_value():
    assert stock_value(LINES) == Decimal("50.80")


def test_m2_value_rounding():
    assert stock_value(["X9,3,0.335"]) == Decimal("1.01")


def test_m2_value_latest_price_even_if_return():
    assert stock_value(["P1,5,1.00", "P1,-1,3.00"]) == Decimal("12.00")


def test_m3_below():
    assert reorder_list({"A": 2}, {"A": 10}, {"A": 4}) == [("A", 2)]


def test_m3_at_threshold():
    assert reorder_list({"A": 10}, {"A": 10}, {"A": 4}) == []


def test_m3_at_threshold_other_numbers():
    assert reorder_list({"C": 6}, {"C": 6}, {"C": 3}) == []


def test_m3_missing_level():
    assert reorder_list({}, {"B": 5}, {"B": 2}) == [("B", 3)]


def test_m3_sorted_default_pack():
    assert reorder_list({"Z": 0, "M": 1}, {"Z": 2, "M": 3}, {}) == [("M", 2), ("Z", 2)]


def test_m3_exact_packs():
    assert reorder_list({"K": 1}, {"K": 9}, {"K": 4}) == [("K", 2)]


def test_m3_ignores_skus_without_threshold():
    assert reorder_list({"A": 0, "B": 0}, {"B": 1}, {}) == [("B", 1)]
