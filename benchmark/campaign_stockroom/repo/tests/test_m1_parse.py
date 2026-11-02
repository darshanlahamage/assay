from decimal import Decimal

import pytest

from stockroom import parse_line


def test_simple():
    assert parse_line("A1,5,2.50") == ("A1", 5, Decimal("2.50"))


def test_spaces_and_case():
    assert parse_line("  b2 , 3 , 10 ") == ("B2", 3, Decimal("10"))


def test_quoted_sku_with_comma():
    assert parse_line('"c,3",1,0.99') == ("C,3", 1, Decimal("0.99"))


def test_negative_qty():
    assert parse_line("D4,-2,5.00") == ("D4", -2, Decimal("5.00"))


def test_bad_number():
    with pytest.raises(ValueError):
        parse_line("E5,x,1.00")


def test_wrong_field_count():
    with pytest.raises(ValueError):
        parse_line("F6,1")
