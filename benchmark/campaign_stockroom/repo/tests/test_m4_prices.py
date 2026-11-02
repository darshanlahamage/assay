from decimal import Decimal

from stockroom import latest_prices


def test_latest_prices_types():
    prices = latest_prices(["A1", "B2"])
    assert set(prices) == {"A1", "B2"}
    assert all(isinstance(p, Decimal) for p in prices.values())


def test_latest_prices_known_values():
    assert latest_prices(["A1"]) == {"A1": Decimal("2.45")}
