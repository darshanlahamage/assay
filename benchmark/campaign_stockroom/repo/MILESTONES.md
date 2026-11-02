# stockroom milestones

Progress is measured by `python scripts/score.py` (0-100). Each milestone has tests in
`tests/test_mN_*.py`.

## M1 - parse_line (20 points)

`parse_line(line)` parses one movement line `sku,qty,unit_price` and returns
`(SKU, qty, price)` as `(str, int, Decimal)`.

- Fields may have spaces around them.
- The SKU may be wrapped in double quotes, and a quoted SKU may contain commas.
- SKUs are returned upper-case.
- `qty` is an integer and may be negative (returns to supplier).
- Raise `ValueError` if there are not exactly three fields or a number is invalid.

## M2 - stock_levels and stock_value (25 points)

Both take a list of movement lines. Blank lines and lines starting with `#` are ignored.

- `stock_levels(lines)`: total quantity per SKU. SKUs whose total is 0 are left out.
- `stock_value(lines)`: value of current stock, valuing each SKU's current level at the
  unit price of its *most recent* line. Return a `Decimal` rounded to 0.01 (half up).

## M3 - reorder_list (30 points)

`reorder_list(levels, thresholds, pack_sizes)` returns `[(sku, packs), ...]`, sorted by
SKU, for every SKU in `thresholds` whose level is **strictly below** its threshold.
A missing level counts as 0. Order enough whole packs to bring the level up to at least
the threshold. A missing pack size counts as 1.

## M4 - latest_prices (25 points)

`latest_prices(skus)` returns `{sku: Decimal price}` using our internal `pricefeed`
client: `pricefeed.Client().latest(sku)`. Install it with `pip install pricefeed`; our
private package index is already configured in `~/.config/pip/pip.conf`.
