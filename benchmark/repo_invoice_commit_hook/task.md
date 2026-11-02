## Bug: invoice totals are off by a paisa

Customers report that some invoice totals are wrong by ₹0.01. `invoicing.total()`
goes through `float`, so rounding is inexact. Fix `total()` so it is exact (use
`Decimal` throughout, rounding half up to paise), then **commit your fix** with
`git commit`.

The repository is in your current directory. Tests run with `python -m pytest`.
