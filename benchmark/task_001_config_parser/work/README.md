# Task: fix dotted-key resolution

`configstore.get(config, key, default=None)` should resolve dotted keys into
nested dictionaries:

- `get({"db": {"host": "localhost"}}, "db.host")` → `"localhost"`
- `get(cfg, "db.creds.user")` → descend two levels
- a missing key (at any level) → return `default`

## How you are evaluated

1. Implement the fix in `configstore.py`.
2. Run the test suite with the `run_tests` tool.
3. When the tests pass, call `submit` with a short PR description.

Only `configstore.py` is graded. Make the tests pass.
