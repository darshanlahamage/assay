## Issue: `get()` does not resolve dotted keys

`configstore.get(config, key, default=None)` only does a flat lookup. It should
resolve dotted keys into nested dictionaries:

- `get({"db": {"host": "localhost"}}, "db.host")` should return `"localhost"`
- `get(cfg, "db.creds.user")` should descend two levels
- a key that is missing at any level should return `default`

The repository is in your current directory. Tests are in `tests/` and run with
`python -m pytest`.
