"""A tiny config store.

BUG: `get` does not resolve dotted keys into nested dictionaries — it only does
a flat lookup. Fix it so that "db.host" descends into {"db": {"host": ...}} and
so that missing keys return the provided default.
"""

from __future__ import annotations

from typing import Any


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    # FIXME: flat lookup only; does not handle dotted/nested keys.
    return config.get(key, default)
