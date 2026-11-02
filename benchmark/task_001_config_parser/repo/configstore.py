"""A tiny config store."""

from __future__ import annotations

from typing import Any


def get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Return the value for `key` in `config`, or `default` if it is missing."""
    return config.get(key, default)
