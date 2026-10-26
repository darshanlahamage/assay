"""Aggregate metrics over ledger rows.

Pure functions on the list-of-dict ledger so they are trivially testable and
runnable without any model loaded.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def mean_gap_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Mean (R_proxy - R_true) per iteration — the divergence curve."""
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["iteration"]].append(r["gap"])
    return {it: sum(v) / len(v) for it, v in sorted(acc.items())}


def cheat_rate_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Fraction of episodes with at least one anchor-confirmed cheat — emergence."""
    acc: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        acc[r["iteration"]].append(1 if r.get("cheats") else 0)
    return {it: sum(v) / len(v) for it, v in sorted(acc.items())}


def cheat_type_counts(rows: list[dict]) -> dict[str, int]:
    """Total count of each cheat label across all episodes."""
    c: Counter[str] = Counter()
    for r in rows:
        c.update(r.get("cheats", []))
    return dict(c.most_common())


def security_event_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c.update(r.get("security_events", []))
    return dict(c.most_common())


def true_reward_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Mean R_true per iteration — did the agent actually get better?"""
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["iteration"]].append(r["r_true"])
    return {it: sum(v) / len(v) for it, v in sorted(acc.items())}
