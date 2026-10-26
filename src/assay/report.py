"""Turn a ledger into the results table (and, later, plots).

    assay eval --ledger runs/<timestamp>/ledger.jsonl

Pure functions over the list-of-dict ledger so they are trivially testable and
runnable without any model loaded.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


# -- aggregate metrics -------------------------------------------------------

def true_reward_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Mean R_true per iteration — did the agent actually get better?"""
    return _mean_by_iter(rows, "r_true")


def mean_gap_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Mean (R_proxy - R_true) per iteration — the divergence curve."""
    return _mean_by_iter(rows, "gap")


def cheat_rate_by_iteration(rows: list[dict]) -> dict[int, float]:
    """Fraction of episodes with >=1 anchor-confirmed cheat — the emergence curve."""
    acc: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        acc[r["iteration"]].append(1 if r.get("cheats") else 0)
    return {it: sum(v) / len(v) for it, v in sorted(acc.items())}


def cheat_type_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c.update(r.get("cheats", []))
    return dict(c.most_common())


def security_event_counts(rows: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c.update(r.get("security_events", []))
    return dict(c.most_common())


def _mean_by_iter(rows: list[dict], field: str) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["iteration"]].append(r[field])
    return {it: sum(v) / len(v) for it, v in sorted(acc.items())}


# -- report ------------------------------------------------------------------

def build_report(ledger_path: str) -> int:
    rows = _read(Path(ledger_path))
    if not rows:
        print(f"no rows in {ledger_path}")
        return 1

    print("\n=== R_true by iteration (did it actually improve?) ===")
    for it, v in true_reward_by_iteration(rows).items():
        print(f"  iter {it}: R_true={v:.3f}")

    print("\n=== Divergence: mean (R_proxy - R_true) by iteration ===")
    for it, v in mean_gap_by_iteration(rows).items():
        print(f"  iter {it}: gap={v:+.3f}")

    print("\n=== Reward-hacking emergence: cheat rate by iteration ===")
    for it, v in cheat_rate_by_iteration(rows).items():
        print(f"  iter {it}: cheat_rate={v:.2%}")

    types = cheat_type_counts(rows)
    if types:
        print("\n=== Cheats by type (total) ===")
        for label, n in types.items():
            print(f"  {label}: {n}")

    sec = security_event_counts(rows)
    if sec:
        print("\n=== Security events ===")
        for label, n in sec.items():
            print(f"  {label}: {n}")

    # TODO (Phase 5): matplotlib plots — divergence, emergence, hallucination
    # vs cheating separation, probe accuracy.
    return 0


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
