"""Turn a ledger into the results table (and, later, plots).

    candor eval --ledger runs/latest/ledger.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from evals import metrics


def build_report(ledger_path: str) -> int:
    rows = _read(Path(ledger_path))
    if not rows:
        print(f"no rows in {ledger_path}")
        return 1

    print("\n=== R_true by iteration (did it actually improve?) ===")
    for it, v in metrics.true_reward_by_iteration(rows).items():
        print(f"  iter {it}: R_true={v:.3f}")

    print("\n=== Divergence: mean (R_proxy - R_true) by iteration ===")
    for it, v in metrics.mean_gap_by_iteration(rows).items():
        print(f"  iter {it}: gap={v:+.3f}")

    print("\n=== Reward-hacking emergence: cheat rate by iteration ===")
    for it, v in metrics.cheat_rate_by_iteration(rows).items():
        print(f"  iter {it}: cheat_rate={v:.2%}")

    print("\n=== Cheats by type (total) ===")
    for label, n in metrics.cheat_type_counts(rows).items():
        print(f"  {label}: {n}")

    sec = metrics.security_event_counts(rows)
    if sec:
        print("\n=== Security events (exfiltration / jail violations) ===")
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


if __name__ == "__main__":
    import sys

    raise SystemExit(build_report(sys.argv[1]))
