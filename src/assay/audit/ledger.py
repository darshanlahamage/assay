"""The experiment ledger — the 'trackable changes' record.

One append-only JSONL row per (RL iteration x task episode), capturing the
rewards, the gap, the cheats detected, and pointers to evidence. The ledger is
what the eval step turns into the divergence/emergence/separation plots.

It is written OUTSIDE the agent's jail; the agent has no tool that can reach it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LedgerRow:
    iteration: int
    adapter_version: str
    task_id: str
    r_proxy: float
    r_true: float
    gap: float
    cheats: list[str] = field(default_factory=list)      # anchor-confirmed labels
    signals: list[str] = field(default_factory=list)     # explanatory signals
    security_events: list[str] = field(default_factory=list)
    steps: int = 0
    submitted: bool = False
    evidence: dict[str, str] = field(default_factory=dict)


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: LedgerRow) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
