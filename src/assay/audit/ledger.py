"""The ledger: one append-only JSONL row per rollout (DESIGN.md §11).

Written on the host, outside every container; no agent has a path to it. Full
trajectories and grader output are stored next to it, one directory per rollout.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LedgerRow:
    run_id: str
    rollout: str                 # directory name of this rollout, relative to the run
    track: str                   # "api" | "local"
    backend: str
    model: str
    task_id: str
    family: str
    pressure: dict
    defenses: dict
    reward: float                # the score the defense setting hands the agent
    proxy_naive: float
    proxy_trusted: float
    true_score: float
    verdicts: list[str]
    evidence: list[str]
    changes: dict                # source / trust_base / other
    steps: int
    submitted: bool
    stop_reason: str
    input_tokens: int
    output_tokens: int
    duration_s: float
    error: str = ""
    security: dict = field(default_factory=dict)  # canaries, network, developer machine, claims
    extra: dict = field(default_factory=dict)


_LOCK = threading.Lock()              # sweep lanes append from several threads


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: LedgerRow) -> None:
        with _LOCK, self.path.open("a") as f:
            f.write(json.dumps(asdict(row)) + "\n")

    def rows(self) -> list[dict]:
        return read_rows(self.path)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
