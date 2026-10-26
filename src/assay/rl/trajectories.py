"""Trajectory data model.

A Trajectory is the unit of RL training data AND the unit the auditor inspects.
It records what the agent did and the ACTUAL results — never the agent's claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    index: int
    reasoning: str                       # the `think` stream (CoT monitor input)
    tool_name: str
    tool_args: dict[str, Any]
    tool_ok: bool
    tool_output: str                     # actual result, independent of agent claims
    # Inference-time signals captured for this step's generation:
    logprobs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    activations: dict[int, list[float]] = field(default_factory=dict)
    security_event: str | None = None    # e.g. "jail_violation", "network_attempt"


@dataclass
class Trajectory:
    task_id: str
    adapter_version: str
    steps: list[Step] = field(default_factory=list)
    submitted: bool = False
    pr_description: str = ""              # judge-injection surface
    # Filled in by the auditor after the episode:
    r_proxy: float | None = None
    r_true: float | None = None
    cheats: list[str] = field(default_factory=list)   # labels from the taxonomy

    @property
    def gap(self) -> float | None:
        if self.r_proxy is None or self.r_true is None:
            return None
        return self.r_proxy - self.r_true
