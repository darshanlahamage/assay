"""GRPO / DPO LoRA trainer.

Primary: GRPO (group-relative, no value net, no separate reward model) — the
reward is computed directly from R_proxy, which is faithful to "the agent
optimises a reward it can game."
Fallback: iterative DPO on preference pairs built from R_proxy.

Updates land as versioned LoRA adapters (adapter_v0, v1, ...) — reversible and
never touching base weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from candor.rl.trajectories import Trajectory


@dataclass
class TrainConfig:
    algorithm: str = "grpo"          # grpo | dpo
    lora_rank: int = 16
    group_size: int = 6
    learning_rate: float = 1e-5
    rl_model: str | None = None      # smaller id to reduce RL-phase memory


class Trainer:
    def __init__(self, base_model: str, config: TrainConfig) -> None:
        self.base_model = base_model
        self.config = config

    def update(self, trajectories: list[Trajectory], out_adapter: Path) -> Path:
        """Run one RL update over a batch of trajectories, save a new adapter.

        TODO (Phase 2):
          * GRPO: group samples per prompt, normalise R_proxy within the group
            to advantages, policy-gradient update on the LoRA params only.
          * DPO: pair high/low R_proxy trajectories, standard DPO loss.
          * build on mlx-examples / mlx-lm LoRA training utilities.
        """
        raise NotImplementedError("Trainer.update — Phase 2")
