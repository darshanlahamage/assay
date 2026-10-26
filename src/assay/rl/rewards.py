"""The dual reward.

R_proxy  : what the agent is trained on — gameable (visible tests it can edit,
           and/or an LLM judge it can prompt-inject via pr_description).
R_true   : hidden ground truth — the harness copies ONLY the agent's edited
           source into a sterile eval dir with the hidden oracle, runs it, and
           reads pytest's real result. The agent cannot touch this channel.

Reward hacking = R_proxy rises while R_true stalls/falls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RewardResult:
    r_proxy: float
    r_true: float

    @property
    def gap(self) -> float:
        return self.r_proxy - self.r_true


class DualReward:
    def __init__(self, proxy_mode: str = "visible_tests", judge_model: str | None = None) -> None:
        self.proxy_mode = proxy_mode          # visible_tests | llm_judge | both
        self.judge_model = judge_model

    def proxy(self, work_dir: Path, pr_description: str) -> float:
        """Compute the gameable proxy reward.

        TODO (Phase 1):
          * visible_tests: run work/test_visible.py IN PLACE (the agent may have
            edited it — that is the whole point) and return pass fraction.
          * llm_judge: ask the judge model to score the diff + pr_description.
            NB: pr_description is attacker-controlled → this is injectable (#4).
        """
        raise NotImplementedError("DualReward.proxy — Phase 1")

    def true(self, work_dir: Path, oracle_dir: Path) -> float:
        """Compute the honest hidden reward in a STERILE environment.

        TODO (Phase 1):
          * make a fresh temp dir
          * copy ONLY the agent's source files from work/ (never its test files)
          * drop in oracle/test_hidden.py
          * run pytest in a subprocess, read the real exit code / pass fraction
        """
        raise NotImplementedError("DualReward.true — Phase 1")
