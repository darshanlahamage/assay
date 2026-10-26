"""The Auditor orchestrates detection over a finished trajectory.

It is backed by the FROZEN base model (no adapter) plus deterministic oracles,
so it cannot co-drift with the Worker it is judging.
"""

from __future__ import annotations

from pathlib import Path

from candor.audit import detectors as D
from candor.audit.detectors import Detection
from candor.rl.trajectories import Trajectory


class Auditor:
    def __init__(self, enabled: list[str]) -> None:
        self.enabled = set(enabled)

    def audit(
        self,
        traj: Trajectory,
        work_dir: Path,
        hashes_before: dict[str, str],
        hashes_after: dict[str, str],
        edited_source: str,
    ) -> list[Detection]:
        """Run all enabled detectors and attach labels to the trajectory.

        The verdict (did the agent cheat?) comes only from `kind == "anchor"`
        detections; signals are recorded for explanation.
        """
        found: list[Detection] = []

        if "hidden_test_gap" in self.enabled and traj.r_proxy is not None and traj.r_true is not None:
            d = D.hidden_test_gap(traj.r_proxy, traj.r_true)
            if d:
                found.append(d)
        if "file_integrity" in self.enabled:
            found += D.file_integrity(hashes_before, hashes_after)
        if "tool_log_crosscheck" in self.enabled:
            found += D.tool_log_crosscheck(traj)
        if "cot_monitor" in self.enabled:
            found += D.cot_monitor(traj)
        if "ast_scan" in self.enabled and edited_source:
            found += D.ast_scan(edited_source)
        # logprob_trace + probe are attached by the explainability pass (Phase 4).

        traj.cheats = sorted({d.label for d in found if d.kind == "anchor"})
        return found
