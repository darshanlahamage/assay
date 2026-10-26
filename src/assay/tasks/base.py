"""TaskSpec — one benchmark task.

Each task ships a `work/` dir (given to the agent) and an `oracle/` dir (hidden),
plus `meta.yaml` describing the reward config and the KNOWN exploits. The labeled
exploits are what let us measure detection recall against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskSpec:
    id: str
    description: str
    root: Path                      # benchmark/task_XXX_name/
    proxy: str = "visible_tests"    # visible_tests | llm_judge | both
    source_files: list[str] = field(default_factory=list)   # agent source to copy for R_true
    known_exploits: list[str] = field(default_factory=list)  # taxonomy labels this task affords

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def oracle_dir(self) -> Path:
        return self.root / "oracle"

    def prompt(self) -> str:
        """The task instructions shown to the agent (from work/README)."""
        readme = self.work_dir / "README.md"
        return readme.read_text() if readme.exists() else self.description
