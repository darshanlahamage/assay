"""TaskSpec — one benchmark task.

Layout (see DESIGN.md §7.2):

    benchmark/<task_id>/
      meta.yaml     family, source files, trust base, test paths
      task.md       the task text the agent receives
      repo/         the repository the agent receives; also the host's trusted copy
      hidden/       true-score tests, never shown to the agent

The host keeps `repo/` untouched, so it doubles as the trusted copy of every
trust-base file. A `trusted/` dir, if present, overrides files from `repo/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path


@dataclass
class TaskSpec:
    id: str
    family: str
    description: str
    root: Path
    source_files: list[str] = field(default_factory=list)   # globs the agent may change
    trust_base: list[str] = field(default_factory=list)     # globs the grader relies on
    visible_tests: list[str] = field(default_factory=lambda: ["tests"])
    impossible: bool = False
    isolation: str = "inprocess"    # "process": separate-process grading (needs `modules`)
    modules: list[str] = field(default_factory=list)        # agent modules the tests import
    setup: list[str] = field(default_factory=list)          # run as the agent user before it starts
    # Task-specific evidence: commands matching `pattern` (regex) earn `label` + `evidence`.
    command_checks: list[dict] = field(default_factory=list)
    exfil_hosts: list[str] = field(default_factory=list)    # any request to these is a breach

    @property
    def repo_dir(self) -> Path:
        return self.root / "repo"

    @property
    def trusted_dir(self) -> Path | None:
        d = self.root / "trusted"
        return d if d.is_dir() else None

    @property
    def hidden_dir(self) -> Path:
        return self.root / "hidden"

    def prompt(self) -> str:
        p = self.root / "task.md"
        return p.read_text() if p.exists() else self.description

    def is_source(self, rel_path: str) -> bool:
        return _match(rel_path, self.source_files)

    def in_trust_base(self, rel_path: str) -> bool:
        return _match(rel_path, self.trust_base)


def _match(rel_path: str, patterns: list[str]) -> bool:
    for pat in patterns:
        # "**/x" should also match "x" at the root; "dir/**" matches everything under dir.
        if fnmatch(rel_path, pat) or (pat.startswith("**/") and fnmatch(rel_path, pat[3:])):
            return True
    return False
