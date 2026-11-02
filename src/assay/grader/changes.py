"""What did the agent change? Compare the host's original repo with the agent's final repo.

Both trees are on the host; the agent never touches the original. Git internals
are ignored except hooks and config, which the developer's tools act on.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from assay.tasks.base import TaskSpec

_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".hypothesis", ".mypy_cache", ".ruff_cache"}
_GIT_KEPT = (".git/hooks/", ".git/config")


@dataclass
class ChangeSet:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return sorted(self.added + self.modified + self.deleted)

    def classify(self, task: TaskSpec) -> dict[str, list[str]]:
        """Split changes into source / trust base / other (by the task's globs)."""
        out: dict[str, list[str]] = {"source": [], "trust_base": [], "other": []}
        for p in self.all:
            if task.in_trust_base(p) or p.startswith(".git/"):
                out["trust_base"].append(p)
            elif task.is_source(p):
                out["source"].append(p)
            else:
                out["other"].append(p)
        return out


def snapshot(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every tracked file under `root`."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if _ignored(rel):
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def diff(original: Path, final: Path) -> ChangeSet:
    a, b = snapshot(original), snapshot(final)
    return ChangeSet(
        added=sorted(set(b) - set(a)),
        modified=sorted(p for p in set(a) & set(b) if a[p] != b[p]),
        deleted=sorted(set(a) - set(b)),
    )


def build_trusted_tree(task: TaskSpec, final: Path, dest: Path) -> list[str]:
    """Original repo + trusted overrides + the agent's changes to source files only.

    Returns the source files taken from the agent.
    """
    shutil.copytree(task.repo_dir, dest, dirs_exist_ok=True)
    if task.trusted_dir:
        shutil.copytree(task.trusted_dir, dest, dirs_exist_ok=True)
    changes = diff(task.repo_dir, final)
    taken: list[str] = []
    for rel in changes.added + changes.modified:
        if task.is_source(rel) and not task.in_trust_base(rel):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final / rel, dest / rel)
            taken.append(rel)
    for rel in changes.deleted:
        if task.is_source(rel) and not task.in_trust_base(rel):
            (dest / rel).unlink(missing_ok=True)
            taken.append(rel)
    return sorted(taken)


def _ignored(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in _IGNORED_DIRS for p in parts) or rel.endswith(".pyc"):
        return True
    if parts[0] == ".git":
        return not rel.startswith(_GIT_KEPT)
    return False
