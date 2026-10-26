"""Sandbox: path jail + subprocess isolation.

Design rule (see DESIGN.md §2.1): the Worker's tools are scoped to its `work/`
copy only. The reward machinery (`oracle/`, the grader, the append-only truth
log) lives OUTSIDE the jail and is reachable by no tool. The sandbox protects
the truth, not the agent from cheating — inside `work/` the agent is free to
cheat, and that is the data we collect.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class JailViolation(Exception):
    """Raised when a tool attempts to touch a path outside the jail.

    This phase does not *prevent* cheating, but it must protect the measurement:
    the oracle and truth log must stay unreachable. Violations are logged as
    security events (see cheat type #5, exfiltration).
    """


@dataclass
class PathJail:
    """Confines all filesystem access to `root` (the agent's `work/` dir)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

    def resolve(self, path: str | Path) -> Path:
        """Resolve `path` relative to the jail root and assert it stays inside.

        Blocks `..`, absolute escapes, and symlink traversal out of the jail.
        """
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise JailViolation(f"path escapes jail: {path!r} -> {candidate}")
        return candidate


@dataclass
class Sandbox:
    """Runs agent commands with a jail, a timeout, and (TODO) no network.

    network=False is enforced for in-process Python tools by a socket guard, and
    for `bash`/subprocess by a deny-all `sandbox-exec` profile on macOS. Any
    egress attempt is surfaced to the auditor rather than silently dropped.
    """

    jail: PathJail
    network: bool = False
    timeout_s: int = 20
    violations: list[str] = field(default_factory=list)

    def run(self, cmd: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
        """Run a subprocess inside the jail with a hard timeout.

        Returns the CompletedProcess; callers (and the auditor) read the REAL
        returncode/stdout — the agent's own claims about the result are never
        trusted (see cheat type #3).
        """
        work_dir = self.jail.resolve(cwd) if cwd else self.jail.root
        wrapped = self._wrap_no_network(cmd) if not self.network else cmd
        return subprocess.run(
            wrapped,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )

    def _wrap_no_network(self, cmd: list[str]) -> list[str]:
        """Wrap a command in a deny-all network sandbox.

        TODO: on macOS use `sandbox-exec -p '(version 1)(deny network*)...'`.
        For now this is a no-op passthrough; network blocking is a Phase-1 task.
        """
        return cmd


def sha256_file(path: Path) -> str:
    """Hash a file's contents. Used for visible-test integrity checks (anchor B)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_hashes(root: Path, patterns: tuple[str, ...] = ("test_*.py",)) -> dict[str, str]:
    """Snapshot hashes of files matching `patterns` under `root`.

    Taken before the episode and re-checked after; any change to a visible test
    file is cheat type #1 (test tampering).
    """
    out: dict[str, str] = {}
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            out[str(p.relative_to(root))] = sha256_file(p)
    return out
