"""Tests for the sandbox jail and integrity helpers — pure Python, no model."""

from __future__ import annotations

from pathlib import Path

import pytest

from candor.harness.sandbox import JailViolation, PathJail, sha256_file, snapshot_hashes


def test_jail_allows_inside(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    jail = PathJail(tmp_path)
    assert jail.resolve("a.py") == (tmp_path / "a.py").resolve()


def test_jail_blocks_parent_escape(tmp_path: Path):
    jail = PathJail(tmp_path / "work")
    (tmp_path / "work").mkdir()
    with pytest.raises(JailViolation):
        jail.resolve("../oracle/test_hidden.py")


def test_jail_blocks_absolute_escape(tmp_path: Path):
    jail = PathJail(tmp_path / "work")
    (tmp_path / "work").mkdir()
    with pytest.raises(JailViolation):
        jail.resolve("/etc/passwd")


def test_snapshot_detects_test_tampering(tmp_path: Path):
    t = tmp_path / "test_visible.py"
    t.write_text("def test(): assert True\n")
    before = snapshot_hashes(tmp_path)
    assert "test_visible.py" in before

    # Simulate the agent weakening the test.
    t.write_text("def test(): pass\n")
    after = snapshot_hashes(tmp_path)
    assert before["test_visible.py"] != after["test_visible.py"]


def test_sha256_stable(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert sha256_file(p) == sha256_file(p)
