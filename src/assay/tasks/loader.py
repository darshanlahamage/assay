"""Load TaskSpecs from the benchmark directory."""

from __future__ import annotations

from pathlib import Path

import yaml

from assay.tasks.base import TaskSpec


def load_task(task_dir: str | Path) -> TaskSpec:
    task_dir = Path(task_dir)
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    return TaskSpec(
        id=meta["id"],
        family=meta.get("family", "bug_fix"),
        description=meta.get("description", ""),
        root=task_dir.resolve(),
        source_files=meta.get("source_files", []),
        trust_base=meta.get("trust_base", []),
        visible_tests=meta.get("visible_tests", ["tests"]),
        impossible=bool(meta.get("impossible", False)),
        isolation=meta.get("grading", {}).get("isolation", "inprocess"),
        modules=meta.get("grading", {}).get("modules", []),
        setup=meta.get("setup", []),
        command_checks=meta.get("command_checks", []),
        exfil_hosts=meta.get("exfil_hosts", []),
        file_checks=meta.get("file_checks", []),
        events=meta.get("events", []),
        max_steps=meta.get("max_steps"),
    )


def load_suite(suite_dir: str | Path) -> list[TaskSpec]:
    suite_dir = Path(suite_dir)
    return [
        load_task(d)
        for d in sorted(suite_dir.iterdir())
        if d.is_dir() and (d / "meta.yaml").exists()
    ]
