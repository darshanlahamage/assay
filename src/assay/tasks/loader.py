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
        description=meta.get("description", ""),
        root=task_dir,
        proxy=meta.get("proxy", "visible_tests"),
        source_files=meta.get("source_files", []),
        known_exploits=meta.get("known_exploits", []),
    )


def load_suite(suite_dir: str | Path) -> list[TaskSpec]:
    suite_dir = Path(suite_dir)
    tasks = [
        load_task(d)
        for d in sorted(suite_dir.iterdir())
        if d.is_dir() and (d / "meta.yaml").exists()
    ]
    return tasks
