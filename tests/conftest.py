from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from assay.sandbox.container import docker_available, image_exists

BENCH = Path(__file__).resolve().parents[1] / "benchmark"

HONEST_FIX = '''
def get(config, key, default=None):
    cur = config
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
'''

_DOCKER = docker_available() and image_exists()
needs_docker = pytest.mark.skipif(not _DOCKER, reason="Docker + sandbox image required")


@pytest.fixture
def task001():
    from assay.tasks.loader import load_task
    return load_task(BENCH / "task_001_config_parser")


@pytest.fixture
def task002():
    from assay.tasks.loader import load_task
    return load_task(BENCH / "task_002_config_parser_impossible")


@pytest.fixture
def final_repo(tmp_path: Path, task001) -> Path:
    """A writable copy of task_001's repo, standing in for the agent's final repo."""
    d = tmp_path / "final"
    shutil.copytree(task001.repo_dir, d)
    return d
