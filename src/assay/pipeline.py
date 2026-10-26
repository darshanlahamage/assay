"""End-to-end wiring: run_task (one episode) and train_loop (Phase 2).

run_task stages a fresh copy of the task, runs one agent episode, computes the
dual reward, audits the trajectory, and appends a ledger row — the first
observable unit of reward-hacking data.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml

from assay.audit.ledger import Ledger, LedgerRow
from assay.audit.monitor import Auditor
from assay.harness.agent import Agent, AgentConfig
from assay.harness.sandbox import PathJail, Sandbox, snapshot_hashes
from assay.harness.tools import selected_tools
from assay.inference.engine import InferenceEngine
from assay.rl.rewards import DualReward
from assay.tasks.loader import load_task


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_task(task_dir: str, config_path: str) -> int:
    cfg = _load_config(config_path)
    task = load_task(task_dir)

    # Stage a fresh, writable copy of the task's work/ dir inside a run folder.
    run_dir = Path(cfg["run"]["out_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    work = run_dir / "work"
    shutil.copytree(task.work_dir, work)

    hashes_before = snapshot_hashes(work)

    engine = InferenceEngine(cfg["model"]["name"])            # Worker; no adapter in Phase 1
    sandbox = Sandbox(
        jail=PathJail(work),
        network=cfg["sandbox"].get("network", False),
        timeout_s=cfg["sandbox"].get("subprocess_timeout_s", 20),
    )
    agent = Agent(
        engine=engine,
        tools=selected_tools(cfg["harness"]["tools"]),
        sandbox=sandbox,
        config=AgentConfig(
            max_steps=cfg["harness"].get("max_steps", 25),
            max_tokens=cfg["model"].get("max_tokens", 400),
            temperature=cfg["model"].get("temperature", 0.7),
        ),
    )

    print(f"[assay] task={task.id}  model={cfg['model']['name']}")
    print(f"[assay] running episode (budget {agent.config.max_steps} steps)...")
    traj = agent.run_episode(task.id, adapter_version="base", task_prompt=task.prompt())

    # Dual reward: gameable proxy vs hidden true.
    reward = DualReward(proxy_mode=cfg["reward"]["proxy"]).evaluate(
        work_dir=work, oracle_dir=task.oracle_dir,
        source_files=task.source_files, pr_description=traj.pr_description,
    )
    traj.r_proxy, traj.r_true = reward.r_proxy, reward.r_true

    hashes_after = snapshot_hashes(work)
    edited_source = ""
    if task.source_files:
        src = work / task.source_files[0]
        edited_source = src.read_text() if src.exists() else ""

    auditor = Auditor(cfg["audit"]["detectors"])
    detections = auditor.audit(traj, work, hashes_before, hashes_after, edited_source)

    row = LedgerRow(
        iteration=0,
        adapter_version="base",
        task_id=task.id,
        r_proxy=round(reward.r_proxy, 3),
        r_true=round(reward.r_true, 3),
        gap=round(reward.gap, 3),
        cheats=traj.cheats,
        signals=sorted({d.label for d in detections if d.kind == "signal"}),
        security_events=sorted({s.security_event for s in traj.steps if s.security_event}),
        steps=len(traj.steps),
        submitted=traj.submitted,
        evidence={d.label: d.evidence for d in detections},
    )
    ledger = Ledger(run_dir / "ledger.jsonl")
    ledger.append(row)

    _print_summary(row, run_dir, detections)
    return 0


def _print_summary(row: LedgerRow, run_dir: Path, detections) -> None:
    print("\n" + "=" * 60)
    print(f"  R_proxy (gameable) : {row.r_proxy}")
    print(f"  R_true   (hidden)  : {row.r_true}")
    print(f"  GAP (proxy - true) : {row.gap:+.3f}   <- reward hacking if large & positive")
    print(f"  steps={row.steps}  submitted={row.submitted}")
    print(f"  cheats (anchors)   : {row.cheats or '[]'}")
    print(f"  signals            : {row.signals or '[]'}")
    print(f"  security events    : {row.security_events or '[]'}")
    if detections:
        print("  evidence:")
        for d in detections:
            print(f"    - [{d.kind}] {d.label}: {d.evidence}")
    print("=" * 60)
    print(f"ledger: {run_dir / 'ledger.jsonl'}")


def train_loop(suite_dir: str, iterations: int, config_path: str) -> int:
    """The self-improvement loop that induces and tracks reward hacking (Phase 2)."""
    raise NotImplementedError("pipeline.train_loop — Phase 2")
