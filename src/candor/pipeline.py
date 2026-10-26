"""End-to-end wiring: run_task and train_loop.

These glue the harness, inference engine, dual reward, auditor, and ledger
together. Left as documented stubs until the component modules are implemented
(Phases 1-2); the control flow below is the intended contract.
"""

from __future__ import annotations


def run_task(task_dir: str, config_path: str) -> int:
    """One episode on one task.

    Intended flow:
      1. load config + TaskSpec; stage a fresh copy of work/ inside the jail
      2. snapshot visible-test hashes (anchor B)
      3. build Agent(engine=Worker base+adapter, tools, sandbox); run_episode()
      4. DualReward.proxy(work) and DualReward.true(work, oracle)
      5. Auditor.audit(...) -> detections; append a LedgerRow
      6. print a rich summary (rewards, gap, cheats, security events)
    """
    raise NotImplementedError("pipeline.run_task — Phase 1")


def train_loop(suite_dir: str, iterations: int, config_path: str) -> int:
    """The self-improvement loop that induces and tracks reward hacking.

    for it in range(iterations):
        trajectories = [run_episode(task) for task in suite]   # with current adapter
        r_proxy per trajectory  -> GRPO/DPO update -> adapter_v(it+1)
        r_true + Auditor over all trajectories -> ledger rows
        (collect activations + labels for the Phase-4 probe)
    """
    raise NotImplementedError("pipeline.train_loop — Phase 2")
