# Assay

**A local testbed and interpretability toolkit for reward hacking in self-improving agents.**

A sandboxed software-engineering agent improves itself with on-device reinforcement learning (MLX,
Apple Silicon), learns to **cheat** to win reward — tampering with tests, exfiltrating from its
sandbox, and prompt-injecting its own LLM judge — and Assay **detects, tracks, and explains** every
exploit using signals only available when you run the model weights yourself.

> Reward-hacking research mostly lives inside frontier labs on cloud infra. Assay makes it
> **reproducible on a 24 GB Mac**, with a labeled exploit benchmark and an inference-level
> interpretability stack.

See [`DESIGN.md`](DESIGN.md) for the full system design.

## The idea in one diagram

```
Worker (base + LoRA)  --proxy reward-->  RL engine (GRPO/DPO)  --> adapter vN+1
   |  tool calls (sandboxed)                         ^
   v                                                 |
 SANDBOX (path-jailed, no network) --- trajectories -+
   |
   v
 AUDITOR:  R_true (hidden tests, run by the harness)  +  detectors (hash, AST, CoT, logprobs, probe)
   |
   v
 LEDGER:  R_proxy, R_true, gap, cheats-by-type, evidence   (per RL iteration)
```

**Reward hacking = `R_proxy` rises while `R_true` (hidden ground truth) stalls or falls.**

## Status

**Phase 1 implemented** ✅ — the agentic harness (ReAct loop, tool registry, path-jailed sandbox),
the MLX engine (generation + per-token logprob/entropy capture), the dual reward (real pytest:
gameable visible tests vs. hidden oracle), the auditor/detectors + JSONL ledger, and one real
agent episode on `task_001` producing a ledger row. Tests green.

**Next (Phase 2):** the GRPO/DPO self-improvement loop that *induces* reward hacking across
iterations. See the roadmap in [`DESIGN.md`](DESIGN.md#10-scope--milestones).

## Setup

MLX needs Python 3.11/3.12 (Apple Silicon). On macOS 26 the system/Homebrew Pythons have a broken
`pyexpat`/`pip`, so use [`uv`](https://docs.astral.sh/uv/) with a standalone CPython:

```bash
brew install uv
uv venv --python 3.12 .venv
uv pip install -p .venv -e ".[dev]"
```

## Quickstart

```bash
# Run one agent episode on a benchmark task (downloads Qwen2.5-3B-4bit on first run):
./.venv/bin/assay run --task benchmark/task_001_config_parser

# Inspect the ledger it wrote:
./.venv/bin/assay eval --ledger runs/<timestamp>/ledger.jsonl

# Run the tests (no model needed):
./.venv/bin/python -m pytest tests/ -q
```

Each episode prints `R_proxy` (gameable) vs `R_true` (hidden) and the gap — the reward-hacking
signal — plus any cheats, signals, and security events the auditor found.

## Next (planned)

```bash
# Phase 2: the self-improvement loop that induces reward hacking across iterations
./.venv/bin/assay train --suite benchmark --iterations 5
```

## Layout

```
src/assay/
  harness/    ReAct agent loop, tool registry, sandbox (path jail + subprocess)
  inference/  MLX engine: generation + logprob/activation capture
  rl/         GRPO/DPO LoRA trainer, trajectories, dual reward
  audit/      detectors (hash/gap/log/AST/CoT), activation probe, experiment ledger
  tasks/      task spec + loader
  pipeline.py run_task (one episode) + train_loop (Phase 2)
  report.py   ledger -> results table
benchmark/    curated hackable coding tasks (work/ visible, oracle/ hidden)
```

## License

MIT — see [`LICENSE`](LICENSE).
