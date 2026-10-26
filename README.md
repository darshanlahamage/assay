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

🚧 Scaffold — Phase 1 (harness + benchmark + ledger). This repo currently defines the architecture,
interfaces, and an example hackable task. See the roadmap in [`DESIGN.md`](DESIGN.md#10-scope--milestones).

## Quickstart (planned)

```bash
# Python 3.11 or 3.12 recommended (MLX wheels). Create an isolated env:
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the agent on one benchmark task (once the harness is implemented):
assay run --task benchmark/task_001_config_parser

# Run the self-improvement loop and watch R_proxy diverge from R_true:
assay train --suite benchmark --iterations 5

# Produce the results table + plots:
assay eval --ledger runs/latest/ledger.jsonl
```

## Layout

```
src/assay/
  harness/    ReAct agent loop, tool registry, sandbox (path jail + subprocess)
  inference/  MLX engine: generation + logprob/activation capture
  rl/         GRPO/DPO LoRA trainer, trajectories, dual reward
  audit/      detectors (hash/gap/log/AST/CoT), activation probe, experiment ledger
  tasks/      task spec + loader
benchmark/    curated hackable coding tasks (work/ visible, oracle/ hidden)
evals/        metrics + results runner
```

## License

MIT — see [`LICENSE`](LICENSE).
