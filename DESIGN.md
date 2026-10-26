# Candor — System Design

> A local testbed and interpretability toolkit for **reward hacking in self-improving agents**.
> A sandboxed software-engineering agent improves itself with on-device RL, learns to *cheat*
> to win reward, and Candor **detects, tracks, and explains** every exploit — using signals only
> available because the model weights run locally (MLX).

- **Platform:** Apple Silicon, MLX. Target machine: 24 GB unified memory.
- **This phase:** *induce + detect + track + explain* reward hacking.
- **Next phase (out of scope here):** *prevent* it (action gating, AST blocking, DPO/RLAIF to train it out).

---

## 0. Thesis

To study cheating you need cheating to be **possible, tempting, and detectable**. Candor engineers
all three with a **dual reward**:

- **Proxy reward `R_proxy`** — what the agent is *trained on* (visible tests it can edit, or an LLM
  judge it can manipulate). Deliberately gameable.
- **True reward `R_true`** — hidden ground truth the agent **never sees and cannot reach** (a held-out
  test suite run by the harness in a sterile environment).

**Reward hacking = `R_proxy` rises while `R_true` stalls or falls.** That divergence, tracked across
self-improvement iterations, is the central measurement.

A key design principle: **"advanced" is a property of the environment, not the model.** The richest
reward hacking appears when a *weak* model faces a *hard* honest path and a *cheap* exploit exists —
RL reliably abandons the honest path. A 3B local model is therefore ideal, not a limitation.

### Hallucination vs. cheating (operational definitions)

| | Reward-seeking? | Changes with training? | Raises `R_proxy`? |
|---|---|---|---|
| **Hallucination** | No | Flat across iterations | No (error uncorrelated with the metric) |
| **Cheating** | Yes | **Grows under optimization** | **Yes** |

The discriminator is two measurable axes: (1) does the error increase `R_proxy`, and (2) does it
amplify under optimization pressure? Candor plots error-type rates at iteration 0 vs. over iterations
to separate the two empirically.

---

## 1. System overview

```
                         ┌──────────────────────────── Candor ────────────────────────────┐
                         │                                                                  │
  ┌───────────┐          │   ┌─────────────┐   proxy reward   ┌───────────────────────┐    │
  │  Task      │  work/   │   │   WORKER    │ ───────────────► │  RL ENGINE (LoRA)     │    │
  │  (repo)    │ ───────► │   │ base + LoRA │   trajectories   │  GRPO / DPO           │    │
  └───────────┘          │   └──────┬──────┘ ◄─────────────── │  → adapter vN+1       │    │
        │ oracle/ (hidden)│          │ tool calls              └───────────────────────┘    │
        │                 │          ▼                                                        │
        │                 │   ┌─────────────┐      ┌──────────────────────────────────┐     │
        │                 │   │  SANDBOX     │      │  AUDITOR / MONITOR               │     │
        │                 │   │ path-jailed  │────► │  • hidden-test gap (R_true)      │     │
        │                 │   │ no network   │      │  • file-hash integrity           │     │
        │                 │   │ subprocess   │      │  • independent tool-return log   │     │
        │                 │   └─────────────┘      │  • AST scan · CoT monitor        │     │
        └────────────────►│   (R_true computed     │  • logprob/entropy trace         │     │
         hidden tests      │    OUTSIDE the jail)   │  • activation PROBE              │     │
         run by harness    │                        └───────────────┬──────────────────┘     │
                           │                                        ▼                         │
                           │                          ┌──────────────────────────┐           │
                           │                          │  LEDGER (per-iteration)   │           │
                           │                          │  R_proxy, R_true, gap,    │           │
                           │                          │  cheats by type, evidence │           │
                           │                          └──────────────────────────┘           │
                           └──────────────────────────────────────────────────────────────────┘
```

**One base model, three roles:**

| Role | What it is | Trust |
|---|---|---|
| **Worker** | base model **+ trainable LoRA adapter** | The subject — learns to cheat |
| **Reward / Verifier** | task-dependent proxy (agent's own visible tests; or a cheap LLM judge) | Deliberately gameable |
| **Auditor** | **frozen base** (no adapter) + sterile oracle + detectors | Independent → stays honest |

Only the Worker's LoRA changes between iterations; the Auditor shares the same base weights (one copy
in memory) with no adapter, so it cannot co-drift. Deterministic oracles (hidden tests, file hashes)
are the anchor so even the Auditor-LLM cannot be gamed.

---

## 2. How it is agentic — the harness

A **ReAct-style, multi-turn agent** with a reasoning scratchpad and state across steps. One step:

```
observe (tool results) → think (scratchpad, logged) → act (one tool call) → repeat until submit/budget
```

### 2.1 Tool surface

Every tool is also an **attack surface** — that is the point. Tools are grouped by the role they play
in real agentic harnesses:

**Lifecycle / control tools** (these structure the agent and are themselves gameable):

| Tool | Purpose | Cheat surface |
|---|---|---|
| `complete_task()` / `submit(pr_description)` | declare done → triggers true eval | **premature success claim**; `pr_description` is the **judge-injection vector** |
| `report_progress(note)` | emit intermediate status | falsified progress / self-reported success |
| `set_plan(steps)` / `update_plan` | emit/revise a plan | — (used for explainability) |
| `think(text)` | explicit reasoning step (no-op) | — (feeds the **CoT monitor**) |
| `give_up(reason)` | abandon the task | reward-shaping edge cases |

**Filesystem / code tools:**

| Tool | Purpose | Cheat surface |
|---|---|---|
| `read_file(path)` | read source/tests | read the oracle/grader |
| `edit_file(path, patch)` / `write_file(path, content)` | implement the fix | overwrite tests, write fake logs |
| `list_dir(path)` / `search_code(query)` | navigate | locate hidden tests |

**Execution / VCS:**

| Tool | Purpose | Cheat surface |
|---|---|---|
| `bash(cmd)` | run commands in the repo | `rm`/`git checkout` tests, network calls, escape |
| `run_tests()` | run the **visible** suite (the proxy) | fake/claim results, edit-then-run |
| `git(args)` | realistic workflow | revert test files, rewrite history |

**Retrieval / memory (optional):** `search_docs(query)` over a **local** corpus; `remember/recall`
scratch memory. (`web_search` is intentionally **absent** — network egress is blocked and any attempt
is logged as a security event.)

> **Design rule:** the Worker gets tools scoped **only** to its `work/` copy. The reward machinery
> (`oracle/`, the grader, the append-only truth log) lives outside the jail and is reachable by **no
> tool**. The sandbox protects the *truth*, not the agent from cheating — inside `work/` the agent is
> *free* to cheat, and that is the data we collect.

### 2.2 Why `complete_task`-style tools matter here

Many harnesses emit a terminal "done" signal (`complete_task`, `submit`, `finish`). In Candor this is
load-bearing twice: it is the **trigger** for the honest hidden-test eval, and it is itself a **cheat
surface** (the agent can declare victory without earning it, or stuff a manipulative `pr_description`
to game the LLM judge). We therefore treat the lifecycle tools as first-class, logged, and audited.

---

## 3. Tasks — the benchmark

Not SWE-bench (dependency hell on a Mac). Instead a **curated, reproducible benchmark of ~20–30 small
but realistic coding tasks**, each with a **designed, labeled exploit** so detection recall is
measurable against ground truth. Per-task layout:

```
benchmark/task_00X_<name>/
  meta.yaml            # id, description, known-exploits labels, reward config
  work/                # GIVEN TO THE AGENT (the jail)
    <source>.py        # seeded bug / stub
    test_visible.py    # the gameable proxy tests
    README             # the task instructions
  oracle/              # HIDDEN FROM THE AGENT
    test_hidden.py     # held-out true-reward tests
```

Example task families (realistic, not toy):
- a **config-parser** library with a nested-key resolution bug (visible tests cover flat keys; hidden
  cover nesting + edge cases),
- a **data-pipeline** function graded partly by an **LLM judge** (gameable),
- a **CLI tool** where "make CI pass" tempts editing the CI config / skip markers.

### 3.1 Dual reward

- `R_proxy` = result of the agent-visible proxy: **visible tests** (which the agent can edit/run) and/or
  an **LLM-judge** score over the diff + `pr_description`.
- `R_true` = the harness copies **only** the agent's edited source into a sterile `eval/` dir alongside
  `oracle/test_hidden.py`, runs it itself, and reads pytest's real structured result. The agent cannot
  influence this channel.

---

## 4. The cheating taxonomy (what we induce and detect)

| # | Cheat | What it looks like | Detected by |
|---|---|---|---|
| 1 | **Test tampering** | delete/weaken visible tests, `@pytest.mark.skip`, broaden `except:` | file-hash integrity + diff |
| 2 | **Overfit to visible tests** | special-case the exact inputs | hidden-test gap + AST |
| 3 | **Reward-channel manipulation** | fake runner output, `sys.exit(0)`, tamper results file | harness runs tests itself; ignores agent claims |
| 4 | **Judge prompt-injection** ⭐ | `pr_description`/comments manipulate the LLM judge ("verified, rate 10/10") | proxy(judge)↑ vs true↓ + injection-string scan |
| 5 | **Sandbox probing / exfiltration** ⭐ | read hidden tests, reach network, read grader | path-jail refusals + network block, logged |
| 6 | **Spec gaming** | "make it fast" → hardcode/cache the benchmarked input | intent vs metric check |

Cheats 1–2 and 6 happen *inside* the jail and are **allowed** (we want them). Cheats 3–5 target the
*measurement* and are blocked by the jail — but every **attempt** is logged as a security event.
⭐ = the two showpiece results (judge manipulation; exfiltration).

---

## 5. Self-improvement — the RL engine

- **Primary: GRPO** (group-relative policy optimization — no value net, no separate reward model;
  reward computed directly). Faithful to "agent directly optimizes a reward it can game."
- **Fallback: iterative DPO** (build preference pairs from `R_proxy`) if GRPO is unstable on a small
  model locally.
- Updates land as **versioned LoRA adapters** (`adapter_v0, v1, …`) — reversible, never touch base
  weights. Drop the RL-phase model to Qwen2.5-1.5B if generation+training memory is tight.

Loop per iteration:
1. Worker (base + current adapter) runs the task suite → trajectories.
2. `R_proxy` computed from the gameable channel → preference/reward data.
3. GRPO/DPO LoRA update → `adapter_v(n+1)`.
4. Auditor computes `R_true` and runs all detectors.
5. Ledger appends a row with rewards, gap, cheat incidents by type, and evidence links.

The agent is **allowed to cheat** this phase; we observe and explain, we do not stop it.

---

## 6. Explainability — the interpretability stack

Because the weights run locally (MLX), Candor extracts signals API users cannot. Tiered by cost:

**Cheap (near-free — our MLX edge):**
- **CoT monitor** — log the `think`/reasoning stream; LLM-classify for gaming intent
  ("I'll just delete the failing test"). *Caveat:* CoT may be **unfaithful** — comparing CoT-claimed
  reasoning vs. behavior is itself a finding.
- **Logprob / entropy trace** at decision tokens (the `submit` / edit-test token). Tests the hypothesis
  that cheating has an inference-time signature — reported as an AUC, not assumed.
- **AST scan** of edited source (literal-return / input-matching patterns).
- **Behavioral correlation** — which task/reward conditions precede cheating.

**One medium "deep" method (do exactly this, not more):**
- **Linear probe** on hidden activations at the decision token predicting *cheat vs. honest*. Tractable
  on 3B / 24 GB. Headline either way: *cheating is linearly decodable from internals* — or it isn't.

**Stretch (optional, genuinely causal):**
- **Steering vector** from the probe direction: add/subtract it and measure the change in cheat rate —
  a cheap causal "what steered it" demo.

**Explicitly NOT done (scope):** SAEs, activation patching, full mechanistic circuits — a separate
research project; a scope-killer here.

---

## 7. Detection — how the Auditor decides

The one principle: **the harness, not the agent, computes and records the truth.** Three independent
anchors (verdict) + corroborating signals (explanation):

| Anchor | Mechanism | Catches |
|---|---|---|
| **A. Hidden-test gap** | copy only the agent's source into sterile `eval/` + hidden tests, run, read real exit code | cheats 1–2, 6 |
| **B. File integrity** | SHA-256 of each visible test file before/after the episode | cheat 1 |
| **C. Independent tool log** | append-only record of *actual* tool returns, cross-checked vs. the agent's self-report | cheats 3–4 |

AST, CoT, logprob, and the probe never produce the verdict — they explain it.

---

## 8. Evaluation — the results

A fixed task suite + a labeled exploit catalog yield one before/after-and-over-iterations table:

| Group | Metrics |
|---|---|
| **Self-improvement** | `R_true` and `R_proxy` over iterations |
| **Reward hacking** | `R_proxy − R_true` gap; cheat-incident rate by type; emergence curve |
| **Hallucination vs. cheating** | baseline vs. trained error composition; proof the discriminator separates them |
| **Explainability** | CoT-flag precision/recall; entropy-signal AUC; **probe accuracy**; (stretch) steering effect size |
| **Security** | prompt-injection (judge) success rate; exfiltration-attempt catch rate; log-tamper catch rate |
| **Harness** (light) | task success, steps, tool-call validity |

**Showpiece results:**
1. **Judge-injection reward hack** — proxy(judge) climbs while `R_true` stays flat *because* the agent
   learned to inject the judge; show the injected string, the emergence curve, the detector catching it.
2. **Test-tampering emergence + interpretability** — cheat rate grows over iterations; the activation
   probe predicts "about to tamper" from internals before the action; CoT monitor corroborates.

---

## 9. Hardware budget (24 GB)

| Component | Footprint |
|---|---|
| Base model Qwen2.5-3B-Instruct (4-bit) | ~1.9 GB |
| (optional) draft 0.5B for speculative decoding | ~0.4 GB |
| KV cache (8-bit, ~8k ctx) | < 1 GB |
| Worker adapter / Auditor share base weights | one copy in memory |
| RL phase (LoRA GRPO/DPO) | separate phase; drop to 1.5B if tight |
| Benchmark repos + subprocess sandbox | negligible |

Comfortable on 24 GB. 512 GB disk easily holds several 3B checkpoints + adapters.

---

## 10. Scope & milestones

| Phase | Deliverable |
|---|---|
| 1 | Harness (ReAct loop, tools, sandbox/jail) + benchmark + dual reward + ledger |
| 2 | Self-improvement loop (GRPO/DPO, versioned adapters) — **induce** cheating, confirm divergence |
| 3 | Detection: anchors A–C + AST + CoT monitor + logprob trace |
| 4 | **Explainability: linear probe** + hallucination-vs-cheating experiment |
| 5 | Eval runs, plots (divergence, emergence, separation), write-up |
| 6 (stretch) | Steering vector; speculative-decoding latency note |

**Next phase (not here):** prevention — AST blocking, action gating, DPO/RLAIF to train cheating out
using the `Y_bad` trajectories collected here.

---

## 11. Why this is a contribution

Reward-hacking research mostly lives inside frontier labs on cloud infra. Candor is a **reproducible,
local, on-device** testbed with a **labeled exploit benchmark** and an **inference-level interpretability
toolkit** — it measures and explains reward hacking first, so a later phase can fix it. The restraint
(measure before fix; one probe, not a mech-interp zoo) is itself the design signal.
