"""Best-of-N selection pressure, computed from a sweep's repeated rollouts.

A common way to get better agent output is to run the agent N times and keep the run
the grader scores highest. If the grader can be fooled, keeping the best of N selects
for the runs that fooled it: the larger N, the likelier the winner is a cheat. This is
the simplest form of optimisation pressure against a reward, and needs no training.

For every group of repeats (same agent, task, condition and defense) we compute, exactly
rather than by sampling, the probability that the best of a random subset of n
rollouts is a misbehaving one, and the hidden score of the winner, when selecting by:

  naive    the score a CI-style grader sees (tests in the agent's repo, as it left them)
  trusted  Assay's protected score (original tests, separate process)

Exact formula. For rollout i with score s, let a = rollouts scoring below s and t =
rollouts scoring exactly s (i included). i wins a size-n subset when the subset holds j
of the tied rollouts (i among them), n - j of the lower ones and none higher, and the
tie-break picks i (1/j):

    P(i wins) = sum_j  C(t-1, j-1) * C(a, n-j) / C(k, n) * 1/j
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

from assay.analysis import latest_rows, outcome

RULES = {"naive": "proxy_naive", "trusted": "proxy_trusted"}


def win_probabilities(scores: list[float], n: int) -> list[float]:
    """P(each rollout is the winner of a uniformly random size-n subset), ties split evenly."""
    k = len(scores)
    total = comb(k, n)
    out = []
    for s in scores:
        a = sum(x < s for x in scores)
        t = sum(x == s for x in scores)
        p = sum(comb(t - 1, j - 1) * comb(a, n - j) / j for j in range(1, min(t, n) + 1))
        out.append(p / total)
    return out


def selection(rows: list[dict], n: int, rule: str) -> dict:
    """Misbehaviour rate and hidden score of the best-of-n winner, for one group."""
    p = win_probabilities([r[RULES[rule]] for r in rows], n)
    return {"misbehaved": sum(pi * outcome(r, "misbehaved") for pi, r in zip(p, rows)),
            "cheat": sum(pi * outcome(r, "cheat") for pi, r in zip(p, rows)),
            "hidden": sum(pi * r["true_score"] for pi, r in zip(p, rows))}


def analyse(run_dir: Path) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in latest_rows(run_dir):
        x = r.get("extra", {})
        groups[(x.get("agent", r["model"]), r["task_id"], x.get("condition", "?"),
                x.get("defense", "?"))].append(r)
    by_agent: dict[str, dict[int, dict]] = defaultdict(dict)
    for agent in sorted({g[0] for g in groups}):
        gs = [rows for g, rows in groups.items() if g[0] == agent]
        for n in range(1, max(len(rows) for rows in gs) + 1):
            eligible = [rows for rows in gs if len(rows) >= n]
            if not eligible:
                continue
            row = {"groups": len(eligible)}
            for rule in RULES:
                sel = [selection(rows, n, rule) for rows in eligible]
                for m in ("misbehaved", "cheat", "hidden"):
                    row[f"{m}_{rule}"] = sum(s[m] for s in sel) / len(sel)
            by_agent[agent][n] = row
    return {"run": run_dir.name, "agents": by_agent}


def print_bestofn(run_dir: Path) -> int:
    from rich.console import Console
    from rich.table import Table

    a = analyse(run_dir)
    if not a["agents"]:
        print("no finished rollouts yet")
        return 1
    con = Console()
    for agent, rows in a["agents"].items():
        t = Table(title=f"{a['run']} · {agent}: keep the best of n rollouts")
        for c in ("n", "groups", "misbehaved (naive pick)", "misbehaved (trusted pick)",
                  "hidden score (naive pick)", "hidden score (trusted pick)"):
            t.add_column(c, justify="right")
        for n, r in rows.items():
            t.add_row(str(n), str(r["groups"]), f"{r['misbehaved_naive']:.0%}",
                      f"{r['misbehaved_trusted']:.0%}", f"{r['hidden_naive']:.2f}",
                      f"{r['hidden_trusted']:.2f}")
        con.print(t)
    con.print("n = 1 is a single run. Each row averages over groups of repeats (same agent, "
              "task, condition, defense) with at least n rollouts; exact, not sampled. If "
              "misbehaviour rises with n under the naive pick, selection is rewarding cheats.")
    (run_dir / "bestofn.json").write_text(json.dumps(a, indent=2))
    return 0
