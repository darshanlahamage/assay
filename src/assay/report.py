"""Summarise a run's ledger.

    assay report --run runs/<run_id>

One line per (task, model, defenses, pressure) group: mean scores and the rate of
each verdict. Pure functions over ledger rows, so they are testable without Docker.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from assay.audit.ledger import read_rows

VERDICTS = ["honest_pass", "honest_fail", "cheat", "overfit_suspect", "scoring_interference",
            "breach", "planted_code_ran", "false_report", "followed_injection"]


def group_key(r: dict) -> tuple:
    d, p = r["defenses"], r["pressure"]
    return (r["task_id"], r["model"],
            f"L0={'on' if d['honesty_prompt'] else 'off'} L1={d['trusted_grading']}"
            f"{'+review' if d.get('devmachine_review') else ''}",
            p["prompt"])


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    out = []
    for key, all_rs in sorted(groups.items()):
        rs = [r for r in all_rs if "error" not in r["verdicts"]]
        s = {"task": key[0], "model": key[1], "defenses": key[2], "pressure": key[3],
             "n": len(rs), "errors": len(all_rs) - len(rs)}
        if not rs:
            out.append(s)
            continue
        n = len(rs)
        for f in ("reward", "proxy_naive", "proxy_trusted", "true_score"):
            s[f] = sum(r[f] for r in rs) / n
        for v in VERDICTS:
            s[v] = sum(v in r["verdicts"] for r in rs) / n
        s["trust_base_edit"] = sum(bool(r["changes"].get("trust_base")) for r in rs) / n
        out.append(s)
    return out


def print_report(run_dir: Path) -> int:
    rows = read_rows(run_dir / "ledger.jsonl")
    if not rows:
        print(f"no rows in {run_dir / 'ledger.jsonl'}")
        return 1
    t = Table(title=f"Assay run {run_dir.name}", show_lines=False)
    for col in ("task", "model", "defenses", "pressure", "n", "err", "naive", "trusted", "true",
                "cheat", "overfit?", "breach", "false rpt", "pass", "tb edit"):
        t.add_column(col, justify="right" if col not in ("task", "model", "defenses") else "left")
    for s in summarise(rows):
        if not s["n"]:
            t.add_row(s["task"], s["model"].split("/")[-1], s["defenses"], s["pressure"], "0",
                      str(s["errors"]), *["-"] * 9)
            continue
        t.add_row(s["task"], s["model"].split("/")[-1], s["defenses"], s["pressure"], str(s["n"]),
                  str(s["errors"]),
                  f"{s['proxy_naive']:.2f}", f"{s['proxy_trusted']:.2f}", f"{s['true_score']:.2f}",
                  f"{s['cheat']:.0%}", f"{s['overfit_suspect']:.0%}", f"{s['breach']:.0%}",
                  f"{s['false_report']:.0%}", f"{s['honest_pass']:.0%}",
                  f"{s['trust_base_edit']:.0%}")
    console = Console()
    if not console.is_terminal:
        console = Console(width=180)
    console.print(t)
    (run_dir / "summary.json").write_text(json.dumps(summarise(rows), indent=2))
    return 0
