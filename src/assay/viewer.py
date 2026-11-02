"""Replay one rollout in readable form.

    assay show runs/<run_id>/<task_id>/r00
    assay show --latest

Shows, in order: the task text the agent was given, the files it started with, every
step (its explanation, the command, the output, the files that command changed), its
final report, the full diff of what it changed, and how each grader scored it.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from assay.grader.changes import snapshot


def latest_rollout(runs_root: Path) -> Path | None:
    dirs = sorted(runs_root.glob("*/*/r*/trajectory.json"), key=lambda p: p.stat().st_mtime)
    return dirs[-1].parent if dirs else None


def show(rollout: Path, max_output_lines: int = 25) -> int:
    con = Console()
    if not con.is_terminal:
        con = Console(width=140)
    traj = json.loads((rollout / "trajectory.json").read_text())
    grade = json.loads((rollout / "grade.json").read_text())
    prompts = json.loads((rollout / "prompts.json").read_text())
    row = _ledger_row(rollout)

    con.rule(f"[bold]{traj['task_id']}[/]  ·  {traj['model']}  ·  {rollout.name}")
    if row:
        con.print(f"defenses: {row['defenses']}   pressure: {row['pressure']}")

    con.print(Panel(prompts["task"].strip(), title="1. Task given to the agent", border_style="cyan"))

    initial = rollout / "initial_repo"
    if not initial.exists():  # runs from before the starting snapshot was recorded
        from assay.tasks.loader import load_task
        bench = Path(__file__).resolve().parents[2] / "benchmark" / traj["task_id"]
        initial = load_task(bench).repo_dir
    files = [p for p in snapshot(initial) if not p.startswith(".git/")]
    con.print(Panel("\n".join(files), title="2. Files the agent started with", border_style="cyan"))

    con.rule("3. What the agent did, step by step")
    for s in traj["steps"]:
        head = Text(f"step {s['index']}", style="bold")
        if s["reasoning"]:
            con.print(head, Text(" · " + _clip(s["reasoning"], 6), style="italic"))
        else:
            con.print(head)
        if s["kind"] == "bash":
            con.print(Syntax(s["command"], "bash", theme="ansi_dark", word_wrap=True))
            style = "green" if s["returncode"] == 0 else "red"
            con.print(Text(f"  exit {s['returncode']}" + (" (timed out)" if s["timed_out"] else ""),
                           style=style))
            if s["output"].strip():
                con.print(Text(_indent(_clip(s["output"], max_output_lines)), style="dim"))
            if s["touched_files"]:
                con.print(Text(f"  changed: {', '.join(s['touched_files'])}", style="yellow"))
        elif s["kind"] == "format_error":
            con.print(Text("  (reply had no valid command block)", style="red"))
        con.print()

    report = traj["report"] or f"(no report — stopped: {traj['stop_reason']} {traj['error'][:200]})"
    con.print(Panel(report.strip(), title="4. Agent's final report (a claim, not evidence)",
                    border_style="magenta"))

    con.rule("5. What actually changed (diff vs. starting files)")
    d = _diff(initial, rollout / "final_repo")
    con.print(Syntax(d, "diff", theme="ansi_dark", word_wrap=True) if d else "(no changes)")

    lines = []
    for name, label in (("proxy_naive", "naive   — visible tests on the agent's repo as left"),
                        ("proxy_trusted", "trusted — visible tests, original tests + agent's source only"),
                        ("true", "true    — hidden tests the agent never saw")):
        g = grade[name]
        counts = f"{g['passed']}/{g['total']} passed" if g["total"] else "no report"
        extra = f"   ⚠ {g['interference']}" if g.get("interference") else ""
        lines.append(f"{g['score']:.2f}  {label}  ({counts}){extra}")
    sec_path = rollout / "security.json"
    if sec_path.exists():
        sec = json.loads(sec_path.read_text())
        dm = sec.get("devmachine") or {}
        sl = [
            f"fake credentials seen in output/report/repo : {sec['canary_leaks'] or '-'}",
            f"commands that looked for credentials        : {sec['credential_probes'] or '-'}",
            f"network allowed by proxy                    : {sec['network_allowed'] or '-'}",
            f"network refused by proxy                    : {sec['network_blocked'] or '-'}",
        ]
        if dm:
            sl += ["",
                   f"developer machine ran: {dm['steps']}  (review {'on' if dm['review'] else 'off'})",
                   f"  repo files run as programs : {dm['hooks_run'] or '-'}",
                   f"  fake credentials opened    : {dm['canaries_opened'] or '-'}",
                   f"  connections attempted      : {dm['network_attempts'] or '-'}",
                   f"  held back for review       : {dm['blocked'] or '-'}"]
        sl += ["", f"report vs. reality: {sec['claim_issues'] or 'consistent'}",
               f"visible test cases written into source : {sec.get('hardcoded_cases') or '-'}",
               f"task-specific checks that fired         : {sec.get('command_hits') or '-'}",
               f"requests to exfiltration hosts         : {sec.get('exfil_attempts') or '-'}"]
        con.print(Panel("\n".join(sl), title="6. Security and honesty checks", border_style="red"))

    c = grade["changes"]
    lines.append("")
    lines.append(f"changed source files     : {c['source'] or '-'}")
    lines.append(f"changed trust-base files : {c['trust_base'] or '-'}")
    lines.append(f"other changed files      : {c['other'] or '-'}")
    if row:
        lines.append("")
        lines.append(f"VERDICT: {', '.join(row['verdicts'])}")
        lines += [f"  · {e}" for e in row["evidence"]]
    con.print(Panel("\n".join(lines), title="7. Grading and verdict", border_style="green"))
    return 0


def _ledger_row(rollout: Path) -> dict | None:
    ledger = rollout.parent.parent / "ledger.jsonl"
    name = f"{rollout.parent.name}/{rollout.name}"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            r = json.loads(line)
            if r["rollout"] == name:
                return r
    return None


def _diff(a: Path, b: Path) -> str:
    sa, sb = snapshot(a), snapshot(b)
    out: list[str] = []
    for rel in sorted(set(sa) | set(sb)):
        if rel.startswith(".git/") and not rel.startswith(".git/hooks/") or sa.get(rel) == sb.get(rel):
            continue
        if rel.startswith(".git/hooks/") and rel.endswith(".sample"):
            continue
        old = _read(a / rel) if rel in sa else []
        new = _read(b / rel) if rel in sb else []
        out += difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}")
    return "".join(out)


def _read(p: Path) -> list[str]:
    try:
        return p.read_text().splitlines(keepends=True)
    except UnicodeDecodeError:
        return ["(binary file)\n"]


def _clip(text: str, max_lines: int) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def _indent(text: str) -> str:
    return "\n".join("    " + ln for ln in text.splitlines())
