"""Experiment sweeps: every combination of agent x task x condition x defense, repeated.

    assay sweep experiments/<name>.yaml        run (or resume) a sweep
    assay analyze runs/sweep-<name>            rates per condition, with uncertainty

Experiment file:

    name: prompt_pressure
    agents:     [{backend: groq, model: openai/gpt-oss-120b}]
    tasks:      [repo_invoice_commit_hook, "lcb_*_impossible"]   # names or globs
    max_tasks:  6                    # optional: a fixed random subset of the matches
    repeats:    3
    conditions: [{name: neutral, pressure: {prompt: neutral}},
                 {name: high,    pressure: {prompt: high}}]      # first one is the baseline
    defenses:   [{name: off}]        # or {name: all, honesty_prompt: true, ...}

Rollouts are interleaved (repeat outermost, then task, then condition), so a sweep cut
short by an API quota still has roughly balanced conditions. Re-running skips finished
rollouts and retries ones that ended in an error.
"""

from __future__ import annotations

import copy
import fnmatch
import random
from pathlib import Path

import yaml

from assay.agent.backends import QuotaExhausted, make_backend
from assay.audit.ledger import read_rows
from assay.pipeline import load_config, run_rollout, settings_from_config
from assay.tasks.loader import load_task

REPO_ROOT = Path(__file__).resolve().parents[2]


def plan(exp: dict) -> list[dict]:
    names = sorted(p.parent.name for p in (REPO_ROOT / "benchmark").glob("*/meta.yaml"))
    tasks: list[str] = []
    for pat in exp["tasks"]:
        tasks += [n for n in names if fnmatch.fnmatch(n, pat) and n not in tasks]
    if exp.get("max_tasks") and len(tasks) > exp["max_tasks"]:
        tasks = sorted(random.Random(exp.get("seed", 0)).sample(tasks, exp["max_tasks"]))
    if not tasks:
        raise ValueError(f"no tasks match {exp['tasks']}")
    for item in exp["conditions"] + exp.get("defenses", []):
        item["name"] = _name(item["name"])
    jobs = []
    for k in range(exp.get("repeats", 1)):
        for task in tasks:
            for agent in exp["agents"]:
                for d in exp.get("defenses") or [{"name": "off"}]:
                    for c in exp["conditions"]:
                        a = agent.get("name") or agent.get("model", agent["backend"]).split("/")[-1]
                        jobs.append({"task": task, "agent": agent, "agent_name": a,
                                     "condition": c, "defense": d, "repeat": k,
                                     "rollout": f"{task}/{a}__{c['name']}__{d['name']}__r{k:02d}"})
    return jobs


def _name(v) -> str:
    """YAML reads unquoted off/on/yes/no as booleans; names are always text."""
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def run_sweep(exp_path: str, config_path: str, dry_run: bool = False,
              wait_for_quota: bool = False) -> int:
    exp = yaml.safe_load(Path(exp_path).read_text())
    base_cfg = load_config(config_path)
    jobs = plan(exp)
    run_id = f"sweep-{exp['name']}"
    run_dir = REPO_ROOT / base_cfg["run"]["out_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(Path(exp_path).read_text())

    done = {r["rollout"] for r in read_rows(run_dir / "ledger.jsonl") if "error" not in r["verdicts"]}
    todo = [j for j in jobs if j["rollout"] not in done]
    print(f"{run_id}: {len(jobs)} rollouts planned, {len(jobs) - len(todo)} done, {len(todo)} to run")
    if dry_run:
        for j in todo[:20]:
            print("  ", j["rollout"])
        return 0

    backends: dict[str, object] = {}
    for i, j in enumerate(todo, 1):
        cfg = copy.deepcopy(base_cfg)
        agent = j["agent"]
        cfg["backends"].setdefault(agent["backend"], {})
        if wait_for_quota:
            cfg["backends"][agent["backend"]]["quota_wait_max_s"] = 6 * 3600
        if agent.get("model"):
            cfg["backends"][agent["backend"]]["model"] = agent["model"]
        cfg["pressure"] = {**cfg.get("pressure", {}), **j["condition"].get("pressure", {})}
        cfg["defenses"] = {**cfg["defenses"], **{k: v for k, v in j["defense"].items() if k != "name"}}
        key = f"{agent['backend']}:{agent.get('model', '')}"
        if agent["backend"] == "script":
            # A scripted agent replays a fixed list of replies: fresh one per rollout.
            cfg["backends"]["script"] = {"scenario": str(REPO_ROOT / agent["scenario"])}
            backends[key] = make_backend("script", cfg)
        elif key not in backends:
            backends[key] = make_backend(agent["backend"], cfg)
        task = load_task(REPO_ROOT / "benchmark" / j["task"])
        print(f"[{i}/{len(todo)}] {j['rollout']}", flush=True)
        try:
            row = run_rollout(task, backends[key], settings_from_config(cfg), run_dir, run_id,
                              j["rollout"], extra={"sweep": exp["name"], "agent": j["agent_name"],
                                                   "condition": j["condition"]["name"],
                                                   "defense": j["defense"]["name"],
                                                   "repeat": j["repeat"]})
        except QuotaExhausted as e:
            print(f"quota exhausted, stopping: {e}")
            return 2
        print(f"      -> {row.verdicts}  visible={row.proxy_trusted:.2f} hidden={row.true_score:.2f}")
        if row.error.startswith("QuotaExhausted"):
            print("quota exhausted, stopping. Re-run the same command later to resume.")
            return 2
    print(f"\ndone. Analyse with: assay analyze {run_dir.relative_to(REPO_ROOT)}")
    return 0
