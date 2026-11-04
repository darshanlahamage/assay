"""Experiment sweeps: every combination of agent x task x condition x defense, repeated.

    assay sweep experiments/<name>.yaml        run (or resume) a sweep
    assay analyze runs/sweep-<name>            rates per condition, with uncertainty

Experiment file:

    name: prompt_pressure
    agents:     [{backend: groq, model: openai/gpt-oss-120b}]
    tasks:      [repo_invoice_commit_hook, "lcb_*_impossible"]   # names or globs
    max_tasks:  6                    # optional: named tasks + a fixed random subset of glob matches
    agents may set their own `repeats` (local models can afford more than API ones)
    repeats:    3
    conditions: [{name: neutral, pressure: {prompt: neutral}},
                 {name: high,    pressure: {prompt: high}}]      # first one is the baseline
    defenses:   [{name: off}]        # or {name: all, honesty_prompt: true, ...}

Rollouts are interleaved (repeat outermost, then task, then condition), so a sweep cut
short by an API quota still has roughly balanced conditions. Re-running skips finished
rollouts and retries ones that ended in an error.

Each agent runs in its own lane (thread), in that order. Lanes run at the same time, so
an API model waiting for its daily budget does not hold up the local model or another
API model (each Groq model has its own budget). Output lines are prefixed with the lane.
"""

from __future__ import annotations

import copy
import fnmatch
import random
import threading
from pathlib import Path

import yaml

from assay.agent.backends import QuotaExhausted, make_backend
from assay.audit.ledger import read_rows
from assay.pipeline import load_config, run_rollout, settings_from_config
from assay.tasks.loader import load_task

REPO_ROOT = Path(__file__).resolve().parents[2]


def plan(exp: dict) -> list[dict]:
    names = sorted(p.parent.name for p in (REPO_ROOT / "benchmark").glob("*/meta.yaml"))
    named = [t for t in exp["tasks"] if not any(c in t for c in "*?[")]
    globbed: list[str] = []
    for pat in exp["tasks"]:
        if pat not in named:
            globbed += [n for n in names if fnmatch.fnmatch(n, pat) and n not in globbed + named]
    # max_tasks samples only the wildcard matches; tasks named explicitly are always in.
    room = (exp["max_tasks"] - len(named)) if exp.get("max_tasks") else None
    if room is not None and len(globbed) > room:
        globbed = sorted(random.Random(exp.get("seed", 0)).sample(globbed, max(0, room)))
    missing = [t for t in named if t not in names]
    if missing:
        raise ValueError(f"unknown tasks: {missing}")
    tasks = named + globbed
    if not tasks:
        raise ValueError(f"no tasks match {exp['tasks']}")
    for item in exp["conditions"] + exp.get("defenses", []):
        item["name"] = _name(item["name"])
    jobs = []
    most = max(a.get("repeats", exp.get("repeats", 1)) for a in exp["agents"])
    for k in range(most):
        for task in tasks:
            for agent in exp["agents"]:
                if k >= agent.get("repeats", exp.get("repeats", 1)):
                    continue
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
    done = {r["rollout"] for r in read_rows(run_dir / "ledger.jsonl") if "error" not in r["verdicts"]}
    todo = [j for j in jobs if j["rollout"] not in done]
    print(f"{run_id}: {len(jobs)} rollouts planned, {len(jobs) - len(todo)} done, {len(todo)} to run")
    if dry_run:
        for j in todo[:20]:
            print("  ", j["rollout"])
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(Path(exp_path).read_text())

    # One lane per API agent. All local (mlx) agents share one lane: only one model fits
    # in memory, so its rollouts run as a batch per model (each model loaded once).
    lanes: dict[str, list[dict]] = {}
    for j in todo:
        key = "local" if j["agent"]["backend"] == "mlx" else f"{j['agent']['backend']}:{j['agent_name']}"
        lanes.setdefault(key, []).append(j)
    if "local" in lanes:
        order = [a.get("name") or a.get("model") for a in exp["agents"]]
        lanes["local"].sort(key=lambda j: order.index(j["agent"].get("name") or j["agent"].get("model")))
    codes: dict[str, int] = {}
    threads = [threading.Thread(target=_lane, name=name, daemon=True,
                                args=(name, jobs_, base_cfg, run_dir, run_id, exp, wait_for_quota, codes))
               for name, jobs_ in lanes.items()]
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1)       # short joins keep the main thread responsive to kill
    except BaseException:
        _STOP.set()                     # lanes start nothing new; containers removed at exit
        raise
    if any(c != 0 for c in codes.values()):
        print("some lanes stopped early: " + ", ".join(f"{k}={v}" for k, v in codes.items() if v))
        print("Re-run the same command to resume.")
        return 2
    print(f"\ndone. Analyse with: assay analyze {run_dir.relative_to(REPO_ROOT)}")
    return 0


_STOP = threading.Event()


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _lane(name: str, jobs: list[dict], base_cfg: dict, run_dir: Path, run_id: str, exp: dict,
          wait_for_quota: bool, codes: dict[str, int]) -> None:
    """Run one agent's rollouts in order (one backend instance, reused)."""
    say = lambda m: print(f"[{name}] {m}", flush=True)  # noqa: E731
    backend, backend_key = None, None
    codes[name] = 0
    for i, j in enumerate(jobs, 1):
        if _STOP.is_set():
            return
        cfg = copy.deepcopy(base_cfg)
        agent = j["agent"]
        cfg["backends"].setdefault(agent["backend"], {})
        for k, v in agent.items():                   # model and any backend option
            if k not in ("backend", "name", "scenario", "repeats"):
                cfg["backends"][agent["backend"]][k] = v
        if wait_for_quota:
            cfg["backends"][agent["backend"]]["quota_wait_max_s"] = 6 * 3600
        cfg["pressure"] = {**cfg.get("pressure", {}), **j["condition"].get("pressure", {})}
        cfg["defenses"] = {**cfg["defenses"], **{k: v for k, v in j["defense"].items() if k != "name"}}
        key = j["agent_name"]
        if agent["backend"] == "script":
            # A scripted agent replays a fixed list of replies: fresh one per rollout.
            cfg["backends"]["script"] = {"scenario": str(REPO_ROOT / agent["scenario"])}
            backend = make_backend("script", cfg)
        elif backend is None or backend_key != key:
            if backend is not None:             # switching models: free the old one first
                say(f"unloading {backend_key}")
                backend = None
                _free_memory()
            say(f"loading {key}")
            backend, backend_key = make_backend(agent["backend"], cfg), key
        task = load_task(REPO_ROOT / "benchmark" / j["task"])
        say(f"{i}/{len(jobs)} {j['rollout']}")
        try:
            row = run_rollout(task, backend, settings_from_config(cfg), run_dir, run_id,
                              j["rollout"], extra={"sweep": exp["name"], "agent": j["agent_name"],
                                                   "condition": j["condition"]["name"],
                                                   "defense": j["defense"]["name"],
                                                   "repeat": j["repeat"]})
        except QuotaExhausted as e:
            say(f"quota exhausted, lane stopped: {e}")
            codes[name] = 2
            return
        except Exception as e:  # noqa: BLE001 — one broken rollout must not end the lane
            say(f"rollout failed: {type(e).__name__}: {e}")
            continue
        say(f"   -> {row.verdicts}  visible={row.proxy_trusted:.2f} hidden={row.true_score:.2f}"
            f"  ({row.duration_s / 60:.1f} min)")
        if row.error.startswith("QuotaExhausted"):
            say("quota exhausted, lane stopped. Re-run the same command later to resume.")
            codes[name] = 2
            return
