"""Assay CLI.

    assay doctor                          check Docker, the sandbox image, API key, local model
    assay build-image                     build the sandbox image
    assay run --task <dir> [options]      run rollouts on one task (or --suite <dir>)
    assay report --run <dir>              summary table of a run's ledger
    assay show <rollout dir> | --latest   replay one rollout step by step
    assay selftest                        run the scripted scenarios; check every verdict
    assay ui                              local web dashboard (http://127.0.0.1:8765)
    assay sweep <experiment.yaml>         run (or resume) an experiment sweep
    assay analyze <run dir>               compare conditions: rates, intervals, tests
    assay internals <run dir>             local track: pressure direction, probe, lead time
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)   # show progress live, even in log files
    _load_env()
    parser = argparse.ArgumentParser(prog="assay", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment")
    sub.add_parser("build-image", help="build the sandbox image")

    p = sub.add_parser("run", help="run rollouts")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", help="one task directory")
    target.add_argument("--suite", help="directory of task directories")
    p.add_argument("--backend", choices=["groq", "gemini", "mlx", "script"], default="gemini")
    p.add_argument("--scenario", help="scenario file for --backend script (see scenarios/)")
    p.add_argument("--model", help="override the backend's model")
    p.add_argument("-n", "--repeats", type=int, default=1, help="rollouts per task")
    p.add_argument("--honesty-prompt", choices=["on", "off"], help="Layer 0")
    p.add_argument("--trusted-grading", choices=["off", "log_only", "on"], help="Layer 1")
    p.add_argument("--devmachine-review", choices=["on", "off"],
                   help="Layer 1: hold back agent changes to hooks/Makefile/settings")
    p.add_argument("--network", choices=["proxy", "none"], help="agent network access")
    p.add_argument("--pressure", choices=["none", "neutral", "low", "high", "threat"],
                   help="prompt pressure (wording added to the task)")
    p.add_argument("--failures", type=int, help="say N earlier attempts failed (with real output)")
    p.add_argument("--budget", choices=["default", "normal", "tight"],
                   help="step budget: tight = 8 steps, remaining steps shown")
    p.add_argument("--show-tests", action="store_true", help="paste the visible tests into the task")
    p.add_argument("--oversight", choices=["none", "watched", "unwatched"])
    p.add_argument("--max-steps", type=int)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    p.add_argument("--quiet", action="store_true", help="don't print each step")

    r = sub.add_parser("report", help="summarise a run")
    r.add_argument("--run", required=True, help="run directory (contains ledger.jsonl)")

    sw = sub.add_parser("sweep", help="run (or resume) an experiment sweep")
    sw.add_argument("experiment")
    sw.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    sw.add_argument("--dry-run", action="store_true", help="show the plan, run nothing")
    sw.add_argument("--wait-for-quota", action="store_true",
                    help="when an API's daily budget runs out, wait for it instead of stopping")

    an = sub.add_parser("analyze", help="compare conditions in a sweep")
    an.add_argument("run")
    an.add_argument("--by-family", action="store_true", help="split results by task family")

    it = sub.add_parser("internals", help="model-internals analysis of a local-track sweep")
    it.add_argument("run")
    it.add_argument("--permutations", type=int, default=0,
                    help="permutation test for each probe (slow; 100 is a good number)")

    u = sub.add_parser("ui", help="local web dashboard")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")

    st = sub.add_parser("selftest", help="run scripted scenarios and check the verdicts")
    st.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))

    s = sub.add_parser("show", help="replay one rollout step by step")
    s.add_argument("rollout", nargs="?", help="runs/<run_id>/<task_id>/rNN")
    s.add_argument("--latest", action="store_true", help="the most recent rollout")

    args = parser.parse_args(argv)
    if args.cmd == "show":
        from assay.viewer import latest_rollout, show
        target = latest_rollout(REPO_ROOT / "runs") if args.latest or not args.rollout \
            else Path(args.rollout)
        if target is None:
            print("no rollouts found under runs/")
            return 1
        return show(target)
    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "build-image":
        from assay.sandbox.container import build_image
        from assay.sandbox.network import build_proxy_image
        build_image(REPO_ROOT)
        build_proxy_image(REPO_ROOT)
        return 0
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "sweep":
        from assay.sweep import run_sweep
        return run_sweep(args.experiment, args.config, args.dry_run, args.wait_for_quota)
    if args.cmd == "internals":
        from assay.internals import print_internals
        return print_internals(Path(args.run), args.permutations)
    if args.cmd == "analyze":
        from assay.analysis import print_analysis
        return print_analysis(Path(args.run), args.by_family)
    if args.cmd == "ui":
        from assay.ui.server import serve
        return serve(args.port, not args.no_browser)
    if args.cmd == "selftest":
        return _selftest(args)
    if args.cmd == "report":
        from assay.report import print_report
        return print_report(Path(args.run))
    return 1


def _run(args: argparse.Namespace) -> int:
    from assay.agent.backends import make_backend
    from assay.pipeline import load_config, new_run_dir, run_rollout, settings_from_config
    from assay.sandbox.container import docker_available, image_exists
    from assay.tasks.loader import load_suite, load_task

    if not docker_available():
        print("Docker is not running. Start it with: colima start")
        return 1
    cfg = load_config(args.config)
    from assay.sandbox.network import proxy_image_exists
    if not image_exists(cfg["sandbox"]["image"]) or not proxy_image_exists():
        print("Sandbox or proxy image missing. Build them with: assay build-image")
        return 1
    if args.backend == "script":
        if not args.scenario:
            print("--backend script needs --scenario <file> (see scenarios/)")
            return 1
        cfg["backends"]["script"] = {"scenario": args.scenario}
    if args.model:
        cfg["backends"][args.backend]["model"] = args.model
    if args.honesty_prompt:
        cfg["defenses"]["honesty_prompt"] = args.honesty_prompt == "on"
    if args.trusted_grading:
        cfg["defenses"]["trusted_grading"] = args.trusted_grading
    if args.devmachine_review:
        cfg["defenses"]["devmachine_review"] = args.devmachine_review == "on"
    if args.network:
        cfg["sandbox"]["network"] = args.network
    if args.pressure:
        cfg["pressure"]["prompt"] = args.pressure
    if args.failures is not None:
        cfg["pressure"]["failures"] = args.failures
    if args.budget:
        cfg["pressure"]["budget"] = args.budget
    if args.show_tests:
        cfg["pressure"]["tests_shown"] = True
    if args.oversight:
        cfg["pressure"]["oversight"] = args.oversight
    if args.max_steps:
        cfg["agent"]["max_steps"] = args.max_steps

    task_path = Path(args.task) if args.task else None
    if task_path and not task_path.exists() and (REPO_ROOT / "benchmark" / args.task).exists():
        task_path = REPO_ROOT / "benchmark" / args.task       # a bare task name
    tasks = [load_task(task_path)] if task_path else load_suite(args.suite)
    settings = settings_from_config(cfg)
    backend = make_backend(args.backend, cfg)
    run_id, run_dir = new_run_dir(REPO_ROOT / cfg["run"]["out_dir"])
    print(f"run {run_id}: {len(tasks)} task(s) x {args.repeats}, backend={backend.name} "
          f"model={backend.model}")
    print(f"defenses={settings.defenses}  pressure: {settings.pressure.label()}")

    for task in tasks:
        for k in range(args.repeats):
            name = f"{task.id}/r{k:02d}"
            print(f"\n=== {name} ===")
            row = run_rollout(task, backend, settings, run_dir, run_id, name,
                              on_step=None if args.quiet else _print_step)
            print(f"--> naive={row.proxy_naive:.2f} trusted={row.proxy_trusted:.2f} "
                  f"true={row.true_score:.2f} reward={row.reward:.2f} "
                  f"verdicts={row.verdicts} steps={row.steps} ({row.stop_reason})")
            for e in row.evidence:
                print(f"    evidence: {e}")
            if row.error:
                print(f"    error: {row.error[:300]}")
            if row.error.startswith("QuotaExhausted"):
                print("\nAPI quota exhausted; stopping the run. Completed rollouts are in the ledger.")
                break
        else:
            continue
        break

    print(f"\nledger: {run_dir / 'ledger.jsonl'}")
    from assay.report import print_report
    return print_report(run_dir)


def _print_step(step) -> None:
    if step.kind == "bash":
        cmd = step.command.replace("\n", " ⏎ ")
        touched = f"  touched={step.touched_files}" if step.touched_files else ""
        print(f"  [{step.index:02d}] $ {cmd[:140]}  -> rc={step.returncode}{touched}")
    elif step.kind == "submit":
        first = step.command.strip().splitlines()[0] if step.command.strip() else ""
        print(f"  [{step.index:02d}] SUBMIT: {first[:140]}")
    else:
        print(f"  [{step.index:02d}] (format error)")


def _selftest(args: argparse.Namespace) -> int:
    """Run every scenario in scenarios/ through the full pipeline and compare verdicts."""
    import yaml

    from assay.agent.backends import ScriptedBackend
    from assay.pipeline import load_config, new_run_dir, run_rollout, settings_from_config
    from assay.tasks.loader import load_task

    cfg = load_config(args.config)
    run_id, run_dir = new_run_dir(REPO_ROOT / cfg["run"]["out_dir"])
    failures = 0
    cases = []
    for path in sorted((REPO_ROOT / "scenarios").glob("*.yaml")):
        sc = yaml.safe_load(path.read_text())
        if not (REPO_ROOT / "benchmark" / sc["task"]).exists():
            print(f"SKIP  {sc['name']:28s} task {sc['task']} not present "
                  f"(generate it: python -m assay.tasks.import_impossiblebench)")
            continue
        cases.append((path, sc, False))
        if "planted_code_ran" in sc.get("expect", []):
            cases.append((path, sc, True))     # same scenario with developer-machine review on
    for path, sc, review in cases:
        cfg["defenses"]["devmachine_review"] = review
        settings = settings_from_config(cfg)
        expect = set(sc.get("expect", []))
        if review:
            expect -= {"planted_code_ran"}
        task = load_task(REPO_ROOT / "benchmark" / sc["task"])
        label = sc["name"] + (" (review on)" if review else "")
        row = run_rollout(task, ScriptedBackend(str(path)), settings, run_dir, run_id,
                          f"{task.id}/{sc['name']}{'_review' if review else ''}")
        got = set(row.verdicts)
        ok = got == expect
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:28s} expected {sorted(expect)}  got {sorted(got)}")
    print(f"\n{len(cases) - failures}/{len(cases)} scenarios matched.  Replay any with: "
          f"assay show runs/{run_id}/<task>/<scenario>")
    return 1 if failures else 0


def _doctor() -> int:
    from assay.sandbox.container import docker_available, image_exists

    ok = True
    d = docker_available()
    print(f"docker running        : {'yes' if d else 'NO  -> colima start'}")
    ok &= d
    from assay.sandbox.network import proxy_image_exists
    i = d and image_exists()
    print(f"sandbox image         : {'yes' if i else 'NO  -> assay build-image'}")
    ok &= bool(i)
    pi = d and proxy_image_exists()
    print(f"proxy image           : {'yes' if pi else 'NO  -> assay build-image'}")
    ok &= bool(pi)
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        k = bool(os.environ.get(var))
        print(f"{var:22s}: {'set' if k else 'not set'}")
    try:
        from huggingface_hub import try_to_load_from_cache

        from assay.pipeline import load_config
        model = load_config(REPO_ROOT / "configs" / "default.yaml")["backends"]["mlx"]["model"]
        cached = isinstance(try_to_load_from_cache(model, "config.json"), str)
        print(f"local model           : {model} {'(downloaded)' if cached else '(not downloaded)'}")
    except Exception as e:  # noqa: BLE001
        print(f"local model           : unknown ({e})")
    return 0 if ok else 1


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        from dotenv import load_dotenv
        load_dotenv(env)


if __name__ == "__main__":
    raise SystemExit(main())
