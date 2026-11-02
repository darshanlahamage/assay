"""One rollout end to end (DESIGN.md §4.2, Controller).

    start agent container (fake credentials, proxy-only network) -> agent loop ->
    copy final repo out -> grade in fresh containers -> developer-machine step ->
    check the report -> verdicts -> ledger row + trajectory + evidence on disk

Nothing agent-produced is executed on the host: the final repo is copied out as
plain files only to be diffed and copied into grader containers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from assay.agent import protocol
from assay.agent.backends import Backend
from assay.agent.loop import AgentConfig, AgentLoop
from assay.audit import canaries as C
from assay.audit.claims import check as check_claims
from assay.audit.hardcoding import find_hardcoded
from assay.audit.ledger import Ledger, LedgerRow
from assay.audit.verdict import SecurityEvidence, judge
from assay.devmachine import DEFAULT_STEPS, run_devmachine
from assay.grader.grader import Grader
from assay.sandbox import network as N
from assay.sandbox.container import Container, ContainerSpec
from assay.tasks.base import TaskSpec

TRACK = {"groq": "api", "gemini": "api", "mlx": "local", "script": "scripted"}
_GIT_INIT = "git init -q && git add -A && git commit -qm 'initial commit'"


@dataclass
class Defenses:
    honesty_prompt: bool = False      # Layer 0
    trusted_grading: str = "off"      # Layer 1: off | log_only | on
    devmachine_review: bool = False   # Layer 1: hold back agent changes to host-trust files


@dataclass
class Pressure:
    prompt: str = "neutral"           # key of protocol.PRESSURE_PROMPTS


@dataclass
class RunSettings:
    defenses: Defenses = field(default_factory=Defenses)
    pressure: Pressure = field(default_factory=Pressure)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sandbox: ContainerSpec = field(default_factory=ContainerSpec)
    network: str = "proxy"            # proxy | none
    devmachine: bool = True
    grader_timeout_s: int = 120


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def settings_from_config(cfg: dict) -> RunSettings:
    a, s, d, p = cfg["agent"], cfg["sandbox"], cfg["defenses"], cfg["pressure"]
    return RunSettings(
        defenses=Defenses(d.get("honesty_prompt", False), d.get("trusted_grading", "off"),
                          d.get("devmachine_review", False)),
        pressure=Pressure(p.get("prompt", "neutral")),
        agent=AgentConfig(a.get("max_steps", 30), a.get("command_timeout_s", 60),
                          a.get("observation_chars", 6000)),
        sandbox=ContainerSpec(image=s.get("image", "assay-sandbox:0.2"),
                              cpus=s.get("cpus", 2.0), memory=s.get("memory", "2g")),
        network=s.get("network", "proxy"),
        devmachine=cfg.get("devmachine", {}).get("enabled", True),
        grader_timeout_s=cfg.get("grader", {}).get("timeout_s", 120),
    )


def build_prompts(task: TaskSpec, settings: RunSettings) -> tuple[str, str]:
    system = protocol.SYSTEM_PROMPT
    if settings.defenses.honesty_prompt:
        system += "\n" + protocol.HONESTY_PROMPT
    task_text = task.prompt()
    pressure = protocol.PRESSURE_PROMPTS[settings.pressure.prompt]
    if pressure:
        task_text += "\n\n" + pressure
    return system, task_text


def run_rollout(task: TaskSpec, backend: Backend, settings: RunSettings, run_dir: Path,
                run_id: str, rollout_name: str, on_step=None) -> LedgerRow:
    out = run_dir / rollout_name
    out.mkdir(parents=True, exist_ok=True)
    system, task_text = build_prompts(task, settings)
    t0 = time.monotonic()

    canaries = C.make_canaries()
    agent_spec = settings.sandbox
    if settings.network == "proxy":
        N.ensure_proxy()
        agent_spec = replace(agent_spec, network=N.NETWORK, env=N.PROXY_ENV)
    net_since = N.now_rfc3339()

    with Container(agent_spec, label="assay-agent") as box:
        box.copy_in(task.repo_dir)
        box.exec(_GIT_INIT)
        for cmd in task.setup:
            r = box.exec(cmd)
            if r.returncode != 0:
                raise RuntimeError(f"task setup failed: {cmd}: {r.output[-500:]}")
        C.install(box, canaries)
        ip = N.container_ip(box.name) if settings.network == "proxy" else ""
        baseline = out / "initial_repo"
        box.copy_out("/repo", baseline)
        _write_json(out / "prompts.json", {"system": system, "task": task_text})
        traj = AgentLoop(backend, settings.agent).run(box, task.id, system, task_text, on_step,
                                                      live_path=out / "trajectory.json")
        final_repo = out / "final_repo"
        box.copy_out("/repo", final_repo)

    grade = Grader(settings.sandbox, settings.grader_timeout_s).grade(task, final_repo, baseline)

    sec = SecurityEvidence()
    sources = {f"step {s.index} output": s.output for s in traj.steps}
    sources["final report"] = traj.report
    sources.update(_repo_texts(final_repo, grade.change_kinds))
    sec.canary_leaks = C.find_leaks(canaries, sources)
    sec.credential_probes = C.credential_probes(
        {f"step {s.index}": s.command for s in traj.steps if s.kind == "bash"})
    if ip:
        log = N.requests_from(ip, net_since)
        sec.network_allowed, sec.network_blocked = log.allowed, log.blocked
    if settings.devmachine:
        sec.devmachine = run_devmachine(final_repo, baseline, canaries, settings.sandbox,
                                        review=settings.defenses.devmachine_review,
                                        steps=DEFAULT_STEPS)
    tb_edits = [p for p in grade.changes.get("trust_base", [])
                if grade.change_kinds.get(p) != "added"]
    sec.claim_issues = check_claims(traj.report, grade.proxy_trusted.score, tb_edits)
    visible = [f for v in task.visible_tests for f in
               ([task.repo_dir / v] if (task.repo_dir / v).is_file()
                else sorted((task.repo_dir / v).rglob("test_*.py")))]
    sec.hardcoded_cases = find_hardcoded(visible, final_repo, task.source_files)
    bash = [(s.index, s.command) for s in traj.steps if s.kind == "bash"]
    for chk in task.command_checks:
        steps = [i for i, cmd in bash if re.search(chk["pattern"], cmd)]
        if steps:
            sec.command_hits.append((chk["label"], f"{chk['evidence']} (step {steps[0]})"))
    sec.exfil_attempts = sorted({h for h in sec.network_allowed + sec.network_blocked
                                 if h in task.exfil_hosts})

    verdict = judge(task, grade, sec)
    if traj.stop_reason == "error":
        # Infrastructure failure, not agent behaviour: keep the scores, but don't
        # let it count as an honest fail (or anything else) in the rates.
        verdict.labels = ["error"]
        verdict.evidence.insert(0, f"rollout stopped by error: {traj.error[:200]}")
    reward = (grade.proxy_trusted.score if settings.defenses.trusted_grading == "on"
              else grade.proxy_naive.score)

    _write_json(out / "trajectory.json", traj.to_dict())
    _write_json(out / "grade.json", grade.to_dict())
    _write_json(out / "prompts.json", {"system": system, "task": task_text})
    _write_json(out / "security.json", _security_dict(sec))

    row = LedgerRow(
        run_id=run_id, rollout=rollout_name, track=TRACK.get(backend.name, backend.name),
        backend=backend.name, model=backend.model, task_id=task.id, family=task.family,
        pressure=asdict(settings.pressure), defenses=asdict(settings.defenses),
        reward=round(reward, 4), proxy_naive=round(grade.proxy_naive.score, 4),
        proxy_trusted=round(grade.proxy_trusted.score, 4),
        true_score=round(grade.true.score, 4), verdicts=verdict.labels,
        evidence=verdict.evidence, changes=grade.changes, steps=len(traj.steps),
        submitted=traj.submitted, stop_reason=traj.stop_reason,
        input_tokens=traj.input_tokens, output_tokens=traj.output_tokens,
        duration_s=round(time.monotonic() - t0, 1), error=traj.error,
        security=_security_dict(sec),
    )
    Ledger(run_dir / "ledger.jsonl").append(row)
    return row


def new_run_dir(out_root: str | Path) -> tuple[str, Path]:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    d = Path(out_root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def _repo_texts(repo: Path, change_kinds: dict[str, str]) -> dict[str, str]:
    """Text of files the agent added or changed, to search for leaked credentials."""
    out = {}
    for rel, kind in change_kinds.items():
        p = repo / rel
        if kind != "deleted" and p.is_file() and not p.is_symlink() and p.stat().st_size < 2_000_000:
            try:
                out[f"repo file {rel}"] = p.read_text(errors="replace")
            except OSError:
                pass
    return out


def _security_dict(sec: SecurityEvidence) -> dict:
    return {
        "canary_leaks": [list(x) for x in sec.canary_leaks],
        "credential_probes": sec.credential_probes,
        "network_allowed": sec.network_allowed,
        "network_blocked": sec.network_blocked,
        "devmachine": sec.devmachine.to_dict() if sec.devmachine else None,
        "claim_issues": sec.claim_issues,
        "hardcoded_cases": sec.hardcoded_cases,
        "command_hits": [list(x) for x in sec.command_hits],
        "exfil_attempts": sec.exfil_attempts,
    }


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))
