"""The agent loop. Runs on the host; every command runs inside the agent container.

    model reply -> parse one action -> run it in the container -> show the output -> repeat

Ends on `submit`, the step limit, or a backend error. After each command the
harness records which files that command changed, so any later verdict can be
traced to the exact step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from assay.agent import protocol
from assay.agent.backends import Backend
from assay.agent.trajectory import Step, Trajectory
from assay.sandbox.container import Container

# Root-owned marker: the agent user cannot touch it, so "newer than marker" is reliable.
_MARK = "/var/lib/assay_step_mark"
_TOUCHED = (
    f"cd /repo && find . -type f -newer {_MARK} "
    "-not -path './.git/objects/*' -not -path './.git/logs/*' -not -name '*.pyc' "
    "-not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' | sed 's|^\\./||' | sort; "
    f"touch {_MARK}"
)


@dataclass
class AgentConfig:
    max_steps: int = 30
    command_timeout_s: int = 60
    observation_chars: int = 6000
    show_budget: bool = False       # append "[steps remaining: k]" after every step
    # Scheduled messages: {"at_step": k, "from": "...", "message": "..."} is shown to the
    # agent together with the output of step k-1 (so it is read before step k).
    events: list[dict] = field(default_factory=list)


class AgentLoop:
    def __init__(self, backend: Backend, config: AgentConfig) -> None:
        self.backend = backend
        self.config = config

    def run(self, container: Container, task_id: str, system: str, task_text: str,
            on_step=None, live_path: Path | None = None) -> Trajectory:
        """`live_path`: rewrite the trajectory there after every step, so a running
        rollout can be watched (assay show / the UI)."""
        traj = Trajectory(task_id=task_id, backend=self.backend.name, model=self.backend.model)
        messages = [{"role": "user", "content": task_text}]
        container.exec(f"touch {_MARK}", user="root")

        for i in range(self.config.max_steps):
            t0 = time.monotonic()
            try:
                c = self.backend.complete(system, messages)
            except Exception as e:  # noqa: BLE001 — record and stop; the rollout still gets graded
                traj.stop_reason, traj.error = "error", f"{type(e).__name__}: {e}"
                return traj
            action = protocol.parse(c.text)
            messages.append({"role": "assistant", "content": c.text})
            step = Step(index=i, kind=action.kind, reply=c.text, reasoning=action.reasoning,
                        input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                        logprobs=c.logprobs, entropies=c.entropies)

            if action.kind == "submit":
                traj.submitted, traj.report = True, action.content
                step.command = action.content
                step.duration_s = time.monotonic() - t0
                traj.steps.append(step)
                traj.stop_reason = "submitted"
                _save(traj, live_path)
                _notify(on_step, step)
                return traj

            if action.kind == "format_error":
                observation = protocol.FORMAT_ERROR
            else:
                r = container.exec(action.content, timeout_s=self.config.command_timeout_s)
                step.command, step.output = action.content, r.output
                step.returncode, step.timed_out = r.returncode, r.timed_out
                touched = container.exec(_TOUCHED, user="root").output
                step.touched_files = [ln for ln in touched.splitlines() if ln.strip()]
                observation = protocol.format_observation(
                    r.returncode, r.output, r.timed_out, self.config.observation_chars)
                if action.extra_blocks:
                    observation = protocol.ONLY_FIRST + "\n" + observation

            for ev in self.config.events:
                if ev["at_step"] == i + 1:
                    note = f"[Message from {ev.get('from', 'your manager')}]: {ev['message']}"
                    step.notes.append(note)
                    observation += "\n\n" + note
            if self.config.show_budget:
                note = f"[steps remaining: {self.config.max_steps - i - 1}]"
                step.notes.append(note)
                observation += "\n\n" + note
            step.duration_s = time.monotonic() - t0
            traj.steps.append(step)
            _save(traj, live_path)
            _notify(on_step, step)
            messages.append({"role": "user", "content": observation})

        traj.stop_reason = "step_limit"
        return traj


def _notify(cb, step: Step) -> None:
    if cb is not None:
        cb(step)


def _save(traj: Trajectory, path: Path | None) -> None:
    if path is not None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(traj.to_dict(), indent=2))
        tmp.replace(path)
