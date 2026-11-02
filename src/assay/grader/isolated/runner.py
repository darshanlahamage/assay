"""Trusted test runner for separate-process grading (DESIGN.md §5.2). Runs inside the
grader container as root. Standalone: no assay imports.

The tests and pytest run here. The agent's modules are replaced by stand-ins whose
functions forward each call to a child process (child.py) running as the unprivileged
agent user. The child imports the agent's code, calls the function and sends back the
return value as plain JSON. Comparisons, pass/fail and the report all happen here, so
agent code cannot exit the test run, patch pytest, override ==, or write the report.

    python runner.py --repo /repo --modules configstore --report /grader/out/junit.xml \
        --child /opt/assay/child.py -- tests/
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import secrets
import select
import subprocess
import sys
import types

AGENT_UID = 1000
CALL_TIMEOUT_S = 10


class AgentCodeError(Exception):
    """The agent's code failed in a way that has no matching built-in exception."""


class ChildFailure(AgentCodeError):
    """The agent-code process died, hung, or sent something invalid."""


# -- codec (kept identical in child.py) ----------------------------------------------
def encode(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if type(v).__name__ == "Decimal" and type(v).__module__ in ("decimal", "_decimal"):
        return {"__t": "decimal", "v": str(v)}
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, tuple):
        return {"__t": "tuple", "v": [encode(x) for x in v]}
    if isinstance(v, set):
        return {"__t": "set", "v": [encode(x) for x in v]}
    if isinstance(v, dict):
        return {"__t": "dict", "v": [[encode(k), encode(x)] for k, x in v.items()]}
    raise TypeError(f"cannot send value of type {type(v).__name__} between processes")


def decode(v):
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        t, items = v.get("__t"), v.get("v")
        if t == "tuple":
            return tuple(decode(x) for x in items)
        if t == "set":
            return {decode(x) for x in items}
        if t == "decimal":
            import decimal
            return decimal.Decimal(items)
        if t == "dict":
            return {decode(k): decode(x) for k, x in items}
        raise ValueError("malformed value")
    return v


# -- the agent-code process ------------------------------------------------------------
class Worker:
    def __init__(self, repo: str, child_script: str) -> None:
        self.repo, self.child_script = repo, child_script
        self.proc: subprocess.Popen | None = None

    def _start(self) -> None:
        to_child_r, to_child_w = os.pipe()
        from_child_r, from_child_w = os.pipe()
        self.proc = subprocess.Popen(
            [sys.executable, "-I", self.child_script, str(to_child_r), str(from_child_w),
             self.repo],
            cwd=self.repo, pass_fds=(to_child_r, from_child_w),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            user=AGENT_UID, group=AGENT_UID, extra_groups=[],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/agent",
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        os.close(to_child_r)
        os.close(from_child_w)
        self.wf = os.fdopen(to_child_w, "w")
        self.rf = os.fdopen(from_child_r, "r")

    def _kill(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def call(self, module: str, func: str, args: tuple, kwargs: dict):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        req_id = secrets.token_hex(8)
        try:
            msg = json.dumps({"id": req_id, "module": module, "func": func,
                              "args": encode(list(args)), "kwargs": encode(kwargs)})
        except TypeError as e:
            raise ChildFailure(f"arguments can't be sent to the agent's code: {e}") from None
        try:
            self.wf.write(msg + "\n")
            self.wf.flush()
        except (BrokenPipeError, OSError):
            self._kill()
            raise ChildFailure("agent-code process exited") from None
        ready, _, _ = select.select([self.rf], [], [], CALL_TIMEOUT_S)
        if not ready:
            self._kill()
            raise ChildFailure(f"call to {module}.{func} timed out")
        line = self.rf.readline()
        if not line:
            self._kill()
            raise ChildFailure("agent-code process exited during the call")
        try:
            resp = json.loads(line)
            assert resp["id"] == req_id
        except Exception:  # noqa: BLE001
            self._kill()
            raise ChildFailure("invalid response from agent-code process") from None
        if resp["ok"]:
            return decode(resp["value"])
        raise _rebuild_exception(resp.get("error_type", ""), resp.get("message", ""))

    def close(self) -> None:
        if self.proc is None:
            return
        for f in (self.wf, self.rf):
            f.close()
        try:
            self.proc.wait(timeout=2)   # EOF on its input: the child exits on its own
            self.proc = None
        except subprocess.TimeoutExpired:
            self._kill()


def _rebuild_exception(name: str, message: str) -> Exception:
    """Re-raise built-in exceptions by type so `pytest.raises(KeyError)` still works."""
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(message)
        except Exception:  # noqa: BLE001
            pass
    return AgentCodeError(f"{name}: {message}")


class StandIn(types.ModuleType):
    """Replaces an agent module: every public attribute is a function forwarded to the child."""

    def __init__(self, name: str, worker: Worker) -> None:
        super().__init__(name)
        self._worker = worker

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        name, worker = self.__name__, self._worker

        def forward(*args, **kwargs):
            return worker.call(name, attr, args, kwargs)

        forward.__name__ = attr
        return forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--modules", required=True, help="comma-separated agent module names")
    ap.add_argument("--report", required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("tests", nargs="+")
    a = ap.parse_args()

    # Nothing from the agent's repo may be importable in this process.
    sys.path = [p for p in sys.path if not p.startswith(a.repo)]
    worker = Worker(a.repo, a.child)
    for m in a.modules.split(","):
        sys.modules[m.strip()] = StandIn(m.strip(), worker)

    import pytest

    try:
        return int(pytest.main([*a.tests, "-q", "-p", "no:cacheprovider",
                                f"--junit-xml={a.report}"]))
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
