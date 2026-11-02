"""A Docker container that agent commands and agent code run in.

Nothing the agent produces runs on the host. The controller talks to the container
only through the docker CLI: start it, copy files in and out, run commands, remove it.

Isolation per container:
  * no network (`--network none`) unless a network name is given (the proxy, later),
  * all Linux capabilities dropped except the three root needs to set up /repo,
  * no privilege escalation, a process limit, CPU and memory limits,
  * commands run as the unprivileged `agent` user, each with a hard timeout.
"""

from __future__ import annotations

import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "assay-sandbox:0.3"


class ContainerError(RuntimeError):
    pass


@dataclass
class ExecResult:
    returncode: int
    output: str          # stdout and stderr, interleaved
    timed_out: bool


@dataclass
class ContainerSpec:
    image: str = DEFAULT_IMAGE
    network: str = "none"
    cpus: float = 2.0
    memory: str = "2g"
    pids_limit: int = 512
    workdir: str = "/repo"
    # Root needs these three to set up /repo. Agent commands run as a normal user and
    # get no capabilities either way. The grader adds SETUID/SETGID to drop the
    # agent-code process to that user.
    caps: tuple[str, ...] = ("CHOWN", "DAC_OVERRIDE", "FOWNER")
    env: tuple[tuple[str, str], ...] = ()


class Container:
    def __init__(self, spec: ContainerSpec | None = None, label: str = "assay") -> None:
        self.spec = spec or ContainerSpec()
        self.name = f"{label}-{uuid.uuid4().hex[:10]}"
        self._started = False

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> Container:
        s = self.spec
        caps = [a for c in s.caps for a in ("--cap-add", c)]
        env = [a for k, v in s.env for a in ("-e", f"{k}={v}")]
        _docker(
            "run", "-d", "--name", self.name,
            "--network", s.network,
            "--cpus", str(s.cpus), "--memory", s.memory, "--pids-limit", str(s.pids_limit),
            "--cap-drop", "ALL", *caps, *env,
            "--security-opt", "no-new-privileges",
            "--workdir", s.workdir,
            s.image, "sleep", "infinity",
        )
        self._started = True
        return self

    def remove(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, check=False)
            self._started = False

    def __enter__(self) -> Container:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.remove()

    # -- files -----------------------------------------------------------------
    def copy_in(self, host_dir: Path, dest: str | None = None, owner: str = "agent") -> None:
        """Copy the contents of `host_dir` into `dest` and give them to `owner`."""
        dest = dest or self.spec.workdir
        self.exec(f"mkdir -p {shlex.quote(dest)}", user="root")
        _docker("cp", f"{Path(host_dir).resolve()}/.", f"{self.name}:{dest}")
        self.exec(f"chown -R {owner}:{owner} {shlex.quote(dest)}", user="root")

    def copy_out(self, src: str, host_dest: Path) -> None:
        """Copy the contents of `src` in the container into `host_dest`."""
        host_dest.mkdir(parents=True, exist_ok=True)
        _docker("cp", f"{self.name}:{src}/.", str(host_dest))

    def read_file(self, path: str) -> str | None:
        r = self.exec(f"cat {shlex.quote(path)}", user="root")
        return r.output if r.returncode == 0 else None

    # -- commands --------------------------------------------------------------
    def exec(self, cmd: str, timeout_s: int = 60, user: str = "agent",
             workdir: str | None = None) -> ExecResult:
        """Run `cmd` with bash inside the container.

        The timeout is enforced inside the container (coreutils `timeout`), so a
        runaway command is actually killed, not just abandoned by the client.
        """
        inner = f"timeout -k 2 {int(timeout_s)} bash -c {shlex.quote(cmd)}"
        try:
            cp = subprocess.run(
                ["docker", "exec", "-u", user, "-w", workdir or self.spec.workdir,
                 self.name, "bash", "-c", inner],
                capture_output=True, text=True, errors="replace",
                timeout=timeout_s + 15, check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(returncode=124, output="(timed out)", timed_out=True)
        out = cp.stdout + cp.stderr
        return ExecResult(returncode=cp.returncode, output=out, timed_out=cp.returncode == 124)


def _docker(*args: str) -> str:
    cp = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise ContainerError(f"docker {args[0]} failed: {cp.stderr.strip()}")
    return cp.stdout.strip()


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, check=False,
                              timeout=20).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True, check=False).returncode == 0


def build_image(repo_root: Path, image: str = DEFAULT_IMAGE) -> None:
    subprocess.run(
        ["docker", "build", "-f", str(repo_root / "docker" / "sandbox.Dockerfile"),
         "-t", image, str(repo_root / "docker")],
        check=True,
    )
