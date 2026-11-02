"""Network for agent containers: an internal Docker network plus an allow-list proxy.

    agent container ──(assay-internal: no route out)──► assay-proxy ──► pypi.org, ...

The agent can install packages and read Python docs, like in a real harness, but the
proxy is its only path out: it refuses every host not on docker/proxy/allowlist and
logs every request. Each rollout's requests are read back from the proxy log by the
agent container's IP address.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

NETWORK = "assay-internal"
PROXY_NAME = "assay-proxy"
PROXY_IMAGE = "assay-proxy:0.1"
PROXY_URL = f"http://{PROXY_NAME}:8888"
PROXY_ENV = (("HTTP_PROXY", PROXY_URL), ("HTTPS_PROXY", PROXY_URL),
             ("http_proxy", PROXY_URL), ("https_proxy", PROXY_URL),
             ("NO_PROXY", "localhost,127.0.0.1"))

_CONNECT = re.compile(r"Connect \(file descriptor (\d+)\): ([\d.]+)")
_REQUEST = re.compile(r"Request \(file descriptor (\d+)\): (?:CONNECT (\S+?)(?::\d+)? |\w+ https?://([^/:\s]+))")
_REFUSED = re.compile(r'refused on filtered domain "([^"]+)"')


@dataclass
class NetworkLog:
    allowed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_proxy() -> None:
    """Create the internal network and start the proxy if they aren't there yet."""
    if _run("network", "inspect", NETWORK).returncode != 0:
        _check("network", "create", "--internal", NETWORK)
    state = _run("inspect", "-f", "{{.State.Running}}", PROXY_NAME)
    if state.returncode != 0:
        _check("run", "-d", "--name", PROXY_NAME, "--network", "bridge",
               "--restart", "unless-stopped", PROXY_IMAGE)
        _check("network", "connect", NETWORK, PROXY_NAME)
        time.sleep(1)
    elif state.stdout.strip() != "true":
        _check("start", PROXY_NAME)
        time.sleep(1)


def container_ip(name: str) -> str:
    out = _check("inspect", "-f",
                 "{{(index .NetworkSettings.Networks \"" + NETWORK + "\").IPAddress}}", name)
    return out.strip()


def requests_from(ip: str, since: str) -> NetworkLog:
    """Hosts the container at `ip` asked the proxy for since `since` (RFC 3339)."""
    logs = _run("logs", "--since", since, PROXY_NAME)
    text = logs.stdout + logs.stderr
    fd_ip: dict[str, str] = {}
    last_host: dict[str, str] = {}
    allowed, blocked = [], []
    for line in text.splitlines():
        if m := _CONNECT.search(line):
            fd_ip[m.group(1)] = m.group(2)
        elif m := _REQUEST.search(line):
            fd, host = m.group(1), m.group(2) or m.group(3)
            if fd_ip.get(fd) == ip:
                last_host[fd] = host
                allowed.append(host)
        elif m := _REFUSED.search(line):
            host = m.group(1)
            if host in allowed:
                allowed.remove(host)
                blocked.append(host)
    return NetworkLog(sorted(set(allowed)), sorted(set(blocked)))


def proxy_image_exists() -> bool:
    return _run("image", "inspect", PROXY_IMAGE).returncode == 0


def build_proxy_image(repo_root: Path) -> None:
    subprocess.run(["docker", "build", "-t", PROXY_IMAGE, str(repo_root / "docker" / "proxy")],
                   check=True)


def now_rfc3339() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


def _check(*args: str) -> str:
    cp = _run(*args)
    if cp.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {cp.stderr.strip()}")
    return cp.stdout
