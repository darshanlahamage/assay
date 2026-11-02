"""Local web dashboard: `assay ui` -> http://127.0.0.1:8765

Lists runs, replays rollouts (live while they run), and starts new runs. Pure standard
library: a small JSON API over the files in runs/, plus one static page. Binds to
localhost only.
"""

from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

from assay.audit.ledger import read_rows
from assay.report import summarise
from assay.viewer import _diff

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "runs"
STATIC = Path(__file__).parent / "static"
_jobs: dict[str, dict] = {}            # started from the UI: job id -> {proc, args, log}
_lock = threading.Lock()


# -- data ---------------------------------------------------------------------------
def list_runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    for d in sorted((p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")),
                    reverse=True):
        rows = read_rows(d / "ledger.jsonl")
        rollouts = sorted(p.parent for p in d.glob("*/*/trajectory.json"))
        verdicts: dict[str, int] = {}
        for r in rows:
            for v in r["verdicts"]:
                verdicts[v] = verdicts.get(v, 0) + 1
        models = sorted({r["model"] for r in rows}) or sorted(
            {_read_json(p / "trajectory.json").get("model", "?") for p in rollouts})
        tasks = sorted({r["task_id"] for r in rows}) or sorted({p.parent.name for p in rollouts})
        out.append({"id": d.name, "rollouts": len(rollouts), "finished": len(rows),
                    "running": len(rollouts) > len(rows), "models": models, "tasks": tasks,
                    "verdicts": verdicts})
    return out


def run_detail(run_id: str) -> dict:
    d = _safe(RUNS / run_id)
    rows = read_rows(d / "ledger.jsonl")
    by_name = {r["rollout"]: r for r in rows}
    rollouts = []
    for p in sorted(d.glob("*/*/trajectory.json")):
        name = f"{p.parent.parent.name}/{p.parent.name}"
        traj = _read_json(p)
        rollouts.append({"name": name, "row": by_name.get(name),
                         "steps": len(traj.get("steps", [])), "model": traj.get("model"),
                         "task_id": traj.get("task_id")})
    return {"id": run_id, "rollouts": rollouts, "summary": summarise(rows) if rows else []}


def rollout_detail(run_id: str, name: str) -> dict:
    r = _safe(RUNS / run_id / name)
    row = next((x for x in read_rows(RUNS / run_id / "ledger.jsonl") if x["rollout"] == name),
               None)
    initial, final = r / "initial_repo", r / "final_repo"
    files = []
    if initial.exists():
        from assay.grader.changes import snapshot
        files = [f for f in snapshot(initial) if not f.startswith(".git/")]
    return {
        "name": name, "row": row, "finished": row is not None,
        "prompts": _read_json(r / "prompts.json"),
        "trajectory": _read_json(r / "trajectory.json"),
        "grade": _read_json(r / "grade.json"),
        "security": _read_json(r / "security.json"),
        "files": files,
        "diff": _diff(initial, final) if initial.exists() and final.exists() else "",
    }


def options() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    tasks = sorted(p.parent.name for p in (REPO_ROOT / "benchmark").glob("*/meta.yaml"))
    scenarios = sorted(p.name for p in (REPO_ROOT / "scenarios").glob("*.yaml"))
    return {"tasks": tasks, "scenarios": scenarios,
            "backends": {k: v.get("model", "") for k, v in cfg["backends"].items()},
            "defaults": cfg["defenses"]}


def start_run(body: dict) -> dict:
    args = [sys.executable, "-m", "assay.cli", "run", "--quiet"]
    task = body.get("task", "")
    if task == "__all__":
        args += ["--suite", str(REPO_ROOT / "benchmark")]
    else:
        args += ["--task", str(_safe(REPO_ROOT / "benchmark" / task))]
    backend = body.get("backend", "script")
    args += ["--backend", backend, "-n", str(max(1, min(20, int(body.get("repeats", 1)))))]
    if backend == "script":
        args += ["--scenario", str(_safe(REPO_ROOT / "scenarios" / body.get("scenario", "")))]
    if body.get("model"):
        args += ["--model", str(body["model"])]
    for flag, key in (("--honesty-prompt", "honesty_prompt"),
                      ("--devmachine-review", "devmachine_review")):
        args += [flag, "on" if body.get(key) else "off"]
    args += ["--trusted-grading", body.get("trusted_grading", "off"),
             "--pressure", body.get("pressure", "none"),
             "--failures", str(int(body.get("failures", 0))),
             "--budget", body.get("budget", "default"),
             "--oversight", body.get("oversight", "none")]
    if body.get("tests_shown"):
        args.append("--show-tests")
    logs = RUNS / "_ui_logs"
    logs.mkdir(parents=True, exist_ok=True)
    job = time.strftime("%Y%m%d-%H%M%S")
    log = logs / f"{job}.log"
    before = {p.name for p in RUNS.iterdir()}
    proc = subprocess.Popen(args, cwd=REPO_ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT)
    run_id = None
    for _ in range(40):                       # the CLI creates the run folder right away
        new = [p.name for p in RUNS.iterdir() if p.name not in before and not p.name.startswith("_")]
        if new:
            run_id = sorted(new)[-1]
            break
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    with _lock:
        _jobs[job] = {"proc": proc, "run_id": run_id, "log": str(log)}
    return {"job": job, "run_id": run_id,
            "error": None if run_id else log.read_text()[-2000:]}


def _safe(p: Path) -> Path:
    p = p.resolve()
    if not (p == REPO_ROOT or REPO_ROOT in p.parents):
        raise PermissionError("path outside the project")
    return p


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# -- http ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/runs":
                return self._json(list_runs())
            if u.path == "/api/run":
                return self._json(run_detail(q["id"]))
            if u.path == "/api/rollout":
                return self._json(rollout_detail(q["run"], q["name"]))
            if u.path == "/api/analysis":
                from assay.analysis import analyse
                d = _safe(RUNS / q["id"])
                return self._json(analyse(d) if (d / "experiment.yaml").exists() else {})
            if u.path == "/api/options":
                return self._json(options())
            path = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
            f = _safe_static(path)
            if f is None:
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype)
        except (KeyError, PermissionError, FileNotFoundError) as e:
            return self._json({"error": str(e)}, 400)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/start":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._json(start_run(body))
        except (ValueError, PermissionError) as e:
            return self._json({"error": str(e)}, 400)


def _safe_static(rel: str) -> Path | None:
    p = (STATIC / rel).resolve()
    return p if p.is_file() and STATIC.resolve() in p.parents else None


def serve(port: int = 8765, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Assay dashboard: {url}   (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
