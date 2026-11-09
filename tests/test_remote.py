"""Remote GPU track: the Mac-side backend and the server's pure parts (no torch needed)."""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body


def _acts():
    buf = io.BytesIO()
    np.savez(buf, prompt_last=np.ones((2, 8), np.float16), action=np.zeros((2, 8), np.float16),
             gen_mean=np.ones((2, 8), np.float16))
    return base64.b64encode(buf.getvalue()).decode()


class _FakeServer:
    """Job API: POST /jobs, GET /jobs/<id>. `script` lists what each call does."""

    def __init__(self, monkeypatch, env, result, events=()):
        import httpx

        self.env, self.result, self.events = env, result, list(events)
        self.jobs, self.calls = {}, []
        monkeypatch.setattr(httpx, "post", self.post)
        monkeypatch.setattr(httpx, "get", self.get)

    def _event(self, url):
        import httpx

        self.calls.append(url)
        ev = self.events.pop(0) if self.events else None
        if ev == "drop":
            raise httpx.ConnectError("tunnel gone")
        if ev == "new_url":
            self.env["ASSAY_REMOTE_URL"] = "https://new.example"
            raise httpx.ConnectError("tunnel gone")
        if ev == "restart":
            self.jobs.clear()
        return ev

    def post(self, url, json, headers, timeout):
        self._event(url)
        self.jobs.setdefault(json["id"], 0)
        return _Resp(200, {"id": json["id"], "status": "running"})

    def get(self, url, headers, timeout):
        self._event(url)
        jid = url.rsplit("/", 1)[1]
        if jid not in self.jobs:
            return _Resp(404, {"error": "unknown job"})
        self.jobs[jid] += 1
        if self.jobs[jid] < 2:
            return _Resp(200, {"status": "running"})
        return _Resp(200, {"status": "done", **self.result})


_OK = {"text": "<think>look</think>\n```bash\nls\n```", "prompt_tokens": 10, "output_tokens": 5,
       "finish_reason": "stop", "logprobs": [-0.1], "entropies": [0.2]}


def test_remote_job_survives_a_dropped_tunnel_and_a_new_url(monkeypatch):
    from assay.agent import backends
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    env = {"ASSAY_REMOTE_URL": "https://old.example", "ASSAY_REMOTE_TOKEN": "t"}
    monkeypatch.setattr(backends, "_env_value", lambda k: env.get(k, ""))
    srv = _FakeServer(monkeypatch, env, {**_OK, "activations": _acts()},
                      events=[None, None, "drop", "new_url"])
    c = backends.RemoteBackend("Qwen/Qwen3-14B", activation_layers=(0.5, 0.75)).complete(
        "s", [{"role": "user", "content": "go"}])
    assert srv.calls[-1].startswith("https://new.example/jobs/")
    assert sum("/jobs/" not in u for u in srv.calls) == 1       # submitted once: the job kept running
    assert c.text == "```bash\nls\n```" and c.thinking == "look" and c.note == ""
    assert c.activations["prompt_last"].shape == (2, 8)


def test_remote_resubmits_when_the_server_restarted(monkeypatch):
    from assay.agent import backends
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    env = {"ASSAY_REMOTE_URL": "https://a.example", "ASSAY_REMOTE_TOKEN": "t"}
    monkeypatch.setattr(backends, "_env_value", lambda k: env.get(k, ""))
    srv = _FakeServer(monkeypatch, env, _OK, events=[None, "restart"])
    c = backends.RemoteBackend("m").complete("s", [{"role": "user", "content": "go"}])
    assert sum("/jobs/" not in u for u in srv.calls) == 2 and c.text.endswith("```")


def test_remote_gives_up_after_the_wait_limit_and_notes_cutoffs(monkeypatch):
    import httpx
    import pytest

    from assay.agent import backends
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    monkeypatch.setattr(backends, "_env_value", lambda k: "x")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(401, "bad token"))
    with pytest.raises(backends.QuotaExhausted, match="token"):
        backends.RemoteBackend("m").complete("s", [{"role": "user", "content": "go"}])
    env = {"ASSAY_REMOTE_URL": "https://a.example", "ASSAY_REMOTE_TOKEN": "t"}
    monkeypatch.setattr(backends, "_env_value", lambda k: env.get(k, ""))
    _FakeServer(monkeypatch, env, {"text": "<think>still going", "finish_reason": "length"})
    c = backends.RemoteBackend("m", max_tokens=8192).complete("s", [{"role": "user", "content": "go"}])
    assert c.text == "" and c.note.startswith("reply cut off")


def test_server_pure_parts_match_the_mlx_engine():
    from assay.inference import engine, torch_server as ts
    for pieces in (["ok", "\n```", "bash\n", "ls", "\n```"], ["```submit\n", "done"], ["no", "block"]):
        assert ts.action_token(pieces) == engine._action_token(pieces)
    assert ts.layer_indices([0.5, 0.75, 3], 40) == [20, 29, 3]       # engine: round(x * (n - 1))
    # records: prompt call of 5 tokens, then one call per generated token
    recs = [(20, 5, np.full(4, 1.0)), (20, 1, np.full(4, 2.0)), (20, 1, np.full(4, 4.0))]
    a = ts.collect(recs, [20], 5, ["```bash\n", "ls"])
    assert a["prompt_last"][0, 0] == 1 and a["gen_mean"][0, 0] == 3 and a["action"][0, 0] == 4


def test_notebook_embeds_the_current_server():
    import make_kaggle_notebook as mk
    nb = json.loads(mk.OUT.read_text())
    assert mk.embedded_server(nb) == mk.SERVER.read_text(), \
        "notebook is stale: run python tools/make_kaggle_notebook.py"
