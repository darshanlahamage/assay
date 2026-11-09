import numpy as np

from assay.agent.backends import _retry_in
from assay.agent.trajectory import Step, Trajectory
from assay.inference.engine import _action_token


def test_retry_in_parses_groq_durations():
    assert _retry_in("Please try again in 41m35.664s. Need more") == 41 * 60 + 35.664
    assert _retry_in("try again in 1h2m3s") == 3723
    assert _retry_in("rate limited") is None


def test_action_token_is_first_token_inside_the_block():
    pieces = ["I'll", " look", ".\n", "```", "bash", "\n", "ls", " -la", "\n```"]
    assert pieces[_action_token(pieces)] == "ls"
    assert _action_token(["no", " block"]) == 0


def test_activations_saved_beside_json_not_in_it(tmp_path):
    t = Trajectory(task_id="t", backend="mlx", model="m",
                   steps=[Step(index=0, kind="bash", reply="", reasoning="")])
    t.activations.append({"prompt_last": np.ones((2, 4), np.float16),
                          "action": np.zeros((2, 4), np.float16)})
    assert "activations" not in t.to_dict()
    assert t.save_activations(tmp_path / "a.npz")
    z = np.load(tmp_path / "a.npz")
    assert z["prompt_last"].shape == (1, 2, 4) and list(z["step"]) == [0]


def test_no_activations_no_file(tmp_path):
    t = Trajectory(task_id="t", backend="groq", model="m")
    t.activations.append({})
    assert not t.save_activations(tmp_path / "a.npz")


def test_harmony_tool_call_becomes_a_bash_block():
    from assay.agent.backends import _parse_harmony
    raw = ('<|channel|>analysis<|message|>Need to see files.<|end|><|start|>assistant to=functions.bash'
           '<|channel|>commentary json<|message|>{"command": "ls -la"}<|call|>')
    assert _parse_harmony(raw) == ("```bash\nls -la\n```", "Need to see files.")
    raw2 = ('<|channel|>analysis<|message|>ok<|end|><|start|>assistant<|channel|>commentary '
            'to=functions.submit <|constrain|>json<|message|>"{\\"report\\": \\"done\\"}"<|call|>')
    assert _parse_harmony(raw2)[0] == "```submit\ndone\n```"


def test_harmony_final_and_truncated():
    from assay.agent.backends import _parse_harmony
    assert _parse_harmony("<|channel|>analysis<|message|>hm<|end|><|start|>assistant"
                          "<|channel|>final<|message|>All set.<|return|>") == ("All set.", "hm")
    assert _parse_harmony("<|channel|>analysis<|message|>thinking forever") == ("", "thinking forever")


def test_tool_history_args_as_dict_for_templates():
    from assay.agent.backends import _as_tool_history
    h = _as_tool_history([{"role": "user", "content": "go"},
                          {"role": "assistant", "content": "```bash\nls\n```"},
                          {"role": "user", "content": "exit code: 0\nx"}], args_as_dict=True)
    assert h[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert h[2]["role"] == "tool"


def test_memory_guard(monkeypatch):
    import pytest

    from assay.inference import engine
    GB = 2**30
    monkeypatch.setattr(engine, "_local_weights_bytes", lambda m: int(11.2 * GB))
    monkeypatch.setattr(engine, "_available_bytes", lambda: 13 * GB)
    with pytest.raises(MemoryError, match="only 13.0 GB is free"):
        engine._check_room("big", 14.0)           # fits the cap, but would leave macOS < 2 GB
    monkeypatch.setattr(engine, "_available_bytes", lambda: 20 * GB)
    engine._check_room("big", 14.0)               # enough room
    with pytest.raises(MemoryError, match="above the 12 GB cap"):
        engine._check_room("big", 12.0)
    monkeypatch.setattr(engine, "_local_weights_bytes", lambda m: None)
    engine._check_room("not-downloaded", 14.0)    # unknown size: no refusal


def test_every_backend_builds_from_config(monkeypatch):
    """Constructors and make_backend agree (a misplaced attribute once broke two backends)."""
    from assay.agent.backends import make_backend
    from assay.pipeline import load_config
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    cfg = load_config("configs/default.yaml")
    cfg["backends"]["groq"]["quota_wait_max_s"] = 60
    g = make_backend("groq", cfg)
    assert g.quota_wait_max_s == 60 and g.reasoning_effort == "medium"
    make_backend("gemini", cfg)
    m = make_backend("mlx", cfg)            # does not load the model
    assert m.engine.memory_limit_gb == 14


class _Resp:
    def __init__(self, code, body, headers=None):
        import json
        self.status_code, self.headers = code, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)
        self._body = body

    def json(self):
        return self._body


def test_groq_rate_limit_waits_are_not_failures(monkeypatch):
    import httpx

    from assay.agent import backends
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    ok = {"choices": [{"message": {"content": "", "reasoning": "r", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    replies = [_Resp(429, "Rate limit reached ... tokens per minute (TPM)", {"retry-after": "3"})] * 20
    replies.append(_Resp(200, ok))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: replies.pop(0))
    c = backends.GroqBackend("m", max_retries=8).complete("s", [{"role": "user", "content": "go"}])
    assert c.text.strip() == "```bash\nls\n```" and c.thinking == "r"


def test_groq_failure_names_the_last_error(monkeypatch):
    import httpx
    import pytest

    from assay.agent import backends
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(503, "service unavailable"))
    with pytest.raises(RuntimeError, match="503"):
        backends.GroqBackend("m", max_retries=3).complete("s", [{"role": "user", "content": "go"}])


def test_qwen_think_split():
    from assay.agent.backends import _split_think
    assert _split_think("<think>\nlook first\n</think>\n\nOk.\n```bash\nls\n```") == ("Ok.\n```bash\nls\n```", "look first")
    assert _split_think("```bash\nls\n```") == ("```bash\nls\n```", "")
    assert _split_think("<think>still thinking when cut off") == ("", "still thinking when cut off")


def test_compact_shortens_recent_outputs_when_needed():
    from assay.agent.backends import _compact
    big = "x" * 6000
    msgs = [{"role": "user", "content": "TASK " * 50}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": f"```bash\ncmd {i}\n```"}, {"role": "user", "content": big}]
    out = _compact(msgs, 8000)
    assert sum(len(m["content"]) for m in out) <= 8000
    assert out[0] == msgs[0]                                         # task untouched
    assert all(o["content"] == m["content"] for o, m in zip(out, msgs) if m["role"] == "assistant")
    assert out[-1]["content"].startswith("x") and out[-1]["content"].endswith("x")   # head and tail kept
    small = msgs[:3]
    assert _compact(small, 100000) == small                          # nothing to do


def test_groq_output_parse_failure_is_retried(monkeypatch):
    import httpx

    from assay.agent import backends
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    bad = _Resp(400, '{"error":{"code":"output_parse_failed","failed_generation":"thinking about regex"}}')
    ok = {"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]}}], "usage": {}}
    replies = [bad, _Resp(200, ok)]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: replies.pop(0))
    c = backends.GroqBackend("m").complete("s", [{"role": "user", "content": "go"}])
    assert c.text.strip() == "```bash\npytest -q\n```"


def test_groq_reply_cap_fits_the_minute_limit_and_learns_from_413(monkeypatch):
    import json as _json

    import httpx

    from assay.agent import backends
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    sent = []
    ok = {"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}}], "usage": {}}
    replies = [_Resp(413, "Request too large for model: tokens per minute (TPM): Limit 8000, Requested 9500"),
               _Resp(200, ok)]

    def post(url, headers, json, timeout):
        sent.append(json["max_completion_tokens"])
        return replies.pop(0)
    monkeypatch.setattr(httpx, "post", post)
    msgs = [{"role": "user", "content": "y" * 12000}]          # ~4000 tokens of prompt
    backends.GroqBackend("m", max_prompt_chars=18000).complete("s", msgs)
    assert sent[0] + 12000 // 3 <= 8000                        # first request already fits
    assert sent[1] == max(1024, sent[0] - 1700)                # lowered by the reported excess


def test_compact_shortens_old_long_replies_but_keeps_the_action_block():
    from assay.agent import protocol
    from assay.agent.backends import _compact
    script = "```bash\npython - <<'PY'\n" + "print(1)\n" * 600 + "PY\n```"
    msgs = [{"role": "user", "content": "task"}]
    for i in range(8):
        msgs += [{"role": "assistant", "content": script}, {"role": "user", "content": "ok"}]
    out = _compact(msgs, 20000)                # the two recent 4.8k replies stay whole
    assert sum(len(m["content"]) for m in out) <= 20000
    assert out[1]["content"] != script and protocol.parse(out[1]["content"]).kind == "bash"
    assert out[-2]["content"] == script                          # recent reply untouched


def test_groq_keeps_room_for_the_reply_and_notes_cutoffs(monkeypatch):
    import httpx

    from assay.agent import backends, protocol
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(backends.time, "sleep", lambda s: None)
    sent = []
    cut = {"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning": "half a thou"}}],
           "usage": {"prompt_tokens": 4000, "completion_tokens": 3600}}

    def post(url, headers, json, timeout):
        sent.append(json)
        return _Resp(200, cut)
    monkeypatch.setattr(httpx, "post", post)
    msgs = [{"role": "user", "content": "task"}]
    for i in range(12):                                     # a long conversation, ~7k tokens
        msgs += [{"role": "assistant", "content": f"```bash\ncmd {i}\n```"}, {"role": "user", "content": "o" * 1800}]
    c = backends.GroqBackend("m", max_prompt_chars=18000).complete("s", msgs)
    assert sent[0]["max_completion_tokens"] >= 3500         # the reply was not squeezed to the floor
    assert c.note.startswith("reply cut off") and c.thinking == "half a thou"
    assert protocol.parse(c.text).kind == "format_error"
