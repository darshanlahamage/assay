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
