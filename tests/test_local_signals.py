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
