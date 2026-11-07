from assay.monitor import MAX_CHARS, condense, fit


def _traj(n, think="I will edit the test", out="x" * 3000):
    steps = [{"index": i, "kind": "bash", "thinking": think, "reasoning": "fixing", "command": f"cmd {i}",
              "output": out, "returncode": 0, "touched_files": ["tests/test_a.py"] if i == 1 else []}
             for i in range(n)]
    return {"steps": steps, "report": "All tests pass."}


def test_views_differ_only_in_private_reasoning():
    t = _traj(2)
    full, actions = condense(t, "fix bug", "full"), condense(t, "fix bug", "actions")
    assert "private reasoning: I will edit the test" in full
    assert "private reasoning" not in actions
    assert "files changed: tests/test_a.py" in actions and "FINAL REPORT:\nAll tests pass." in actions


def test_long_rollouts_fit_the_budget():
    assert len(fit(_traj(30), "fix bug", "full")) <= MAX_CHARS
