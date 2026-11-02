"""Pressure wording, sweep planning and the statistics. Pure Python, no Docker."""

from __future__ import annotations

import pytest

from assay.analysis import fisher_p, wilson
from assay.pressure import PROMPTS, Pressure, apply, budget
from assay.sweep import plan


def test_neutral_control_is_similar_length_to_high():
    assert 0.6 < len(PROMPTS["neutral"]) / len(PROMPTS["high"]) < 1.4


def test_apply_adds_only_what_is_switched_on():
    base = apply("Fix it.", Pressure(), {"tests/t.py": "assert 1"}, "1 failed")
    assert base == "Fix it."
    full = apply("Fix it.", Pressure(prompt="high", failures=2, tests_shown=True,
                                     oversight="watched"), {"tests/t.py": "assert 1"}, "1 failed")
    for piece in (PROMPTS["high"], "2 earlier attempts", "1 failed", "assert 1", "senior engineer"):
        assert piece in full


def test_budget():
    assert budget(Pressure(), 30) == (30, False)
    assert budget(Pressure(budget="tight"), 30) == (8, True)


def test_invalid_pressure_rejected():
    with pytest.raises(ValueError):
        Pressure(prompt="extreme").validate()


def test_label():
    assert Pressure().label() == "no pressure"
    assert Pressure(prompt="high", failures=3).label() == "prompt=high, failures=3"


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.25 < hi < 0.35
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


def test_fisher():
    p = fisher_p(9, 10, 1, 10)
    assert p is not None and p < 0.01


def test_plan_interleaves_conditions_and_coerces_names():
    exp = {"tasks": ["task_00*"], "repeats": 2,
           "agents": [{"backend": "script", "name": "a", "scenario": "x"}],
           "conditions": [{"name": "base"}, {"name": "high"}], "defenses": [{"name": False}]}
    jobs = plan(exp)
    assert len(jobs) == 2 * 2 * 2                       # repeats x tasks x conditions
    assert [j["condition"]["name"] for j in jobs[:2]] == ["base", "high"]
    assert all("__off__" in j["rollout"] for j in jobs)
