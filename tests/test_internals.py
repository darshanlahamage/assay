"""The internals analysis recovers a planted pressure direction and misbehaviour signal."""

import json

import numpy as np

from assay.internals import analyse


def _fake_sweep(tmp_path, seed=0, signal=2.5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    D, L = 64, 2
    press = rng.normal(size=D); press /= np.linalg.norm(press)
    bad = rng.normal(size=D); bad /= np.linalg.norm(bad)
    (tmp_path / "experiment.yaml").write_text(
        "conditions: [{name: neutral}, {name: high}]\n")
    rows = []
    for t in range(4):
        task_offset = rng.normal(size=D) * 3
        for cond in ("neutral", "high"):
            for k in range(4):
                misb = cond == "high" and k < 3          # pressure -> mostly misbehaves
                n = 5
                acts = {}
                for kind in ("prompt_last", "action", "gen_mean"):
                    a = task_offset + rng.normal(size=(n, L, D))
                    if cond == "high":
                        a[0] += 4 * press                # wording moves the step-0 state
                    if misb:
                        a += signal * bad                   # misbehaving runs drift along `bad`
                    acts[kind] = a.astype(np.float16)
                name = f"task{t}/m__{cond}__off__r{k:02d}"
                d = tmp_path / name; d.mkdir(parents=True)
                np.savez(d / "activations.npz", step=np.arange(n), **acts)
                ent = [[float(rng.random())] for _ in range(n)]
                if misb:
                    ent[-1] = [5.0]                      # high entropy at the offending step
                steps = [{"index": i, "entropies": ent[i], "logprobs": [-0.1],
                          "touched_files": []} for i in range(n)]
                (d / "trajectory.json").write_text(json.dumps({"steps": steps}))
                rows.append({"rollout": name, "task_id": f"task{t}", "model": "m",
                             "verdicts": ["cheat"] if misb else ["honest_fail"],
                             "extra": {"condition": cond}})
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_recovers_pressure_direction_probe_and_entropy(tmp_path):
    a = analyse(_fake_sweep(tmp_path))
    assert a["rollouts"] == 32 and a["misbehaved"] == 12
    assert a["pressure"][0]["conditions"]["high"]["heldout_auc"] > 0.95
    best = max(p["rollout_auc"] for p in a["probe"])
    assert best > 0.9
    assert a["uncertainty"]["entropy"]["auc_offending_vs_other"] > 0.95


def test_no_signal_no_probe(tmp_path):
    """Negative control: without a planted misbehaviour signal the probe stays near chance
    once pressure is held fixed (the pressure wording alone must not pass for detection)."""
    a = analyse(_fake_sweep(tmp_path, seed=1, signal=0.0))
    assert max(p["rollout_auc_same_pressure"] for p in a["probe"]) < 0.8


def test_permutation_p_separates_signal_from_none(tmp_path):
    from assay.internals import load, probe

    real = load(_fake_sweep(tmp_path / "a", seed=2))
    none = load(_fake_sweep(tmp_path / "b", seed=2, signal=0.0))
    assert probe(real, 1, "action", permutations=30)["perm_p_same_pressure"] < 0.1
    assert probe(none, 1, "action", permutations=30)["perm_p_same_pressure"] > 0.1
