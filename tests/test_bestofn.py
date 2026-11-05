import itertools
import random

from assay.bestofn import selection, win_probabilities


def _brute(scores, n):
    wins = [0.0] * len(scores)
    subsets = list(itertools.combinations(range(len(scores)), n))
    for sub in subsets:
        top = max(scores[i] for i in sub)
        tied = [i for i in sub if scores[i] == top]
        for i in tied:
            wins[i] += 1 / len(tied) / len(subsets)
    return wins


def test_exact_formula_matches_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        k = rng.randint(1, 7)
        scores = [rng.choice([0.0, 0.5, 0.6, 1.0]) for _ in range(k)]   # many ties
        for n in range(1, k + 1):
            exact, brute = win_probabilities(scores, n), _brute(scores, n)
            assert all(abs(a - b) < 1e-12 for a, b in zip(exact, brute))
            assert abs(sum(exact) - 1) < 1e-12


def _row(naive, trusted, hidden, verdicts):
    return {"proxy_naive": naive, "proxy_trusted": trusted, "true_score": hidden, "verdicts": verdicts}


def test_naive_selection_picks_the_cheat_trusted_does_not():
    rows = [_row(1.0, 0.0, 0.0, ["cheat"])] + [_row(0.6, 0.6, 0.6, ["honest_fail"])] * 3
    assert selection(rows, 1, "naive")["misbehaved"] == 0.25
    assert selection(rows, 4, "naive")["misbehaved"] == 1.0          # always kept
    assert selection(rows, 4, "trusted")["misbehaved"] == 0.0
    assert selection(rows, 4, "trusted")["hidden"] == 0.6
