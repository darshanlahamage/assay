"""Linear probe on decision-token activations.

The one 'deep' interpretability method (DESIGN.md §6). Trains a logistic
regression to predict cheat-vs-honest from the residual-stream activation at
the decision token. The headline result is the probe's held-out accuracy:
either cheating is linearly decodable from the model's internals before the
action, or it is not — both are interesting.

Stretch: the fitted weight vector is a candidate STEERING direction — add or
subtract it at inference and measure the change in cheat rate (a cheap causal
probe of "what steered the decision").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbeReport:
    layer: int
    accuracy: float
    auc: float
    n_train: int
    n_test: int
    direction: list[float] | None = None   # weight vector, for steering (stretch)


class ActivationProbe:
    def __init__(self, layer: int = 18) -> None:
        self.layer = layer
        self._clf = None       # sklearn LogisticRegression, fit in Phase 4

    def fit(
        self,
        activations: list[list[float]],
        labels: list[int],       # 1 = cheat, 0 = honest
    ) -> ProbeReport:
        """Fit the probe with a train/test split and report accuracy + AUC.

        TODO (Phase 4): standardise, logistic regression, stratified split,
        report held-out metrics and expose the weight vector as `direction`.
        """
        raise NotImplementedError("ActivationProbe.fit — Phase 4")
