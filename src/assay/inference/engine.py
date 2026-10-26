"""MLX inference engine.

This wraps mlx-lm but exposes two things a remote API cannot: per-token
logprobs/entropy, and residual-stream activations at chosen layers. These are
the raw material for the auditor's inference-level signals (DESIGN.md §6).

One base model serves two roles via LoRA:
  * Worker  = base + the current trainable adapter
  * Auditor = base with NO adapter (so it cannot co-drift)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)        # per generated token
    entropies: list[float] = field(default_factory=list)       # per generated token
    # activations[layer] -> vector at the final/decision token, for the probe.
    activations: dict[int, list[float]] = field(default_factory=dict)


class InferenceEngine:
    """Loads an MLX model and generates with optional signal capture."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        activation_layers: tuple[int, ...] = (),
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.activation_layers = activation_layers
        # TODO (Phase 1): mlx_lm.load(model_name, adapter_path=...)
        self._model: Any = None
        self._tokenizer: Any = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        grammar: str | None = None,          # constrained decoding for tool calls
        capture_logprobs: bool = False,
        capture_activations: bool = False,
    ) -> GenerationOutput:
        """Generate a completion.

        TODO (Phase 1): drive the mlx-lm generation loop manually so we can read
        logits (-> logprob + entropy) at each step and hook the residual stream
        at `self.activation_layers` for the decision token.
        """
        raise NotImplementedError("InferenceEngine.generate — Phase 1")

    def with_adapter(self, adapter_path: str | None) -> "InferenceEngine":
        """Return a view of the SAME base weights with a different LoRA adapter
        (or none, for the Auditor). Avoids loading the base model twice."""
        raise NotImplementedError("InferenceEngine.with_adapter — Phase 1")
