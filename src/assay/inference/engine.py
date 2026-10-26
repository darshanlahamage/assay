"""MLX inference engine.

Wraps mlx-lm but exposes what a remote API cannot: per-token logprobs and the
entropy of each step's distribution — the raw material for the auditor's
inference-level signals (DESIGN.md §6). Activation capture (for the Phase-4
probe) is stubbed here and added later.

One base model serves two roles via LoRA:
  * Worker  = base + the current trainable adapter
  * Auditor = base with NO adapter (so it cannot co-drift)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)        # chosen-token logprob per step
    entropies: list[float] = field(default_factory=list)       # distribution entropy per step
    activations: dict[int, list[float]] = field(default_factory=dict)  # Phase 4

    @property
    def mean_entropy(self) -> float:
        return sum(self.entropies) / len(self.entropies) if self.entropies else 0.0


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
        self._model: Any = None
        self._tokenizer: Any = None

    # -- lazy load so importing/testing doesn't pull a 2 GB model ------------
    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load  # noqa: PLC0415

            self._model, self._tokenizer = load(self.model_name, adapter_path=self.adapter_path)

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages to a prompt string via the model's chat template."""
        self._ensure_loaded()
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.7,
        capture_logprobs: bool = True,
        capture_activations: bool = False,  # Phase 4
    ) -> GenerationOutput:
        """Generate a completion, capturing per-token logprob and entropy.

        Uses mlx-lm's stream_generate so we can read the per-step distribution.
        `response.logprobs` is the log-softmax vector over the vocabulary for the
        step; we record the chosen token's logprob and the full entropy.
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        sampler = make_sampler(temp=temperature)
        out = GenerationOutput(text="")
        pieces: list[str] = []

        for resp in stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            pieces.append(resp.text)
            if capture_logprobs and getattr(resp, "logprobs", None) is not None:
                tok = int(resp.token)
                lp_vec = resp.logprobs
                out.token_ids.append(tok)
                out.logprobs.append(float(lp_vec[tok]))
                out.entropies.append(_entropy(lp_vec))

        out.text = "".join(pieces)
        return out

    def with_adapter(self, adapter_path: str | None) -> "InferenceEngine":
        """A view of the SAME model id with a different LoRA adapter (or none)."""
        return InferenceEngine(self.model_name, adapter_path, self.activation_layers)


def _entropy(logprob_vector: Any) -> float:
    """Shannon entropy (nats) of a log-probability vector (mx.array or list)."""
    try:
        import mlx.core as mx  # noqa: PLC0415

        probs = mx.exp(logprob_vector)
        return float(-mx.sum(probs * logprob_vector))
    except Exception:
        # Fallback for a plain Python list of logprobs.
        return float(-sum(math.exp(lp) * lp for lp in logprob_vector))
