"""MLX inference engine.

Wraps mlx-lm but exposes what a remote API cannot: per-token logprobs, the
entropy of each step's distribution, and the residual stream (the hidden state
between layers) at chosen layers (DESIGN.md §10). Per reply, three positions are
recorded for each chosen layer:

  * prompt_last  the last prompt token: the model's state right before it replies,
                 after reading everything so far (including any pressure wording)
  * action       the first token inside the ```bash / ```submit block: the moment
                 it commits to a command
  * gen_mean     the mean over all generated tokens

A prefix cache keeps the model's key/value memory between calls: an agent
conversation only grows, so each step only processes the new messages instead of
the whole history.

One base model serves two roles via LoRA:
  * Worker  = base + the current trainable adapter
  * Auditor = base with NO adapter (so it cannot co-drift)
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)        # chosen-token logprob per step
    entropies: list[float] = field(default_factory=list)       # distribution entropy per step
    # {"prompt_last" | "action" | "gen_mean": float16 array [n_layers, hidden]}
    activations: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    cached_tokens: int = 0                  # prompt tokens reused from the prefix cache

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
        memory_limit_gb: float = 14.0,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.activation_layers = activation_layers
        self.memory_limit_gb = memory_limit_gb
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: Any = None             # prefix (KV) cache from the previous call
        self._cache_tokens: list[int] = []  # the tokens that cache holds
        self._records: list[tuple[int, int, Any]] = []   # (layer, n_tokens, last hidden)

    # -- lazy load so importing/testing doesn't pull a 2 GB model ------------
    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load

            _limit_memory(self.memory_limit_gb)
            _check_room(self.model_name, self.memory_limit_gb)
            self._model, self._tokenizer = load(self.model_name, adapter_path=self.adapter_path)
            layers = self._model.model.layers
            # Fractions (0.5 = halfway down) resolve against this model's depth.
            self.activation_layers = tuple(
                int(round(x * (len(layers) - 1))) if isinstance(x, float) else int(x)
                for x in self.activation_layers)
            for i in self.activation_layers:
                layers[i] = _Recorder(layers[i], i, self._records)

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer

    def format_chat(self, messages: list[dict[str, str]], **template_kwargs: Any) -> str:
        """Render chat messages to a prompt string via the model's chat template.

        Extra kwargs go to the template, e.g. enable_thinking=False for Qwen3.
        """
        self._ensure_loaded()
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.7,
        capture_logprobs: bool = True,
        capture_activations: bool = True,
        top_p: float = 0.0,
        top_k: int = 0,
        presence_penalty: float = 0.0,
    ) -> GenerationOutput:
        """Generate a completion, capturing per-token logprob and entropy, and the
        residual stream at `activation_layers` (if any).

        Uses mlx-lm's stream_generate so we can read the per-step distribution.
        `response.logprobs` is the log-softmax vector over the vocabulary for the
        step; we record the chosen token's logprob and the full entropy.
        """
        self._ensure_loaded()
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        # Reuse the cache for the longest shared prefix; always feed at least one token.
        common = 0
        if self._cache is not None:
            limit = min(len(tokens) - 1, len(self._cache_tokens))
            while common < limit and tokens[common] == self._cache_tokens[common]:
                common += 1
            trim_prompt_cache(self._cache, len(self._cache_tokens) - common)
        else:
            self._cache = make_prompt_cache(self._model)
        suffix = tokens[common:]

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        # Note: a penalty changes the distribution that logprob/entropy are read from.
        procs = (make_logits_processors(presence_penalty=presence_penalty)
                 if presence_penalty else None)
        out = GenerationOutput(text="", prompt_tokens=len(tokens), cached_tokens=common)
        pieces: list[str] = []
        self._records.clear()
        try:
            for resp in stream_generate(self._model, self._tokenizer, suffix, max_tokens=max_tokens,
                                        sampler=sampler, logits_processors=procs,
                                        prompt_cache=self._cache):
                pieces.append(resp.text)
                tok = int(resp.token)
                out.token_ids.append(tok)
                if capture_logprobs and getattr(resp, "logprobs", None) is not None:
                    lp_vec = resp.logprobs
                    out.logprobs.append(float(lp_vec[tok]))
                    out.entropies.append(_entropy(lp_vec))
        except BaseException:
            self.reset_cache()
            raise
        # The cache holds the prompt plus (some of) the generated tokens: mlx-lm may have run
        # one token ahead, or not run the final stop token. Record exactly what it holds.
        held = tokens + out.token_ids
        n_cached = self._cache[0].offset
        if n_cached > len(held):
            trim_prompt_cache(self._cache, n_cached - len(held))
            n_cached = len(held)
        self._cache_tokens = held[:n_cached]

        out.text = "".join(pieces)
        if capture_activations and self.activation_layers:
            out.activations = self._collect(len(suffix), pieces)
        self._records.clear()
        return out

    def reset_cache(self) -> None:
        self._cache, self._cache_tokens = None, []

    def _collect(self, n_prompt: int, pieces: list[str]) -> dict[str, Any]:
        """Turn the per-call records into the three per-layer vectors (see module doc).

        Each model call processes a run of tokens; counting tokens per call tells which
        call ended on the last prompt token and which calls were generated tokens."""
        import numpy as np

        n_gen = len(pieces)
        action_at = _action_token(pieces)
        result: dict[str, list] = {"prompt_last": [], "action": [], "gen_mean": []}
        for layer in self.activation_layers:
            seen, last, gen = 0, None, []
            for li, n, h in self._records:
                if li != layer:
                    continue
                seen += n
                if seen == n_prompt:
                    last = h
                elif seen > n_prompt and len(gen) < n_gen:
                    gen.append(h)       # hidden state at generated token len(gen)
            d = last.shape[-1] if last is not None else 0
            to_np = lambda a: np.array(a.astype(mx_float32()), dtype=np.float32)  # noqa: E731
            result["prompt_last"].append(to_np(last) if last is not None else np.zeros(d))
            if gen:
                g = np.stack([to_np(x) for x in gen])
                result["gen_mean"].append(g.mean(0))
                result["action"].append(g[min(action_at, len(g) - 1)])
            else:
                result["gen_mean"].append(np.zeros(d))
                result["action"].append(np.zeros(d))
        return {k: np.stack(v).astype(np.float16) for k, v in result.items()}

    def with_adapter(self, adapter_path: str | None) -> InferenceEngine:
        """A view of the SAME model id with a different LoRA adapter (or none)."""
        return InferenceEngine(self.model_name, adapter_path, self.activation_layers,
                               self.memory_limit_gb)


# -- keeping the Mac usable --------------------------------------------------------------
# MLX keeps weights in wired memory, which macOS cannot swap out or reclaim, and mlx-lm
# raises the wired limit to the GPU's recommended maximum (17.8 GB on a 24 GB Mac) for
# every generation. With Docker and normal apps running, that starves the system and the
# machine hangs. So: cap MLX's total and wired memory, keep its buffer cache small, and
# refuse to load a model the machine has no room for.

def _limit_memory(limit_gb: float) -> None:
    import mlx.core as mx
    import mlx_lm.generate as gen

    cap = int(limit_gb * 2**30)
    mx.set_memory_limit(cap)
    mx.set_wired_limit(cap)
    mx.set_cache_limit(2**30)                 # freed buffers go back to macOS beyond 1 GB
    # mlx-lm calls this around every generation to raise the wired limit; use our cap.
    gen.maybe_set_recommended_wired_limit = lambda: mx.set_wired_limit(cap)


def _check_room(model_name: str, limit_gb: float) -> None:
    """Refuse to load a model that does not fit the cap or the memory free right now."""
    weights = _local_weights_bytes(model_name)
    if weights is None:
        return                                # not downloaded yet: mlx-lm will fetch it
    need = weights + 1.5 * 2**30              # weights plus room for the KV cache
    if need > limit_gb * 2**30:
        raise MemoryError(f"{model_name} needs ~{need / 2**30:.1f} GB, above the "
                          f"{limit_gb:.0f} GB cap (backends.mlx.memory_limit_gb)")
    free = _available_bytes()
    reserve = 2 * 2**30                       # left for macOS and apps after loading
    if free is not None and need + reserve > free:
        raise MemoryError(f"{model_name} needs ~{need / 2**30:.1f} GB plus 2 GB left for macOS, "
                          f"but only {free / 2**30:.1f} GB is free; close apps or stop other models")


def _local_weights_bytes(model_name: str) -> int | None:
    try:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(model_name, local_files_only=True))
    except Exception:  # noqa: BLE001 — not in the local cache
        return None
    total = sum(f.stat().st_size for f in root.glob("*.safetensors"))   # stat follows links
    return total or None                      # only config files cached: not downloaded


def _available_bytes() -> int | None:
    """Memory macOS can hand out now: free + inactive + purgeable + speculative pages."""
    import re
    import subprocess

    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {k.strip(): int(v) for k, v in re.findall(r"^(.+?):\s+(\d+)\.", out, re.M)}
    keys = ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
    return page * sum(pages.get(k, 0) for k in keys)


def _Recorder(inner: Any, index: int, records: list) -> Any:
    """Stands in for one decoder layer: runs it and keeps its output at the last position.

    An mlx Module (so the model's parameter tree still sees the wrapped layer); the
    record list is held by the closure, not as an attribute, so it never becomes part
    of that tree."""
    import mlx.nn as nn

    class Recorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = inner

        def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
            h = self.inner(x, *args, **kwargs)
            records.append((index, h.shape[1], h[0, -1]))
            return h

    return Recorder()


def mx_float32() -> Any:
    import mlx.core as mx

    return mx.float32


def _action_token(pieces: list[str]) -> int:
    """Index of the first generated token inside the action block (or 0)."""
    text = ""
    for i, p in enumerate(pieces):
        text += p
        for opener in ("```bash\n", "```submit\n"):
            j = text.find(opener)
            if j != -1 and len(text) > j + len(opener):
                return i
    return 0


def _entropy(logprob_vector: Any) -> float:
    """Shannon entropy (nats) of a log-probability vector (mx.array or list)."""
    try:
        import mlx.core as mx

        probs = mx.exp(logprob_vector)
        return float(-mx.sum(probs * logprob_vector))
    except Exception:
        # Fallback for a plain Python list of logprobs.
        return float(-sum(math.exp(lp) * lp for lp in logprob_vector))
