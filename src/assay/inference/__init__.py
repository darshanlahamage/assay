"""MLX inference: generation plus the local-only signals (logprobs, activations)
that make inference-level interpretability possible."""

from assay.inference.engine import GenerationOutput, InferenceEngine

__all__ = ["GenerationOutput", "InferenceEngine"]
