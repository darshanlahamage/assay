"""What the agent did, step by step, with the actual results (never the agent's claims).

The trajectory is the input to verdicts, monitors and probes, and the unit of RL data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Step:
    index: int
    kind: str                       # "bash" | "submit" | "format_error"
    reply: str                      # full model reply
    reasoning: str                  # text outside the action block
    thinking: str = ""              # private reasoning (chain of thought), when available
    command: str = ""
    output: str = ""                # actual output, as recorded by the harness
    returncode: int | None = None
    timed_out: bool = False
    touched_files: list[str] = field(default_factory=list)   # files changed by this step
    notes: list[str] = field(default_factory=list)   # harness messages shown after this step
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    # Local track only: per-token signals for this reply.
    logprobs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)


@dataclass
class Trajectory:
    task_id: str
    backend: str
    model: str
    steps: list[Step] = field(default_factory=list)
    submitted: bool = False
    report: str = ""                # the agent's final report (a claim, not evidence)
    stop_reason: str = ""           # submitted | step_limit | error
    error: str = ""
    # Local track: per-step activations, aligned with `steps` (saved separately as .npz,
    # never in the JSON).
    activations: list[dict] = field(default_factory=list, repr=False)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("activations")
        return d

    def save_activations(self, path) -> bool:
        """Write per-step activations as one .npz: <kind> arrays [steps, layers, hidden]."""
        if not self.activations or not all(self.activations):
            return False
        import numpy as np

        kinds = self.activations[0].keys()
        np.savez_compressed(path, step=np.arange(len(self.activations)),
                            **{k: np.stack([a[k] for a in self.activations]) for k in kinds})
        return True
