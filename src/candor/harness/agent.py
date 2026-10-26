"""The ReAct agent loop.

observe (tool results) -> think (scratchpad, logged) -> act (one tool call)
repeated until `submit`, `give_up`, or the step budget is exhausted.

The loop records a full Trajectory (prompts, tool calls, actual tool results,
reasoning stream, and — when the inference engine supports it — per-token
logprobs and decision-token activations) for the RL engine and the auditor.
"""

from __future__ import annotations

from dataclasses import dataclass

from candor.harness.sandbox import Sandbox
from candor.harness.tools import Tool, ToolCall, ToolResult
from candor.inference.engine import InferenceEngine
from candor.rl.trajectories import Step, Trajectory


@dataclass
class AgentConfig:
    max_steps: int = 25
    scratchpad: bool = True


class Agent:
    """Drives one episode on one task."""

    def __init__(
        self,
        engine: InferenceEngine,
        tools: list[Tool],
        sandbox: Sandbox,
        config: AgentConfig,
    ) -> None:
        self.engine = engine
        self.tools = {t.name: t for t in tools}
        self.sandbox = sandbox
        self.config = config

    def run_episode(self, task_prompt: str) -> Trajectory:
        """Run one episode and return its Trajectory.

        TODO (Phase 1):
          * build the system prompt (task + tool schemas for constrained decoding)
          * loop: engine.generate(..., capture_logprobs, capture_activations)
          * parse exactly one tool call (grammar-constrained), dispatch it,
            append the ACTUAL ToolResult (never the agent's claim)
          * stop on submit / give_up / budget
        """
        raise NotImplementedError("Agent.run_episode — Phase 1")

    def _dispatch(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call against the sandbox.

        Jail violations are caught and returned as a failed ToolResult tagged as
        a security event, rather than crashing the episode.
        """
        raise NotImplementedError("Agent._dispatch — Phase 1")
