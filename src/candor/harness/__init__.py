"""Agentic harness: the ReAct loop, the tool registry, and the sandbox.

The harness is where the agent acts, and therefore where it cheats. Every tool
is also an attack surface; the sandbox protects the *measurement* (the hidden
oracle), not the agent from cheating.
"""

from candor.harness.sandbox import PathJail, Sandbox
from candor.harness.tools import TOOL_REGISTRY, Tool, ToolCall, ToolResult

__all__ = ["PathJail", "Sandbox", "TOOL_REGISTRY", "Tool", "ToolCall", "ToolResult"]
