"""Containers that agent commands and agent code run in. Nothing agent-produced runs on the host."""

from assay.sandbox.container import Container, ContainerSpec, ExecResult

__all__ = ["Container", "ContainerSpec", "ExecResult"]
