"""Task specification + loader for the hackable benchmark."""

from candor.tasks.base import TaskSpec
from candor.tasks.loader import load_task, load_suite

__all__ = ["TaskSpec", "load_task", "load_suite"]
