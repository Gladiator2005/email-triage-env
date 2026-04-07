"""Email Triage & Response OpenEnv package."""

from .models import (
    AgentPerformance,
    Email,
    EnvironmentState,
    GraderResult,
    InboxMetrics,
    TriageAction,
    TriageObservation,
)
from .environment import TriageEnvironment

# Backward-compatible alias for external callers expecting task configs at module scope.
TASK_CONFIGS = TriageEnvironment.TASKS

__all__ = [
    "Email",
    "TriageAction",
    "TriageObservation",
    "EnvironmentState",
    "GraderResult",
    "InboxMetrics",
    "AgentPerformance",
    "TriageEnvironment",
    "TASK_CONFIGS",
]
