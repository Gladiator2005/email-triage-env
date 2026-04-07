"""
models.py — Typed Pydantic models for the Email Triage & Response OpenEnv environment.

Action, Observation, State, and GraderResult follow the OpenEnv specification.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """A single email in the inbox."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str                          # ISO-8601 string
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal[
        "customer_complaint", "billing_inquiry", "technical_support",
        "sales_inquiry", "spam", "internal", "other"
    ]
    requires_response: bool
    sla_hours: float                        # Hours until response deadline
    sentiment: Literal["positive", "neutral", "negative", "angry"]


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    Action the agent takes on an email.

    action_type:
      - triage   : Classify and prioritise the email, optionally add a note.
      - respond  : Send a reply to the email.
      - escalate : Escalate to human agent with a reason.
      - archive  : Mark email as handled without response (valid for spam/resolved).
      - skip     : Do nothing (move to next email). Penalised if SLA is close.

    email_id    : Which email the action targets (must be in current inbox).
    priority    : Override priority (for 'triage' action).
    category    : Override category (for 'triage' action).
    response_text: The reply body (for 'respond' action, max 2000 chars).
    escalation_reason: Reason string (for 'escalate' action).
    note        : Optional internal note attached to any action.
    """
    action_type: Literal["triage", "respond", "escalate", "archive", "skip"]
    email_id: str
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = None
    category: Optional[Literal[
        "customer_complaint", "billing_inquiry", "technical_support",
        "sales_inquiry", "spam", "internal", "other"
    ]] = None
    response_text: Optional[str] = Field(None, max_length=2000)
    escalation_reason: Optional[str] = Field(None, max_length=500)
    note: Optional[str] = Field(None, max_length=300)


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class InboxMetrics(BaseModel):
    total_emails: int
    unhandled: int
    overdue: int                            # Past SLA
    near_sla: int                           # Within 1 hour of SLA
    responded: int
    escalated: int
    archived: int


class AgentPerformance(BaseModel):
    correct_classifications: int
    incorrect_classifications: int
    responses_sent: int
    avg_response_quality: float             # 0.0–1.0 rolling average
    sla_breaches: int
    sla_met: int
    escalations_appropriate: int
    escalations_unnecessary: int
    spam_correctly_archived: int
    spam_missed: int


class TriageObservation(BaseModel):
    """Full observation returned by reset() and step()."""
    # Current email being presented for action
    current_email: Optional[Email]

    # Inbox overview
    inbox_metrics: InboxMetrics

    # Agent's performance so far this episode
    performance: AgentPerformance

    # Step information
    step_num: int
    max_steps: int
    time_elapsed_minutes: float             # Simulated clock

    # Reward from the last action (None on reset)
    reward: Optional[float]

    # Episode done?
    done: bool

    # Context: last 3 actions the agent took (for continuity)
    recent_actions: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Full internal state (for state() endpoint)."""
    task_id: str
    seed: int
    step_num: int
    max_steps: int
    inbox: List[Email]
    current_email_index: int
    handled_email_ids: List[str]
    performance: AgentPerformance
    inbox_metrics: InboxMetrics
    total_reward: float
    done: bool
    time_elapsed_minutes: float


# ---------------------------------------------------------------------------
# Grader result
# ---------------------------------------------------------------------------

class GraderResult(BaseModel):
    """Returned by POST /grade."""
    task_id: str
    score: float                            # 0.0–1.0
    breakdown: Dict[str, float]             # Component scores
    explanation: str
    passed: bool                            # score >= 0.6 threshold