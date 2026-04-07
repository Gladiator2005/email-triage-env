"""
environment.py — Core logic for the Email Triage & Response OpenEnv environment.

Classes:
  EmailGenerator    — Procedurally generates realistic inbox scenarios from a seed.
  ResponseEvaluator — Heuristic scorer for response quality (keyword + length checks).
  TriageEnvironment — Main environment: reset(), step(), state(), grade().
  Graders           — easy_classification, medium_sla_pressure, hard_angry_vip
"""

from __future__ import annotations

import random
import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

from models import (
    Email, TriageAction, TriageObservation,
    EnvironmentState, GraderResult,
    InboxMetrics, AgentPerformance,
)

# ---------------------------------------------------------------------------
# Corpus data for email generation
# ---------------------------------------------------------------------------

_SENDERS = [
    "alice.johnson@acme.com", "bob.smith@globex.net", "carol.white@initech.io",
    "david.brown@umbrella.co", "emma.davis@tyrell.ai", "frank.lee@oscorp.com",
    "grace.kim@cyberdyne.tech", "harry.wilson@wayneent.org", "irene.chen@soylent.co",
    "james.taylor@aperture.io", "karen.martin@stark.tech", "liam.anderson@nakatomi.com",
    "mary.jackson@weyland.corp", "noah.harris@macrohard.com", "olivia.garcia@hooli.co",
    "peter.clark@piedpiper.io", "quinn.walker@dunder.biz", "rachel.hall@pied.io",
    "sam.allen@vandelay.com", "tina.young@gekko.group",
    "vip_ceo@enterprise.com", "vip_cfo@enterprise.com", "vip_cto@enterprise.com",
    "spam@prizewinnertoday.xyz", "noreply@discount-meds.biz", "info@nigerian-prince.ng",
]

_SUBJECTS = {
    "customer_complaint": [
        "Your service is completely unacceptable",
        "URGENT: Product broken after 2 days",
        "Third time contacting you - still no resolution",
        "Extremely disappointed with recent purchase",
        "Formal complaint - Case #{case_id}",
        "Why hasn't my issue been resolved?",
    ],
    "billing_inquiry": [
        "Incorrect charge on my invoice #{inv}",
        "Question about my subscription renewal",
        "Need receipt for order #{ord}",
        "Double charged - please refund",
        "Billing discrepancy in account #{acc}",
    ],
    "technical_support": [
        "Cannot login to my account",
        "API returning 500 errors since yesterday",
        "Integration broken after your update",
        "Dashboard not loading - urgent",
        "Feature X stopped working - bug?",
        "SSL certificate error on your site",
    ],
    "sales_inquiry": [
        "Interested in enterprise plan",
        "Pricing for 500 user seats?",
        "Demo request for Q2 budget approval",
        "Partnership proposal - let's connect",
        "Evaluating your product vs competitors",
    ],
    "spam": [
        "You've WON $1,000,000!!! Claim now",
        "SPECIAL OFFER: Cheap meds online",
        "Make money from home today!!!",
        "Your account needs immediate verification",
        "Exclusive investment opportunity",
    ],
    "internal": [
        "Team sync notes - please review",
        "Q{q} OKR draft for feedback",
        "Reminder: all-hands Friday 3pm",
        "Action items from yesterday's sprint",
        "Updated vacation policy - please read",
    ],
    "other": [
        "Following up on our conversation",
        "Quick question about your product",
        "Collaboration idea",
        "Press inquiry - interview request",
        "Research survey participation",
    ],
}

_BODIES = {
    "customer_complaint": [
        "I purchased your product on {date} and it has been nothing but problems. "
        "This is my {n}rd time reaching out and I have yet to receive any resolution. "
        "If this is not fixed by end of week, I will be disputing the charge and leaving "
        "a public review. My order number is #{order}.",
        "Your customer service has been absolutely terrible. I've been waiting {n} days "
        "for a response. This is completely unacceptable for a paid customer. "
        "I expect an immediate callback and full refund.",
    ],
    "billing_inquiry": [
        "Hi, I noticed a charge of ${amount} on my last invoice dated {date} that I "
        "don't recognize. My account number is #{acc}. Could you please clarify what "
        "this charge is for and issue a corrected invoice if it's an error?",
        "I was charged twice for my monthly subscription this billing cycle. "
        "Please process a refund for the duplicate charge of ${amount} as soon as possible.",
    ],
    "technical_support": [
        "Hi support team, since your latest update on {date} our integration has been "
        "completely broken. We're getting 500 errors on every API call. This is blocking "
        "{n} of our users. Ticket priority: CRITICAL. Please advise immediately.",
        "I'm unable to log in to my account. The password reset email never arrives. "
        "I've tried {n} times. My username is {email}. Please help urgently.",
    ],
    "sales_inquiry": [
        "Hello, I'm evaluating solutions for our team of {n} users. We're particularly "
        "interested in the enterprise tier. Could you send me pricing information and "
        "arrange a demo for next week? Our budget decision is due {date}.",
        "We're a {n}-person startup looking for a scalable solution. Your product was "
        "recommended by a colleague. Can we schedule a 30-minute call this week?",
    ],
    "spam": [
        "CONGRATULATIONS! You have been selected to receive $1,000,000. "
        "Click here to claim your prize NOW: http://scam.xyz/claim",
        "Dear Friend, I am a Nigerian prince and I need your help transferring "
        "$15 million USD. You will receive 30% commission. Reply immediately.",
    ],
    "internal": [
        "Hi team, please find attached the notes from today's sync. "
        "Key action items: 1) Update the roadmap doc 2) Review Q{q} metrics 3) Confirm headcount.",
        "Friendly reminder that all-hands is this Friday at 3pm. Please add agenda items "
        "to the shared doc by Thursday EOD.",
    ],
    "other": [
        "Hi, I came across your product and had a quick question about {topic}. "
        "Would love to connect if you have 15 minutes.",
        "I'm a journalist writing about {topic} for TechCrunch. "
        "Would anyone be available for a brief interview?",
    ],
}

_SENTIMENTS = {
    "customer_complaint": ["negative", "angry"],
    "billing_inquiry": ["neutral", "negative"],
    "technical_support": ["neutral", "negative", "angry"],
    "sales_inquiry": ["positive", "neutral"],
    "spam": ["neutral"],
    "internal": ["neutral", "positive"],
    "other": ["neutral", "positive"],
}

_PRIORITY_WEIGHTS = {
    "customer_complaint": {"urgent": 0.3, "high": 0.4, "medium": 0.2, "low": 0.1},
    "billing_inquiry":    {"urgent": 0.1, "high": 0.3, "medium": 0.4, "low": 0.2},
    "technical_support":  {"urgent": 0.4, "high": 0.35, "medium": 0.2, "low": 0.05},
    "sales_inquiry":      {"urgent": 0.05, "high": 0.2, "medium": 0.5, "low": 0.25},
    "spam":               {"urgent": 0.0, "high": 0.0, "medium": 0.05, "low": 0.95},
    "internal":           {"urgent": 0.05, "high": 0.15, "medium": 0.5, "low": 0.3},
    "other":              {"urgent": 0.02, "high": 0.1, "medium": 0.5, "low": 0.38},
}

_SLA = {"urgent": 1.0, "high": 4.0, "medium": 24.0, "low": 72.0}

_REQUIRES_RESPONSE = {
    "customer_complaint": True, "billing_inquiry": True,
    "technical_support": True, "sales_inquiry": True,
    "spam": False, "internal": False, "other": True,
}

# Response quality keywords by category
_GOOD_RESPONSE_KEYWORDS = {
    "customer_complaint": ["apologize", "sorry", "understand", "resolve", "refund",
                           "escalate", "priority", "follow up", "compensation"],
    "billing_inquiry":    ["invoice", "charge", "refund", "billing", "account",
                           "receipt", "clarify", "process", "days"],
    "technical_support":  ["issue", "investigate", "team", "fix", "update",
                           "workaround", "patch", "ticket", "resolve"],
    "sales_inquiry":      ["demo", "pricing", "schedule", "enterprise", "team",
                           "call", "connect", "proposal", "plan"],
    "other":              ["thank", "happy", "help", "contact", "follow"],
}


# ---------------------------------------------------------------------------
# Email generator
# ---------------------------------------------------------------------------

class EmailGenerator:
    def __init__(self, rng: random.Random, vip_ratio: float = 0.1, spam_ratio: float = 0.15):
        self.rng = rng
        self.vip_ratio = vip_ratio
        self.spam_ratio = spam_ratio
        self._counter = 0

    def _pick_weighted(self, weights: dict) -> str:
        keys = list(weights.keys())
        vals = [weights[k] for k in keys]
        return self.rng.choices(keys, vals)[0]

    def generate(self, now: datetime) -> Email:
        self._counter += 1
        eid = f"email_{self._counter:04d}"

        is_spam = self.rng.random() < self.spam_ratio
        is_vip  = (not is_spam) and self.rng.random() < self.vip_ratio

        if is_spam:
            category = "spam"
        else:
            categories = [c for c in _SUBJECTS if c != "spam"]
            category = self.rng.choice(categories)

        sender = (
            self.rng.choice([s for s in _SENDERS if "vip" in s]) if is_vip
            else self.rng.choice([s for s in _SENDERS if "spam" not in s and "vip" not in s])
            if not is_spam
            else self.rng.choice([s for s in _SENDERS if "spam" in s or "discount" in s or "prince" in s])
        )

        subject_template = self.rng.choice(_SUBJECTS[category])
        subject = subject_template.format(
            case_id=self.rng.randint(1000, 9999),
            inv=self.rng.randint(10000, 99999),
            ord=self.rng.randint(10000, 99999),
            acc=self.rng.randint(10000, 99999),
            q=self.rng.randint(1, 4),
        )

        body_template = self.rng.choice(_BODIES[category])
        body = body_template.format(
            date=(now - timedelta(days=self.rng.randint(1, 10))).strftime("%Y-%m-%d"),
            n=self.rng.randint(2, 7),
            order=self.rng.randint(10000, 99999),
            amount=round(self.rng.uniform(9.99, 499.99), 2),
            acc=self.rng.randint(10000, 99999),
            email=sender,
            q=self.rng.randint(1, 4),
            topic=self.rng.choice(["AI", "cloud infrastructure", "data privacy", "SaaS pricing"]),
        )

        priority = self._pick_weighted(_PRIORITY_WEIGHTS[category])
        if is_vip and priority not in ("urgent", "high"):
            priority = "high"

        sla_hrs = _SLA[priority]
        # Simulate some emails already partially consumed on SLA
        sla_remaining = self.rng.uniform(sla_hrs * 0.1, sla_hrs)

        sentiment = self.rng.choice(_SENTIMENTS[category])
        if is_vip and category == "customer_complaint":
            sentiment = "angry"

        ts_offset = timedelta(minutes=self.rng.randint(1, 120))
        timestamp = (now - ts_offset).isoformat()

        return Email(
            id=eid,
            sender=sender,
            subject=subject,
            body=body,
            timestamp=timestamp,
            priority=priority,
            category=category,
            requires_response=_REQUIRES_RESPONSE[category],
            sla_hours=round(sla_remaining, 2),
            sentiment=sentiment,
        )


# ---------------------------------------------------------------------------
# Response evaluator (heuristic)
# ---------------------------------------------------------------------------

class ResponseEvaluator:
    """Score a response_text against the email being responded to (0.0–1.0)."""

    MIN_GOOD_LENGTH = 60
    MAX_PENALISED_LENGTH = 50

    def score(self, email: Email, response_text: str) -> float:
        if not response_text or len(response_text.strip()) < 10:
            return 0.0

        text_lower = response_text.lower()
        score = 0.0

        # Length heuristic (20%)
        length = len(response_text.strip())
        if length >= self.MIN_GOOD_LENGTH:
            score += 0.20
        elif length >= self.MAX_PENALISED_LENGTH:
            score += 0.10

        # Keyword presence (50%)
        keywords = _GOOD_RESPONSE_KEYWORDS.get(email.category, _GOOD_RESPONSE_KEYWORDS["other"])
        hits = sum(1 for kw in keywords if kw in text_lower)
        keyword_score = min(hits / max(len(keywords) * 0.4, 1), 1.0) * 0.50
        score += keyword_score

        # Personalisation: mentions sender name or email id (10%)
        sender_name = email.sender.split("@")[0].split(".")[0]
        if sender_name in text_lower or email.id in text_lower:
            score += 0.10

        # Empathy for complaints (10%)
        if email.category == "customer_complaint" or email.sentiment in ("negative", "angry"):
            empathy_words = ["sorry", "apologize", "understand", "frustrat", "concern"]
            if any(w in text_lower for w in empathy_words):
                score += 0.10
        else:
            score += 0.10  # Not required; grant full credit

        # Not a canned/empty template (10%)
        filler = ["lorem ipsum", "placeholder", "your text here", "response goes here"]
        if not any(f in text_lower for f in filler):
            score += 0.10

        return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class TriageEnvironment:
    """
    Email Triage & Response OpenEnv environment.

    Tasks:
      easy_classification      — Correctly classify and triage 15 emails.
      medium_sla_pressure      — Handle 25 emails under tight SLA constraints.
      hard_angry_vip           — Handle 35 mixed emails including angry VIP customers.
    """

    TASKS = {
        "easy_classification": {
            "max_steps": 30,
            "inbox_size": 15,
            "vip_ratio": 0.05,
            "spam_ratio": 0.20,
            "sla_multiplier": 2.0,      # relaxed SLA
            "description": "Classify and triage 15 emails with ample time. Focus on correct categorisation.",
        },
        "medium_sla_pressure": {
            "max_steps": 50,
            "inbox_size": 25,
            "vip_ratio": 0.10,
            "spam_ratio": 0.15,
            "sla_multiplier": 0.7,      # tight SLAs
            "description": "Handle 25 emails with tight SLA deadlines. Prioritise urgent emails.",
        },
        "hard_angry_vip": {
            "max_steps": 70,
            "inbox_size": 35,
            "vip_ratio": 0.20,
            "spam_ratio": 0.10,
            "sla_multiplier": 0.5,      # very tight SLAs, many VIPs
            "description": "Handle 35 emails: many VIP customers, angry complaints, tight SLAs.",
        },
    }

    def __init__(self):
        self._state: Optional[EnvironmentState] = None
        self._inbox: List[Email] = []
        self._evaluator = ResponseEvaluator()
        self._rng: Optional[random.Random] = None
        self._task_cfg: Optional[dict] = None
        self._now = datetime(2025, 6, 1, 9, 0, 0)
        self._recent_actions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy_classification", seed: int = 42) -> TriageObservation:
        if task_id not in self.TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(self.TASKS)}")

        self._rng = random.Random(seed)
        self._task_cfg = self.TASKS[task_id]
        cfg = self._task_cfg

        gen = EmailGenerator(self._rng, vip_ratio=cfg["vip_ratio"], spam_ratio=cfg["spam_ratio"])
        self._inbox = [gen.generate(self._now) for _ in range(cfg["inbox_size"])]

        # Apply SLA multiplier
        for email in self._inbox:
            email.sla_hours = round(email.sla_hours * cfg["sla_multiplier"], 2)

        perf = AgentPerformance(
            correct_classifications=0, incorrect_classifications=0,
            responses_sent=0, avg_response_quality=0.0,
            sla_breaches=0, sla_met=0,
            escalations_appropriate=0, escalations_unnecessary=0,
            spam_correctly_archived=0, spam_missed=0,
        )

        self._state = EnvironmentState(
            task_id=task_id, seed=seed,
            step_num=0, max_steps=cfg["max_steps"],
            inbox=self._inbox,
            current_email_index=0,
            handled_email_ids=[],
            performance=perf,
            inbox_metrics=InboxMetrics(
                total_emails=len(self._inbox),
                unhandled=len(self._inbox),
                overdue=0,
                near_sla=0,
                responded=0,
                escalated=0,
                archived=0,
            ),
            total_reward=0.0,
            done=False,
            time_elapsed_minutes=0.0,
        )
        self._state.inbox_metrics = self._compute_inbox_metrics()
        self._recent_actions = []
        return self._make_observation(reward=None)

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        s = self._state
        reward, info = self._compute_reward(action)
        self._apply_action(action)

        s.step_num += 1
        s.time_elapsed_minutes += self._rng.uniform(2.0, 8.0)

        # Advance clock: reduce SLA on unhandled emails
        minutes_passed = self._rng.uniform(2.0, 8.0)
        for email in s.inbox:
            if email.id not in s.handled_email_ids:
                email.sla_hours = max(0.0, email.sla_hours - minutes_passed / 60.0)

        s.inbox_metrics = self._compute_inbox_metrics()
        s.total_reward += reward

        # Episode ends when inbox exhausted OR max_steps reached
        s.done = (
            s.current_email_index >= len(s.inbox)
            or s.step_num >= s.max_steps
        )

        if s.done:
            terminal_reward = self._terminal_reward()
            reward += terminal_reward
            s.total_reward += terminal_reward
            info["terminal_reward"] = terminal_reward

        obs = self._make_observation(reward=reward)
        return obs, reward, s.done, info

    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return deepcopy(self._state)

    def grade(self) -> GraderResult:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        task_id = self._state.task_id
        return GRADERS[task_id](self._state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_email(self) -> Optional[Email]:
        s = self._state
        if s.current_email_index < len(s.inbox):
            return s.inbox[s.current_email_index]
        return None

    def _compute_inbox_metrics(self) -> InboxMetrics:
        s = self._state
        handled = set(s.handled_email_ids)
        unhandled = [e for e in s.inbox if e.id not in handled]
        overdue = sum(1 for e in unhandled if e.sla_hours <= 0)
        near_sla = sum(1 for e in unhandled if 0 < e.sla_hours <= 1.0)
        perf = s.performance
        return InboxMetrics(
            total_emails=len(s.inbox),
            unhandled=len(unhandled),
            overdue=overdue,
            near_sla=near_sla,
            responded=perf.responses_sent,
            escalated=perf.escalations_appropriate + perf.escalations_unnecessary,
            archived=perf.spam_correctly_archived,
        )

    def _make_observation(self, reward: Optional[float]) -> TriageObservation:
        s = self._state
        return TriageObservation(
            current_email=self._current_email(),
            inbox_metrics=s.inbox_metrics,
            performance=s.performance,
            step_num=s.step_num,
            max_steps=s.max_steps,
            time_elapsed_minutes=s.time_elapsed_minutes,
            reward=reward,
            done=s.done,
            recent_actions=list(self._recent_actions[-3:]),
        )

    def _apply_action(self, action: TriageAction):
        s = self._state
        email = self._get_email_by_id(action.email_id)
        if email is None:
            return  # invalid action, already penalised in reward

        if action.email_id not in s.handled_email_ids:
            s.handled_email_ids.append(action.email_id)

        # Advance to next email if acting on current
        if (s.current_email_index < len(s.inbox)
                and s.inbox[s.current_email_index].id == action.email_id):
            s.current_email_index += 1

        self._recent_actions.append({
            "step": s.step_num,
            "email_id": action.email_id,
            "action_type": action.action_type,
        })

    def _get_email_by_id(self, email_id: str) -> Optional[Email]:
        for e in self._state.inbox:
            if e.id == email_id:
                return e
        return None

    def _compute_reward(self, action: TriageAction) -> Tuple[float, Dict]:
        s = self._state
        info: Dict[str, Any] = {}
        reward = 0.0

        email = self._get_email_by_id(action.email_id)
        if email is None:
            return -2.0, {"error": "invalid_email_id"}

        perf = s.performance

        # --- TRIAGE action ---
        if action.action_type == "triage":
            correct_cat = (action.category == email.category) if action.category else False
            correct_pri = (action.priority == email.priority) if action.priority else False

            if correct_cat:
                perf.correct_classifications += 1
                reward += 1.0
                info["category_correct"] = True
            else:
                perf.incorrect_classifications += 1
                reward -= 0.5
                info["category_correct"] = False

            if correct_pri:
                reward += 0.5
                info["priority_correct"] = True
            else:
                reward -= 0.25
                info["priority_correct"] = False

        # --- RESPOND action ---
        elif action.action_type == "respond":
            if not email.requires_response:
                reward -= 0.5
                info["unnecessary_response"] = True
            else:
                quality = self._evaluator.score(email, action.response_text or "")
                perf.responses_sent += 1
                n = perf.responses_sent
                perf.avg_response_quality = (
                    (perf.avg_response_quality * (n - 1) + quality) / n
                )
                reward += 1.5 * quality
                info["response_quality"] = quality

                # SLA reward/penalty
                if email.sla_hours > 0:
                    perf.sla_met += 1
                    reward += 0.5
                    info["sla_met"] = True
                else:
                    perf.sla_breaches += 1
                    reward -= 1.0
                    info["sla_breach"] = True

                # Sentiment bonus: extra reward for good response to angry email
                if email.sentiment == "angry" and quality >= 0.6:
                    reward += 0.5
                    info["angry_handled_well"] = True

        # --- ESCALATE action ---
        elif action.action_type == "escalate":
            # Appropriate: urgent complaints, angry sentiment, VIP senders
            is_appropriate = (
                email.priority == "urgent"
                or email.sentiment == "angry"
                or "vip" in email.sender
                or (email.category == "customer_complaint" and email.priority == "high")
            )
            if is_appropriate:
                perf.escalations_appropriate += 1
                reward += 1.0
                info["escalation_appropriate"] = True
            else:
                perf.escalations_unnecessary += 1
                reward -= 0.75
                info["escalation_appropriate"] = False

        # --- ARCHIVE action ---
        elif action.action_type == "archive":
            if email.category == "spam":
                perf.spam_correctly_archived += 1
                reward += 1.0
                info["spam_archived"] = True
            elif not email.requires_response:
                reward += 0.25
                info["archived_no_response_needed"] = True
            else:
                # Archiving something that needed a response
                perf.spam_missed += 1
                reward -= 1.0
                info["wrongly_archived"] = True

        # --- SKIP action ---
        elif action.action_type == "skip":
            # Penalise skipping urgent/near-SLA emails
            if email.sla_hours <= 1.0 and email.requires_response:
                reward -= 1.5
                info["skipped_near_sla"] = True
            elif email.priority in ("urgent", "high"):
                reward -= 0.5
                info["skipped_high_priority"] = True
            else:
                reward -= 0.1
                info["skipped"] = True

        # SLA-at-zero penalty at step time
        overdue_count = sum(
            1 for e in s.inbox
            if e.id not in s.handled_email_ids and e.sla_hours <= 0
        )
        reward -= overdue_count * 0.05

        return round(reward, 4), info

    def _terminal_reward(self) -> float:
        s = self._state
        perf = s.performance
        total = len(s.inbox)

        handled_count = len(s.handled_email_ids)
        inbox_completion = handled_count / total if total else 0.0

        # Penalise unhandled urgent emails
        unhandled_urgent = sum(
            1 for e in s.inbox
            if e.id not in s.handled_email_ids and e.priority == "urgent"
        )

        classification_acc = (
            perf.correct_classifications /
            max(perf.correct_classifications + perf.incorrect_classifications, 1)
        )

        terminal = (
            5.0 * inbox_completion
            + 3.0 * classification_acc
            + 2.0 * perf.avg_response_quality
            - 2.0 * unhandled_urgent
            - 1.0 * perf.sla_breaches
        )
        return round(terminal, 4)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _grade_easy_classification(state: EnvironmentState) -> GraderResult:
    """
    Easy task grader: correct classification and triage of inbox emails.
    Score = 0.6 * classification_accuracy + 0.4 * inbox_completion
    """
    perf = state.performance
    total = perf.correct_classifications + perf.incorrect_classifications
    acc = perf.correct_classifications / max(total, 1)

    inbox_completion = len(state.handled_email_ids) / max(len(state.inbox), 1)

    score = 0.60 * acc + 0.40 * inbox_completion
    score = round(min(max(score, 0.0), 1.0), 4)

    return GraderResult(
        task_id="easy_classification",
        score=score,
        breakdown={"classification_accuracy": round(acc, 4), "inbox_completion": round(inbox_completion, 4)},
        explanation=(
            f"Classification accuracy: {acc:.1%} ({perf.correct_classifications}/{max(total,1)}). "
            f"Inbox completion: {inbox_completion:.1%} ({len(state.handled_email_ids)}/{len(state.inbox)})."
        ),
        passed=score >= 0.60,
    )


def _grade_medium_sla_pressure(state: EnvironmentState) -> GraderResult:
    """
    Medium task grader: SLA compliance + response quality + classification.
    Score = 0.40 * sla_score + 0.35 * response_quality + 0.25 * classification_accuracy
    """
    perf = state.performance
    sla_total = perf.sla_met + perf.sla_breaches
    sla_score = perf.sla_met / max(sla_total, 1)

    total_class = perf.correct_classifications + perf.incorrect_classifications
    acc = perf.correct_classifications / max(total_class, 1)

    rq = perf.avg_response_quality

    score = 0.40 * sla_score + 0.35 * rq + 0.25 * acc
    score = round(min(max(score, 0.0), 1.0), 4)

    return GraderResult(
        task_id="medium_sla_pressure",
        score=score,
        breakdown={
            "sla_score": round(sla_score, 4),
            "response_quality": round(rq, 4),
            "classification_accuracy": round(acc, 4),
        },
        explanation=(
            f"SLA compliance: {sla_score:.1%} ({perf.sla_met} met, {perf.sla_breaches} breached). "
            f"Avg response quality: {rq:.2f}. "
            f"Classification accuracy: {acc:.1%}."
        ),
        passed=score >= 0.55,
    )


def _grade_hard_angry_vip(state: EnvironmentState) -> GraderResult:
    """
    Hard task grader: escalations, spam handling, response quality for VIP/angry emails.
    Score = 0.30 * escalation_score + 0.25 * spam_score + 0.25 * response_quality
          + 0.20 * sla_score
    """
    perf = state.performance

    # Escalation score
    esc_total = perf.escalations_appropriate + perf.escalations_unnecessary
    esc_score = perf.escalations_appropriate / max(esc_total, 1) if esc_total > 0 else 0.5

    # Spam score
    spam_emails = [e for e in state.inbox if e.category == "spam"]
    spam_score = perf.spam_correctly_archived / max(len(spam_emails), 1) if spam_emails else 1.0

    # Response quality
    rq = perf.avg_response_quality

    # SLA score
    sla_total = perf.sla_met + perf.sla_breaches
    sla_score = perf.sla_met / max(sla_total, 1)

    score = 0.30 * esc_score + 0.25 * spam_score + 0.25 * rq + 0.20 * sla_score
    score = round(min(max(score, 0.0), 1.0), 4)

    return GraderResult(
        task_id="hard_angry_vip",
        score=score,
        breakdown={
            "escalation_score": round(esc_score, 4),
            "spam_score": round(spam_score, 4),
            "response_quality": round(rq, 4),
            "sla_score": round(sla_score, 4),
        },
        explanation=(
            f"Escalation accuracy: {esc_score:.1%} ({perf.escalations_appropriate} appropriate, "
            f"{perf.escalations_unnecessary} unnecessary). "
            f"Spam handling: {spam_score:.1%}. "
            f"Avg response quality: {rq:.2f}. "
            f"SLA compliance: {sla_score:.1%}."
        ),
        passed=score >= 0.50,
    )


GRADERS = {
    "easy_classification": _grade_easy_classification,
    "medium_sla_pressure": _grade_medium_sla_pressure,
    "hard_angry_vip": _grade_hard_angry_vip,
}