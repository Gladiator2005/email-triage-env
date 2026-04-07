"""
inference.py — Baseline evaluation script for the Email Triage & Response OpenEnv.

Uses the OpenAI client (pointed at API_BASE_URL) to run an LLM agent against all 3 tasks.
Falls back to a deterministic rule-based agent if no API key is available.

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-your-key-here"
  python inference.py

Environment variables:
  API_BASE_URL  — LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (default: gpt-4o-mini)
  HF_TOKEN      — API key (falls back to rule-based agent if unset)
  ENV_URL       — URL of the running environment server (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import requests

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

SEED         = 42
TASKS        = ["easy_classification", "medium_sla_pressure", "hard_angry_vip"]

# ─── OpenAI client setup ──────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _llm_client: Optional[OpenAI] = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None
except ImportError:
    _llm_client = None
    print("[WARN] openai package not installed. Using rule-based agent.", flush=True)


# ─── Helper: call the environment server ─────────────────────────────────────

def _post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{ENV_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    resp = requests.get(f"{ENV_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── Rule-based agent (deterministic fallback) ───────────────────────────────

_CANNED_RESPONSES = {
    "customer_complaint": (
        "Dear {name},\n\nThank you for reaching out. I sincerely apologize for the "
        "inconvenience you've experienced. I completely understand your frustration and "
        "want to assure you that we take this very seriously.\n\n"
        "I have escalated your case to our priority resolution team and we will follow up "
        "within 2 business hours with a concrete resolution plan. If you have a preferred "
        "contact method, please let us know.\n\nWarm regards,\nSupport Team"
    ),
    "billing_inquiry": (
        "Dear {name},\n\nThank you for contacting us about your billing inquiry. "
        "I have reviewed your account and will process the necessary corrections. "
        "A corrected invoice and/or refund will be issued within 3–5 business days.\n\n"
        "Please reference your account for any follow-up questions.\n\nBest regards,\nBilling Team"
    ),
    "technical_support": (
        "Dear {name},\n\nThank you for reporting this issue. Our engineering team has been "
        "notified and is actively investigating. We have opened a priority ticket for your case.\n\n"
        "As a workaround, please try clearing your cache and retrying. We will provide a "
        "status update within 1 hour.\n\nBest regards,\nTechnical Support"
    ),
    "sales_inquiry": (
        "Dear {name},\n\nThank you for your interest in our enterprise offering! "
        "I'd be happy to schedule a demo and walk you through our pricing plans.\n\n"
        "Please let me know your availability for a 30-minute call this week and I will "
        "send a calendar invite. Looking forward to connecting!\n\nBest regards,\nSales Team"
    ),
    "other": (
        "Dear {name},\n\nThank you for reaching out. I am happy to help with your inquiry. "
        "Please allow us 1–2 business days to review and respond with more details.\n\n"
        "Best regards,\nSupport Team"
    ),
}


def _rule_based_action(obs: dict) -> dict:
    """Deterministic rule-based agent."""
    email = obs.get("current_email")
    if not email:
        return {"action_type": "skip", "email_id": "none"}

    eid = email["id"]
    category = email.get("category", "other")
    priority = email.get("priority", "medium")
    sender = email.get("sender", "Customer")
    name = sender.split("@")[0].replace(".", " ").title()
    requires_response = email.get("requires_response", True)
    sentiment = email.get("sentiment", "neutral")
    is_vip = "vip" in sender.lower()

    # 1. Archive spam
    if category == "spam":
        return {"action_type": "archive", "email_id": eid}

    # 2. Escalate angry VIP or urgent complaints
    should_escalate = (
        is_vip
        or sentiment == "angry"
        or (priority == "urgent" and category == "customer_complaint")
    )
    if should_escalate:
        return {
            "action_type": "escalate",
            "email_id": eid,
            "escalation_reason": f"High-priority {category} from {sender}. Sentiment: {sentiment}.",
        }

    # 3. Respond if requires response
    if requires_response:
        template = _CANNED_RESPONSES.get(category, _CANNED_RESPONSES["other"])
        response_text = template.format(name=name)
        return {
            "action_type": "respond",
            "email_id": eid,
            "response_text": response_text,
        }

    # 4. Triage internal / other
    return {
        "action_type": "triage",
        "email_id": eid,
        "priority": priority,
        "category": category,
        "note": "Triaged by rule-based agent.",
    }


# ─── LLM agent ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent for a B2B SaaS company.
Your job is to process incoming emails efficiently: classify, respond, escalate, or archive them.

For each email presented to you, output a JSON action with EXACTLY these fields:
{
  "action_type": "triage" | "respond" | "escalate" | "archive" | "skip",
  "email_id": "<id of the email>",
  "priority": "low" | "medium" | "high" | "urgent"  (only for triage),
  "category": "customer_complaint" | "billing_inquiry" | "technical_support" | "sales_inquiry" | "spam" | "internal" | "other"  (only for triage),
  "response_text": "<your email reply>"  (only for respond, 60-300 chars, be professional),
  "escalation_reason": "<reason>"  (only for escalate)
}

Rules:
- ARCHIVE spam immediately
- ESCALATE: angry VIP customers, urgent complaints, or if you cannot resolve it
- RESPOND: billing, technical, sales inquiries with a clear professional reply
- TRIAGE: internal emails and others — assign correct category and priority
- Never skip high-priority or near-SLA emails

Output ONLY the JSON object, no markdown, no explanation.
"""


def _llm_action(obs: dict) -> dict:
    """LLM-powered agent using OpenAI-compatible client."""
    if _llm_client is None:
        return _rule_based_action(obs)

    email = obs.get("current_email")
    if not email:
        return {"action_type": "skip", "email_id": "none"}

    metrics = obs.get("inbox_metrics", {})

    user_msg = f"""
CURRENT EMAIL:
ID: {email['id']}
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
SLA remaining: {email['sla_hours']} hours
Current priority: {email['priority']}
Current category: {email['category']}
Requires response: {email['requires_response']}
Sentiment: {email['sentiment']}

INBOX STATUS:
Unhandled: {metrics.get('unhandled', '?')} | Overdue: {metrics.get('overdue', '?')} | Near-SLA: {metrics.get('near_sla', '?')}

Step {obs.get('step_num', 0)} / {obs.get('max_steps', '?')}

Decide your action for email {email['id']}:
""".strip()

    try:
        response = _llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw)
        if "email_id" not in action:
            action["email_id"] = email["id"]
        return action
    except Exception as exc:
        print(f"[WARN] LLM call failed ({exc}), falling back to rule-based.", flush=True)
        return _rule_based_action(obs)


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool = True) -> dict:
    print(f"[START] task={task_id}", flush=True)

    obs = _post("/reset", {"task_id": task_id, "seed": SEED})
    total_reward = 0.0
    step = 0

    while not obs.get("done", False):
        action = _llm_action(obs) if use_llm else _rule_based_action(obs)

        email = obs.get("current_email")
        if email and "email_id" not in action:
            action["email_id"] = email["id"]

        try:
            result = _post("/step", action)
        except requests.HTTPError as e:
            print(f"[WARN] step error: {e} — action={action}", flush=True)
            if email:
                _post("/step", {"action_type": "skip", "email_id": email["id"]})
            break

        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward
        step += 1

        print(
            f"[STEP] step={step} reward={reward:+.4f} action={action.get('action_type', '?')} "
            f"email_id={action.get('email_id', '?')}",
            flush=True,
        )

    # Grade
    grade_result = _post("/grade", {})
    score = grade_result["score"]
    passed = grade_result["passed"]
    breakdown = grade_result["breakdown"]

    print(
        f"[END] task={task_id} score={score:.4f} steps={step} passed={passed}",
        flush=True,
    )
    print(f"[BREAKDOWN] {json.dumps(breakdown)}", flush=True)
    print(f"[EXPLANATION] {grade_result['explanation']}", flush=True)
    print(f"[TOTAL_REWARD] {total_reward:.4f}", flush=True)

    return {"task_id": task_id, "score": score, "passed": passed, "breakdown": breakdown}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_llm = bool(_llm_client)
    agent_label = f"LLM ({MODEL_NAME})" if use_llm else "Rule-based (no HF_TOKEN)"

    print(f"[INFO] Email Triage & Response OpenEnv — Baseline Inference", flush=True)
    print(f"[INFO] agent={agent_label} env={ENV_URL} seed={SEED}", flush=True)

    # Health check
    try:
        health = _get("/health")
        print(f"[INFO] health={health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    results = {}
    start = time.time()

    for task_id in TASKS:
        result = run_episode(task_id, use_llm=use_llm)
        results[task_id] = result["score"]

    elapsed = time.time() - start
    avg_score = sum(results.values()) / len(results)

    print(f"\n[RESULTS]", flush=True)
    for tid, score in results.items():
        print(f"  {tid:<30s}  {score:.4f}", flush=True)
    print(f"  {'AVERAGE':<30s}  {avg_score:.4f}", flush=True)

    output = {
        "scores": results,
        "average": round(avg_score, 4),
        "agent": agent_label,
        "seed": SEED,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(output, indent=2), flush=True)
    return output


if __name__ == "__main__":
    main()