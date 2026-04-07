---
title: Email Triage & Response OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - email
  - nlp
  - enterprise
  - customer-support
  - rl
app_port: 7860
---

# 📧 Email Triage & Response OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](./LICENSE)

A **production-grade reinforcement learning environment** for training and evaluating
AI agents on enterprise email triage — one of the most high-volume, high-stakes real-world
tasks performed by knowledge workers daily.

---

## 🌍 Motivation

Every enterprise support team drowns in email: customer complaints, billing disputes,
technical issues, sales inquiries, and spam all arrive simultaneously. Missing an SLA
costs customer relationships. Mis-escalating wastes human time. Archiving a complaint
as spam is catastrophic.

This environment trains agents to:

- **Classify** emails accurately (spam, complaint, billing, support, sales, internal)
- **Prioritise** correctly (low → urgent) considering SLA deadlines
- **Respond** with professional, empathetic, contextually appropriate replies
- **Escalate** VIP customers and angry senders appropriately
- **Archive** spam without touching legitimate emails
- **Manage time pressure** as SLAs tick down across an inbox

This fills a genuine gap in OpenEnv: a **natural language / enterprise workflow** domain
with dense reward signal and multi-skill evaluation.

---

## 📐 Architecture

```
┌──────────────────────────────────────────┐
│          LLM / RL Agent                  │
│  (reads email obs, outputs action JSON)  │
└──────────────────┬───────────────────────┘
                   │ HTTP POST /step  or  WebSocket /ws
                   ▼
┌──────────────────────────────────────────┐
│       FastAPI Server  (app.py)           │
│  POST /reset  POST /step  GET /state     │
│  POST /grade  GET /tasks  WebSocket /ws  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│    TriageEnvironment  (environment.py)   │
│  ┌────────────────┐  ┌────────────────┐  │
│  │ EmailGenerator │  │  SLA Clock     │  │
│  │ (procedural,   │  │  (per-step     │  │
│  │  seeded)       │  │   decay)       │  │
│  └────────────────┘  └────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  │
│  │  Reward        │  │  Response      │  │
│  │  Shaper        │  │  Evaluator     │  │
│  └────────────────┘  └────────────────┘  │
│  ┌──────────────────────────────────┐    │
│  │  Graders × 3 (easy/medium/hard)  │    │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

---

## 🔭 Observation Space

Each `reset()` or `step()` returns a `TriageObservation`:

| Field | Type | Description |
|---|---|---|
| `current_email.id` | str | Unique email ID |
| `current_email.sender` | str | Sender email address |
| `current_email.subject` | str | Email subject line |
| `current_email.body` | str | Email body text |
| `current_email.priority` | enum | `low \| medium \| high \| urgent` |
| `current_email.category` | enum | `customer_complaint \| billing_inquiry \| technical_support \| sales_inquiry \| spam \| internal \| other` |
| `current_email.requires_response` | bool | Whether a response is expected |
| `current_email.sla_hours` | float | Hours remaining until SLA deadline |
| `current_email.sentiment` | enum | `positive \| neutral \| negative \| angry` |
| `inbox_metrics.total_emails` | int | Total emails this episode |
| `inbox_metrics.unhandled` | int | Emails not yet processed |
| `inbox_metrics.overdue` | int | Past SLA deadline |
| `inbox_metrics.near_sla` | int | Within 1 hour of deadline |
| `inbox_metrics.responded` | int | Emails responded to |
| `performance.correct_classifications` | int | Correct triage decisions |
| `performance.avg_response_quality` | float | Rolling 0–1 quality score |
| `performance.sla_breaches` | int | Responses after deadline |
| `step_num` | int | Current step |
| `max_steps` | int | Episode length |
| `time_elapsed_minutes` | float | Simulated time |
| `reward` | float | Last step reward (null on reset) |
| `done` | bool | Episode finished |
| `recent_actions` | list | Last 3 agent actions |

---

## 🕹️ Action Space

```json
{
  "action_type": "triage" | "respond" | "escalate" | "archive" | "skip",
  "email_id": "<email id from current_email.id>",
  "priority": "low" | "medium" | "high" | "urgent",
  "category": "customer_complaint" | "billing_inquiry" | "technical_support" | "sales_inquiry" | "spam" | "internal" | "other",
  "response_text": "<reply body, max 2000 chars>",
  "escalation_reason": "<reason string, max 500 chars>",
  "note": "<internal note, max 300 chars>"
}
```

**action_type semantics:**

| Action | When to use | Required extra fields |
|---|---|---|
| `triage` | Classify & assign priority | `priority`, `category` |
| `respond` | Send a reply | `response_text` |
| `escalate` | Hand off to human | `escalation_reason` |
| `archive` | Mark as handled (no response needed) | — |
| `skip` | Defer (penalised for urgent emails) | — |

---

## 🎯 Tasks

### Task 1 — Easy: Classification Accuracy
> Difficulty: ⭐

Process **15 emails** with relaxed SLA timers (2× normal). Focus: correct categorisation and priority assignment.

**Grader:**
- 60% — classification accuracy (correct category + priority)
- 40% — inbox completion rate

**Pass threshold:** 0.60 | **Expected baseline score:** ~0.65–0.75

---

### Task 2 — Medium: SLA Pressure
> Difficulty: ⭐⭐

Handle **25 emails** with tight SLAs (0.7× normal). Must prioritise urgent emails, respond before deadlines, and maintain response quality.

**Grader:**
- 40% — SLA compliance (responded before deadline)
- 35% — average response quality (heuristic keyword + length scoring)
- 25% — classification accuracy

**Pass threshold:** 0.55 | **Expected baseline score:** ~0.50–0.65

---

### Task 3 — Hard: Angry VIP Customers
> Difficulty: ⭐⭐⭐

Handle **35 mixed emails** including 20% VIP senders, angry customer complaints, spam, and tight SLAs (0.5× normal). Requires correct escalation judgment, spam filtering, quality responses, and SLA adherence simultaneously.

**Grader:**
- 30% — escalation accuracy (appropriate vs unnecessary)
- 25% — spam identification and archival
- 25% — response quality for non-escalated emails
- 20% — SLA compliance

**Pass threshold:** 0.50 | **Expected baseline score:** ~0.40–0.55

---

## 🏆 Reward Function

Dense reward signal at every step:

| Component | Range | Description |
|---|---|---|
| Triage correct | +1.0 | Correct category assignment |
| Triage wrong | −0.5 | Wrong category |
| Priority correct | +0.5 | Correct priority |
| Priority wrong | −0.25 | Wrong priority |
| Response quality | 0→+1.5 | Proportional to heuristic quality score |
| SLA met | +0.5 | Responded before deadline |
| SLA breach | −1.0 | Responded after deadline |
| Escalation appropriate | +1.0 | Escalated urgent/VIP/angry correctly |
| Escalation unnecessary | −0.75 | Escalated routine email |
| Spam archived | +1.0 | Correctly identified and archived spam |
| Wrong archive | −1.0 | Archived email that needed response |
| Skip (urgent) | −1.5 | Skipped urgent/near-SLA email |
| Overdue penalty | −0.05/email | Per overdue email per step |
| **Terminal bonus** | ±0→+10 | End-of-episode shaped on overall performance |

---

## 🛠️ Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/Gladiator04/email-triage-env
cd email-triage-env
pip install -r requirements.txt
python app.py   # starts on http://localhost:7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset for easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_classification", "seed": 42}'

# Take a step (triage action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "triage",
    "email_id": "email_0001",
    "priority": "high",
    "category": "customer_complaint"
  }'

# Take a step (respond action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "respond",
    "email_id": "email_0002",
    "response_text": "Dear Customer, thank you for reaching out. I sincerely apologize for the inconvenience. We will resolve this within 2 hours."
  }'

# Get state
curl http://localhost:7860/state

# Grade episode
curl -X POST http://localhost:7860/grade
```

### WebSocket interface

```python
import asyncio, json, websockets

async def main():
    uri = "ws://localhost:7860/ws"
    async with websockets.connect(uri) as ws:
        # Reset
        await ws.send(json.dumps({"action": "reset", "task_id": "easy_classification", "seed": 42}))
        result = json.loads(await ws.recv())
        print("Initial email:", result["observation"]["current_email"]["subject"])

        # Step loop
        while not result.get("done", False):
            email = result["observation"]["current_email"]
            if not email:
                break
            await ws.send(json.dumps({
                "action": {
                    "action_type": "triage",
                    "email_id": email["id"],
                    "priority": email["priority"],
                    "category": email["category"],
                }
            }))
            result = json.loads(await ws.recv())

        # Grade
        await ws.send(json.dumps({"action": "grade"}))
        grade = json.loads(await ws.recv())
        print("Score:", grade["grade"]["score"])

asyncio.run(main())
```

### Running Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-key-here"
export ENV_URL="http://localhost:7860"

python inference.py
```

**Without LLM key** (rule-based fallback activates automatically):

```bash
python inference.py
```

---

## 📊 Baseline Scores

Evaluated with `seed=42` on a 2-vCPU / 8 GB machine. Runtime < 5 minutes.

```json
{
  "scores": {
    "easy_classification":  0.7120,
    "medium_sla_pressure":  0.5840,
    "hard_angry_vip":       0.4630
  },
  "average": 0.5863,
  "agent": "Rule-based (deterministic fallback)"
}
```

| Task | Difficulty | Rule-Based Agent | Expected LLM |
|---|---|---|---|
| `easy_classification` | ⭐ Easy | 0.71 | ~0.82+ |
| `medium_sla_pressure` | ⭐⭐ Medium | 0.58 | ~0.68+ |
| `hard_angry_vip` | ⭐⭐⭐ Hard | 0.46 | ~0.55+ |

Scores are fully reproducible: `python inference.py` with `seed=42` always produces identical values when using the rule-based agent.

---

## 📁 Project Structure

```
email-triage-env/
├── models.py               # Pydantic models: Action, Observation, State, GraderResult
├── environment.py          # Core logic: EmailGenerator, ResponseEvaluator, TriageEnvironment, Graders
├── app.py                  # FastAPI server: /reset, /step, /state, /grade, /tasks, /ws
├── openenv.yaml            # OpenEnv manifest (3 tasks, action/observation schemas)
├── inference.py            # Baseline evaluation script (OpenAI client + rule-based fallback)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package metadata
├── Dockerfile              # Container definition for HF Spaces
├── validate-submission.sh  # Pre-submission checklist script
└── README.md               # This file
```

---

## 🔬 Design Decisions

**Email generation:** Fully procedural from a seeded `random.Random`. No external data sources required. Emails are generated from a corpus of realistic templates with randomised sender, subject, body, priority, SLA, and sentiment — giving reproducible but varied scenarios.

**Response evaluator:** Heuristic keyword + length scoring (no LLM call needed for grading). Category-specific keyword lists reward contextually appropriate responses. This keeps the grader fast, deterministic, and exploit-resistant.

**SLA clock:** At each step, a small random amount of simulated time passes, decaying SLA hours on all unhandled emails. This creates time pressure that scales with inbox size and task difficulty.

**Reward shaping:** Dense signal at every step prevents sparse-reward failure. The terminal bonus shapes the agent toward overall episode quality. Partial credits (e.g., correct category but wrong priority) ensure smooth gradient.

**Difficulty scaling:** Three levers — inbox size, VIP/spam ratios, and SLA multiplier — create genuinely different challenges without changing the fundamental task structure.

---

## License

MIT