"""
app.py — FastAPI server for the Email Triage & Response OpenEnv environment.
Endpoints:
  GET  /             → environment info (root)
  GET  /health       → liveness check
  POST /reset        → start new episode
  POST /step         → take one action
  GET  /state        → inspect current state
  POST /grade        → score the current episode
  GET  /tasks        → list available tasks
  WebSocket /ws      → streaming interface
"""

from __future__ import annotations

import json
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import TriageEnvironment
from models import TriageAction, TriageObservation, EnvironmentState, GraderResult

app = FastAPI(
    title="Email Triage & Response OpenEnv",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage, "
        "classify, respond, and escalate emails — a real-world enterprise task."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-session environment (sufficient for HF Spaces demo)
env = TriageEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_classification"
    seed: int = 42


class StepResponse(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Triage & Response OpenEnv</title>
        <style>
            body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 40px; }
            h1 { color: #58a6ff; }
            h2 { color: #79c0ff; margin-top: 30px; }
            a { color: #58a6ff; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th { background: #21262d; color: #79c0ff; padding: 8px 12px; text-align: left; }
            td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
            .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
            code { background: #161b22; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>📧 Email Triage &amp; Response OpenEnv</h1>
        <span class="badge">● Running</span> &nbsp; version 1.0.0
        <p>A production-grade RL environment for training AI agents on enterprise email triage —
        classify, respond, escalate, and archive emails under realistic SLA constraints.</p>

        <h2>API Endpoints</h2>
        <table>
            <tr><th>Method</th><th>Path</th><th>Description</th></tr>
            <tr><td>GET</td><td><a href="/health">/health</a></td><td>Liveness check</td></tr>
            <tr><td>POST</td><td>/reset</td><td>Start new episode (body: task_id, seed)</td></tr>
            <tr><td>POST</td><td>/step</td><td>Take one action</td></tr>
            <tr><td>GET</td><td><a href="/state">/state</a></td><td>Inspect current state</td></tr>
            <tr><td>POST</td><td>/grade</td><td>Score the current episode</td></tr>
            <tr><td>GET</td><td><a href="/tasks">/tasks</a></td><td>List available tasks</td></tr>
            <tr><td>GET</td><td><a href="/docs">/docs</a></td><td>Interactive API docs (Swagger)</td></tr>
            <tr><td>WS</td><td>/ws</td><td>WebSocket streaming interface</td></tr>
        </table>

        <h2>Tasks</h2>
        <table>
            <tr><th>ID</th><th>Difficulty</th><th>Inbox Size</th><th>Max Steps</th></tr>
            <tr><td>easy_classification</td><td>⭐ Easy</td><td>15</td><td>30</td></tr>
            <tr><td>medium_sla_pressure</td><td>⭐⭐ Medium</td><td>25</td><td>50</td></tr>
            <tr><td>hard_angry_vip</td><td>⭐⭐⭐ Hard</td><td>35</td><td>70</td></tr>
        </table>

        <h2>Quick Start</h2>
        <pre>
# Reset environment
curl -X POST /reset -H "Content-Type: application/json" \\
     -d '{"task_id": "easy_classification", "seed": 42}'

# Take a step
curl -X POST /step -H "Content-Type: application/json" \\
     -d '{"action_type": "triage", "email_id": "email_0001",
          "priority": "high", "category": "customer_complaint"}'
        </pre>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.post("/reset", response_model=TriageObservation)
def reset(req: Optional[ResetRequest] = None):
    try:
        if req is None:
            req = ResetRequest()
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: TriageAction):
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade", response_model=GraderResult)
def grade():
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "description": cfg["description"],
                "inbox_size": cfg["inbox_size"],
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TriageEnvironment.TASKS.items()
        ]
    }


# ---------------------------------------------------------------------------
# WebSocket streaming interface
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_env = TriageEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            action_field = msg.get("action")

            if action_field == "reset":
                task_id = msg.get("task_id", "easy_classification")
                seed = msg.get("seed", 42)
                obs = ws_env.reset(task_id=task_id, seed=seed)
                await websocket.send_text(json.dumps({
                    "type": "reset",
                    "observation": obs.model_dump(),
                }))

            elif action_field == "grade":
                result = ws_env.grade()
                await websocket.send_text(json.dumps({
                    "type": "grade",
                    "grade": result.model_dump(),
                }))

            elif action_field == "state":
                s = ws_env.state()
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "state": s.model_dump(),
                }))

            else:
                try:
                    action_data = action_field if isinstance(action_field, dict) else msg
                    act = TriageAction(**action_data)
                    obs, reward, done, info = ws_env.step(act)
                    await websocket.send_text(json.dumps({
                        "type": "step",
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))

    except WebSocketDisconnect:
        pass


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
