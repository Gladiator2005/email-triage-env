"""
app.py — FastAPI server for the Email Triage & Response OpenEnv environment.

Endpoints:
  GET  /health          → liveness check
  POST /reset           → start new episode
  POST /step            → take one action
  GET  /state           → inspect current state
  POST /grade           → score the current episode
  GET  /tasks           → list available tasks
  WebSocket /ws         → streaming interface
"""

from __future__ import annotations

import json
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/health")
def health():
    return {"status": "ok", "environment": "email-triage-env", "version": "1.0.0"}


@app.post("/reset", response_model=TriageObservation)
def reset(req: ResetRequest):
    try:
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
                # action_field is a dict representing TriageAction
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
