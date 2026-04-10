from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from model_router_openenv.environment import ModelRouterEnv
from model_router_openenv.actions import Action
import uuid

app = FastAPI(title="Model Router OpenEnv API")

# Simple in-memory session manager
sessions = {}

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_001"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.post("/reset")
def reset_env(req: ResetRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        env = ModelRouterEnv(task_id=req.task_id)
        obs = env.reset()
        sessions[session_id] = env
        return {
            "session_id": session_id,
            "observation": obs.model_dump()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error during reset: {str(e)}")

@app.post("/step")
def step_env(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = sessions[req.session_id]
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action or state error: {str(e)}")

@app.get("/state")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    env = sessions[session_id]
    try:
        return {
            "session_id": session_id,
            "observation": env._get_obs().model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch state: {str(e)}")
