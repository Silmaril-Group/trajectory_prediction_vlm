"""Duck Hunt OpenEnv FastAPI Server"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import DuckHuntEnvironment


app = FastAPI(title="Duck Hunt OpenEnv", version="1.0.0")

# Global environment instance
env = DuckHuntEnvironment()


class ActionRequest(BaseModel):
    x: int
    y: int
    horizon: int
    confidence: str | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", environment="duck_hunt")


@app.post("/reset")
def reset():
    """Reset the environment and return initial observation."""
    observation = env.reset()
    return observation


@app.post("/step")
def step(action: ActionRequest):
    """Execute action and return observation."""
    action_dict = {
        "x": action.x,
        "y": action.y,
        "horizon": action.horizon,
        "confidence": action.confidence,
    }
    observation = env.step(action_dict)
    return observation
