"""FastAPI stub. Session 1 goal is just proving the service boots and has a
health endpoint — the real routes (/ask etc.) start showing up in Week 2."""
from fastapi import FastAPI

from app.config import settings

app = FastAPI(
    title="AlphaTrace",
    description="Multi-agent, self-verifying equity-research copilot.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict:
    return {"service": "AlphaTrace", "status": "stub — see /health"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
