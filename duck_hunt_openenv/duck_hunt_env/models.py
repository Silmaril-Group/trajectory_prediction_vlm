"""Duck Hunt OpenEnv Models"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ShootAction:
    """Action to shoot at a target location."""

    x: float  # Target x coordinate (0.0-1.0 normalized)
    y: float  # Target y coordinate (0.0-1.0 normalized)
    horizon: int  # Frames to wait before shot (0-30)
    confidence: Literal["high", "medium", "low"] | None = None

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "horizon": self.horizon,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShootAction":
        return cls(
            x=data["x"],
            y=data["y"],
            horizon=data["horizon"],
            confidence=data.get("confidence"),
        )


@dataclass
class DuckHuntObservation:
    """Observation returned from the environment."""

    # Visual data
    frames: list[str] = field(default_factory=list)  # base64 PNG images
    num_frames: int = 0  # Set by environment

    # Game progress
    round_number: int = 1  # Current round (1, 2, 3...)
    match_number: int = 1  # Current match (1-5)

    # Current match state
    ducks_flying: int = 0  # 0, 1, or 2
    bullets_remaining: int = 3  # 0-3
    match_ducks_hit: int = 0  # 0-2

    # Round/game progress
    round_ducks_hit: int = 0  # 0-10
    total_misses: int = 0  # Towards game over

    # Hardware simulation
    processing_latency_ms: int = 0  # Simulated VLM latency

    # Feedback from last action
    last_action_result: Literal["hit", "miss", "double_kill"] | None = None
    last_ducks_hit: int = 0  # 0, 1, or 2

    # Standard
    reward: float = 0.0
    done: bool = False

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "num_frames": self.num_frames,
            "round_number": self.round_number,
            "match_number": self.match_number,
            "ducks_flying": self.ducks_flying,
            "bullets_remaining": self.bullets_remaining,
            "match_ducks_hit": self.match_ducks_hit,
            "round_ducks_hit": self.round_ducks_hit,
            "total_misses": self.total_misses,
            "processing_latency_ms": self.processing_latency_ms,
            "last_action_result": self.last_action_result,
            "last_ducks_hit": self.last_ducks_hit,
            "reward": self.reward,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DuckHuntObservation":
        return cls(
            frames=data.get("frames", []),
            num_frames=data.get("num_frames", 0),
            round_number=data.get("round_number", 1),
            match_number=data.get("match_number", 1),
            ducks_flying=data.get("ducks_flying", 0),
            bullets_remaining=data.get("bullets_remaining", 3),
            match_ducks_hit=data.get("match_ducks_hit", 0),
            round_ducks_hit=data.get("round_ducks_hit", 0),
            total_misses=data.get("total_misses", 0),
            processing_latency_ms=data.get("processing_latency_ms", 0),
            last_action_result=data.get("last_action_result"),
            last_ducks_hit=data.get("last_ducks_hit", 0),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
        )
