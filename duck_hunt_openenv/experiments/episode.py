"""Episode Runner with Weave Tracking."""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import weave
from PIL import Image

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from environment import DuckHuntEnvironment
from .agent import DuckHuntVLMAgent, base64_to_pil, ShootPrediction


@dataclass
class StepResult:
    """Result of a single step."""
    action: dict
    result: str | None
    reward: float
    frame_sample: Image.Image | None = None  # Sample frame for logging


@dataclass
class EpisodeResult:
    """Result of a full episode."""
    total_reward: float
    total_steps: int
    ducks_hit: int
    total_misses: int
    final_round: int
    final_match: int
    steps: list[StepResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def resize_frame(image: Image.Image, max_size: tuple = (256, 256)) -> Image.Image:
    """Resize frame for efficient logging."""
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


@weave.op
def run_step(
    agent: DuckHuntVLMAgent,
    frames: list[Image.Image],
    game_state: dict,
    env: DuckHuntEnvironment,
) -> tuple[dict, StepResult]:
    """Run a single step: predict and execute action."""
    # Get prediction from VLM
    prediction = agent.predict(frames, game_state)

    # Execute action
    action = {
        "x": prediction.x,
        "y": prediction.y,
        "horizon": prediction.horizon,
        "confidence": prediction.confidence,
    }
    obs = env.step(action)

    # Create step result with sample frame
    step_result = StepResult(
        action=action,
        result=obs["last_action_result"],
        reward=obs["reward"],
        frame_sample=resize_frame(frames[-1]) if frames else None,
    )

    return obs, step_result


@weave.op
def run_episode(
    agent: DuckHuntVLMAgent,
    max_steps: int = 1000,
    log_frames: bool = True,
) -> EpisodeResult:
    """
    Run a full episode with the VLM agent.

    Args:
        agent: The VLM agent to use
        max_steps: Maximum steps before forced termination
        log_frames: Whether to log frame samples

    Returns:
        EpisodeResult with full episode data
    """
    env = DuckHuntEnvironment()
    obs = env.reset()

    episode_result = EpisodeResult(
        total_reward=0.0,
        total_steps=0,
        ducks_hit=0,
        total_misses=0,
        final_round=1,
        final_match=1,
        config={
            "model_name": agent.model_name,
            "temperature": agent.temperature,
        },
    )

    step = 0
    while not obs["done"] and step < max_steps:
        # Convert frames to PIL
        frames = [base64_to_pil(f) for f in obs["frames"]]

        # Build game state for context
        game_state = {
            "round_number": obs["round_number"],
            "match_number": obs["match_number"],
            "ducks_flying": obs["ducks_flying"],
            "bullets_remaining": obs["bullets_remaining"],
            "processing_latency_ms": obs["processing_latency_ms"],
        }

        # Run step
        obs, step_result = run_step(agent, frames, game_state, env)

        # Don't log frames if disabled (saves space)
        if not log_frames:
            step_result.frame_sample = None

        episode_result.steps.append(step_result)
        episode_result.total_reward += step_result.reward
        step += 1

    # Final stats
    episode_result.total_steps = step
    episode_result.ducks_hit = obs["round_ducks_hit"]
    episode_result.total_misses = obs["total_misses"]
    episode_result.final_round = obs["round_number"]
    episode_result.final_match = obs["match_number"]

    return episode_result


@weave.op
def run_episodes(
    agent: DuckHuntVLMAgent,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    log_frames: bool = False,
) -> dict:
    """
    Run multiple episodes and aggregate results.

    Args:
        agent: The VLM agent to use
        num_episodes: Number of episodes to run
        max_steps_per_episode: Max steps per episode
        log_frames: Whether to log frame samples

    Returns:
        Aggregated results dict
    """
    results = []

    for ep in range(num_episodes):
        print(f"Running episode {ep + 1}/{num_episodes}...")
        result = run_episode(agent, max_steps_per_episode, log_frames)
        results.append(result)
        print(f"  Reward: {result.total_reward:.2f}, Ducks hit: {result.ducks_hit}")

    # Aggregate
    total_rewards = [r.total_reward for r in results]
    total_ducks = [r.ducks_hit for r in results]
    total_steps = [r.total_steps for r in results]

    return {
        "num_episodes": num_episodes,
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "min_reward": min(total_rewards),
        "max_reward": max(total_rewards),
        "avg_ducks_hit": sum(total_ducks) / len(total_ducks),
        "avg_steps": sum(total_steps) / len(total_steps),
        "config": {
            "model_name": agent.model_name,
            "temperature": agent.temperature,
        },
        "episodes": [
            {
                "reward": r.total_reward,
                "ducks_hit": r.ducks_hit,
                "steps": r.total_steps,
            }
            for r in results
        ],
    }
