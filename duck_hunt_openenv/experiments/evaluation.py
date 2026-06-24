"""Evaluation Framework for Duck Hunt VLM Agents."""

import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass

import weave
from weave import Evaluation
from PIL import Image

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from environment import DuckHuntEnvironment
from .agent import DuckHuntVLMAgent, base64_to_pil
from .scorers import (
    accuracy_scorer,
    hit_type_scorer,
    horizon_efficiency_scorer,
    confidence_calibration_scorer,
    combined_scorer,
)


def create_evaluation_dataset(
    num_scenarios: int = 20,
    frames_per_scenario: int = 4,
) -> list[dict]:
    """
    Create evaluation dataset from environment.

    Each scenario captures a moment in the game with frames
    and the game state at that moment.
    """
    env = DuckHuntEnvironment()
    dataset = []

    scenarios_collected = 0
    while scenarios_collected < num_scenarios:
        obs = env.reset()

        # Run a few steps to get varied scenarios
        for step in range(50):
            if obs["done"]:
                break

            # Only collect if ducks are flying
            if obs["ducks_flying"] > 0:
                frames = [base64_to_pil(f) for f in obs["frames"]]

                dataset.append({
                    "scenario_id": f"scenario_{scenarios_collected}",
                    "frames": frames,
                    "game_state": {
                        "round_number": obs["round_number"],
                        "match_number": obs["match_number"],
                        "ducks_flying": obs["ducks_flying"],
                        "bullets_remaining": obs["bullets_remaining"],
                        "processing_latency_ms": obs["processing_latency_ms"],
                    },
                    # Expected behavior hints
                    "has_target": obs["ducks_flying"] > 0,
                })
                scenarios_collected += 1

                if scenarios_collected >= num_scenarios:
                    break

            # Random action to progress game
            import random
            action = {
                "x": random.random(),  # 0.0-1.0 normalized
                "y": random.random(),  # 0.0-1.0 normalized
                "horizon": random.randint(0, 10),
            }
            obs = env.step(action)

    return dataset


@weave.op
def model_predict_wrapper(
    agent: DuckHuntVLMAgent,
    frames: list[Image.Image],
    game_state: dict,
    scenario_id: str,
    has_target: bool,
) -> dict:
    """
    Wrapper to call agent and format output for scoring.
    """
    # Get prediction
    prediction = agent.predict(frames, game_state)

    # For evaluation, we need to actually execute to know result
    # But we can also just evaluate the prediction quality
    return {
        "x": prediction.x,
        "y": prediction.y,
        "horizon": prediction.horizon,
        "confidence": prediction.confidence,
        "scenario_id": scenario_id,
        "has_target": has_target,
    }


class EvaluationRunner:
    """Runner for VLM agent evaluations."""

    def __init__(
        self,
        project_name: str = "duck-hunt-vlm-evaluation",
    ):
        self.project_name = project_name
        weave.init(project_name)

    def create_evaluation(
        self,
        dataset: list[dict],
        name: str = "duck_hunt_eval",
    ) -> Evaluation:
        """Create a Weave Evaluation object."""
        return Evaluation(
            name=name,
            dataset=dataset,
            scorers=[
                hit_type_scorer,
                horizon_efficiency_scorer,
                confidence_calibration_scorer,
            ],
        )

    async def run_evaluation(
        self,
        agent: DuckHuntVLMAgent,
        num_scenarios: int = 20,
    ) -> dict:
        """Run evaluation on agent."""
        print(f"Creating evaluation dataset with {num_scenarios} scenarios...")
        dataset = create_evaluation_dataset(num_scenarios)

        print("Running evaluation...")
        evaluation = self.create_evaluation(dataset)

        # Create model wrapper
        class ModelWrapper(weave.Model):
            agent: DuckHuntVLMAgent

            @weave.op
            def predict(
                self,
                frames: list[Image.Image],
                game_state: dict,
                scenario_id: str,
                has_target: bool,
            ) -> dict:
                prediction = self.agent.predict(frames, game_state)
                return {
                    "x": prediction.x,
                    "y": prediction.y,
                    "horizon": prediction.horizon,
                    "confidence": prediction.confidence,
                }

        wrapper = ModelWrapper(agent=agent)
        results = await evaluation.evaluate(wrapper)

        return results


@weave.op
def run_live_evaluation(
    agent: DuckHuntVLMAgent,
    num_episodes: int = 5,
    max_steps: int = 100,
    name: str | None = None,
) -> dict:
    """
    Run live evaluation where predictions are actually executed.

    This gives real hit/miss results rather than just prediction quality.

    Args:
        agent: The VLM agent to evaluate
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
        name: Optional name for this evaluation run
    """
    eval_name = name or f"live-eval-{num_episodes}ep"
    print(f"Evaluation: {eval_name}")

    env = DuckHuntEnvironment()

    all_results = []

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        obs = env.reset()
        episode_results = []

        step = 0
        while not obs["done"] and step < max_steps:
            frames = [base64_to_pil(f) for f in obs["frames"]]
            game_state = {
                "round_number": obs["round_number"],
                "match_number": obs["match_number"],
                "ducks_flying": obs["ducks_flying"],
                "bullets_remaining": obs["bullets_remaining"],
                "processing_latency_ms": obs["processing_latency_ms"],
            }

            # Get prediction
            prediction = agent.predict(frames, game_state)

            # Execute
            action = {
                "x": prediction.x,
                "y": prediction.y,
                "horizon": prediction.horizon,
                "confidence": prediction.confidence,
            }
            obs = env.step(action)

            # Record result
            result = {
                "x": prediction.x,
                "y": prediction.y,
                "horizon": prediction.horizon,
                "confidence": prediction.confidence,
                "result": obs["last_action_result"],
                "reward": obs["reward"],
            }
            episode_results.append(result)

            # Score this step
            scores = combined_scorer(result)
            print(f"  Step {step}: {obs['last_action_result']} | "
                  f"reward={obs['reward']:.2f} | "
                  f"overall={scores['overall_score']:.2f}")

            step += 1

        all_results.extend(episode_results)

    # Aggregate scores
    total_hits = sum(1 for r in all_results if r["result"] in ["hit", "double_kill"])
    total_shots = len(all_results)

    return {
        "name": eval_name,
        "total_shots": total_shots,
        "total_hits": total_hits,
        "hit_rate": total_hits / total_shots if total_shots > 0 else 0,
        "avg_reward": sum(r["reward"] for r in all_results) / total_shots if total_shots > 0 else 0,
        "avg_horizon": sum(r["horizon"] for r in all_results) / total_shots if total_shots > 0 else 0,
        "results": all_results,
    }


# Convenience function
def evaluate_agent(
    model_name: str = "gpt-4o",
    num_episodes: int = 3,
    max_steps: int = 50,
    name: str | None = None,
) -> dict:
    """
    Quick evaluation of an agent.

    Usage:
        from experiments.evaluation import evaluate_agent
        results = evaluate_agent("gpt-4o", num_episodes=3, name="my-eval")
    """
    weave.init("duck-hunt-vlm-evaluation")

    agent = DuckHuntVLMAgent(model_name=model_name)
    results = run_live_evaluation(agent, num_episodes, max_steps, name=name)

    print(f"\n=== Results: {results['name']} ===")
    print(f"Hit rate: {results['hit_rate']:.1%}")
    print(f"Avg reward: {results['avg_reward']:.2f}")
    print(f"Avg horizon: {results['avg_horizon']:.1f}")

    return results
