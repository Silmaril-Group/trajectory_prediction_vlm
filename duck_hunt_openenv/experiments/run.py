#!/usr/bin/env python3
"""Main entry point for running Duck Hunt VLM experiments."""

import argparse
import sys
from pathlib import Path

import weave

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from .agent import DuckHuntVLMAgent
from .episode import run_episode, run_episodes
from .evaluation import run_live_evaluation, evaluate_agent


def main():
    parser = argparse.ArgumentParser(description="Duck Hunt VLM Experiments")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "evaluate"],
        default="single",
        help="Run mode: single episode, batch episodes, or evaluation",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes for batch/evaluate mode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--project",
        default="duck-hunt-vlm-research",
        help="Weave project name",
    )
    parser.add_argument(
        "--log-frames",
        action="store_true",
        help="Log frame samples (uses more storage)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the evaluation run",
    )

    args = parser.parse_args()

    # Initialize Weave
    weave.init(args.project)
    print(f"Weave project: {args.project}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print("=" * 50)

    # Create agent
    agent = DuckHuntVLMAgent(
        model_name=args.model,
        temperature=args.temperature,
    )

    if args.mode == "single":
        print("Running single episode...")
        result = run_episode(agent, args.max_steps, args.log_frames)

        print("\n=== Episode Result ===")
        print(f"Total reward: {result.total_reward:.2f}")
        print(f"Total steps: {result.total_steps}")
        print(f"Ducks hit: {result.ducks_hit}")
        print(f"Total misses: {result.total_misses}")
        print(f"Final round: {result.final_round}")
        print(f"Final match: {result.final_match}")

        # Summary of actions
        hits = sum(1 for s in result.steps if s.result in ["hit", "double_kill"])
        misses = sum(1 for s in result.steps if s.result == "miss")
        print(f"\nAction summary: {hits} hits, {misses} misses")

    elif args.mode == "batch":
        print(f"Running {args.episodes} episodes...")
        results = run_episodes(
            agent,
            args.episodes,
            args.max_steps,
            args.log_frames,
        )

        print("\n=== Batch Results ===")
        print(f"Episodes: {results['num_episodes']}")
        print(f"Avg reward: {results['avg_reward']:.2f}")
        print(f"Min reward: {results['min_reward']:.2f}")
        print(f"Max reward: {results['max_reward']:.2f}")
        print(f"Avg ducks hit: {results['avg_ducks_hit']:.1f}")
        print(f"Avg steps: {results['avg_steps']:.1f}")

    elif args.mode == "evaluate":
        eval_name = args.name or f"eval-{args.model}-{args.episodes}ep"
        print(f"Running evaluation '{eval_name}' with {args.episodes} episodes...")
        results = run_live_evaluation(
            agent,
            args.episodes,
            args.max_steps,
            name=eval_name,
        )

        print(f"\n=== Evaluation Results: {results['name']} ===")
        print(f"Total shots: {results['total_shots']}")
        print(f"Total hits: {results['total_hits']}")
        print(f"Hit rate: {results['hit_rate']:.1%}")
        print(f"Avg reward: {results['avg_reward']:.2f}")
        print(f"Avg horizon: {results['avg_horizon']:.1f}")

    print("\n" + "=" * 50)
    print("View results at: https://wandb.ai/")


if __name__ == "__main__":
    main()
