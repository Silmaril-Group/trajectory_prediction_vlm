"""Evaluation script for trained Duck Hunt GRPO models.

Usage::

    # Evaluate a trained checkpoint
    python evaluate.py --config configs/ministral_config.yaml \\
        --checkpoint outputs/ministral_duckhunt_grpo/final \\
        --num-episodes 5

    # Include baselines
    python evaluate.py --config configs/ministral_config.yaml \\
        --checkpoint outputs/ministral_duckhunt_grpo/final \\
        --baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import FullConfig
from src.environment import DuckHuntEnvWrapper
from src.model import load_model_and_processor, apply_lora
from src.reward import compute_reward
from src.utils import TOOLS, Action, build_prompt, parse_tool_call, set_model_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
#  Data structures
# ===================================================================
@dataclass
class StepResult:
    action: Action | None
    hit_a: bool = False
    hit_b: bool = False
    had_target: bool = False
    reward: float = 0.0
    horizon: int = 0
    latency_frames: int = 0
    raw_output: str = ""


@dataclass
class EpisodeResult:
    latency_ms: int = 0
    steps: list[StepResult] = field(default_factory=list)
    total_reward: float = 0.0

    @property
    def total_shots(self) -> int:
        return len(self.steps)

    @property
    def total_hits(self) -> int:
        return sum(
            int(s.hit_a) + int(s.hit_b) for s in self.steps
        )

    @property
    def double_kills(self) -> int:
        return sum(1 for s in self.steps if s.hit_a and s.hit_b)

    @property
    def misses(self) -> int:
        return sum(
            1 for s in self.steps
            if s.had_target and not s.hit_a and not s.hit_b and s.action is not None
        )

    @property
    def invalid_actions(self) -> int:
        return sum(1 for s in self.steps if s.action is None)

    @property
    def hit_rate(self) -> float:
        return self.total_hits / max(self.total_shots, 1)

    @property
    def horizons(self) -> list[int]:
        return [s.horizon for s in self.steps if s.action is not None]


# ===================================================================
#  8.1  Evaluation loop
# ===================================================================
def evaluate(
    model,
    processor,
    env: DuckHuntEnvWrapper,
    config: FullConfig,
    num_episodes: int = 5,
    max_steps_per_episode: int = 100,
) -> dict:
    """Run evaluation episodes across all latency buckets.

    Parameters
    ----------
    model : PreTrainedModel or PeftModel
    processor : AutoProcessor
    env : DuckHuntEnvWrapper
    config : FullConfig
    num_episodes : int
        Episodes **per latency bucket**.
    max_steps_per_episode : int
        Safety cap.

    Returns
    -------
    dict
        Aggregated metrics (see ``_aggregate_metrics``).
    """
    model.eval()
    device = next(model.parameters()).device
    latency_options = config.environment.latency_options_ms
    episodes: list[EpisodeResult] = []

    for latency_ms in latency_options:
        latency_frames = int(latency_ms / 1000 * config.environment.fps)
        logger.info("Evaluating latency=%dms (%d frames) …", latency_ms, latency_frames)

        for ep_idx in range(num_episodes):
            ep = _run_episode(
                model=model,
                processor=processor,
                env=env,
                config=config,
                device=device,
                latency_ms=latency_ms,
                latency_frames=latency_frames,
                max_steps=max_steps_per_episode,
            )
            episodes.append(ep)
            logger.info(
                "  episode %d/%d  hits=%d  shots=%d  reward=%.2f",
                ep_idx + 1, num_episodes, ep.total_hits, ep.total_shots, ep.total_reward,
            )

    metrics = _aggregate_metrics(episodes, latency_options)
    model.train()
    return metrics


def _run_episode(
    *,
    model,
    processor,
    env: DuckHuntEnvWrapper,
    config: FullConfig,
    device: torch.device,
    latency_ms: int,
    latency_frames: int,
    max_steps: int,
) -> EpisodeResult:
    """Run a single evaluation episode with a fixed latency."""
    # Force the desired latency
    env._env.latency_ms = latency_ms
    env._env.latency_frames = latency_frames
    env.reset()

    ep = EpisodeResult(latency_ms=latency_ms)
    max_horizon = config.environment.max_horizon

    for _ in range(max_steps):
        if env.is_done():
            break

        frames = env.get_frames()
        state = env.get_state()

        if state.get("ducks_flying", 0) == 0:
            env.advance_frames(15)
            if env.is_done():
                break
            frames = env.get_frames()
            state = env.get_state()
            if state.get("ducks_flying", 0) == 0:
                continue

        # Build prompt and generate (greedy)
        messages, tools = build_prompt(frames, state)
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.grpo.max_completion_length,
                do_sample=False,
            )

        new_tokens = output_ids[0, prompt_len:]
        decoded = processor.decode(new_tokens, skip_special_tokens=False)

        # Parse and execute
        action = parse_tool_call(decoded, max_horizon=max_horizon)

        if action is not None:
            obs = env.step(action.x, action.y, action.horizon)
            result_str = obs.get("last_action_result", "miss")
            hit_a = result_str in ("hit", "double_kill")
            hit_b = result_str == "double_kill"
            had_target = result_str != "no_target"
        else:
            # Invalid output — fire at centre as penalty
            obs = env.step(0.5, 0.5, 0)
            hit_a, hit_b, had_target = False, False, True

        result_dict = {"hit_a": hit_a, "hit_b": hit_b, "had_target": had_target}
        reward = compute_reward(result_dict, action, config.reward)

        step_result = StepResult(
            action=action,
            hit_a=hit_a,
            hit_b=hit_b,
            had_target=had_target,
            reward=reward,
            horizon=action.horizon if action else 0,
            latency_frames=latency_frames,
            raw_output=decoded[:300],
        )
        ep.steps.append(step_result)
        ep.total_reward += reward

    return ep


# ===================================================================
#  8.2  Metrics aggregation
# ===================================================================
def _aggregate_metrics(
    episodes: list[EpisodeResult],
    latency_options: list[int],
) -> dict:
    all_steps = [s for ep in episodes for s in ep.steps]
    total_shots = len(all_steps)

    if total_shots == 0:
        return {"error": "no steps recorded"}

    # ---- Core metrics ----
    total_hits = sum(int(s.hit_a) + int(s.hit_b) for s in all_steps)
    total_double = sum(1 for s in all_steps if s.hit_a and s.hit_b)
    total_miss = sum(
        1 for s in all_steps
        if s.had_target and not s.hit_a and not s.hit_b and s.action is not None
    )
    total_invalid = sum(1 for s in all_steps if s.action is None)
    total_reward = sum(s.reward for s in all_steps)

    core = {
        "hit_rate": total_hits / total_shots,
        "double_kill_rate": total_double / total_shots,
        "miss_rate": total_miss / total_shots,
        "invalid_action_rate": total_invalid / total_shots,
        "average_reward": total_reward / total_shots,
        "total_shots": total_shots,
        "total_hits": total_hits,
        "num_episodes": len(episodes),
    }

    # ---- Horizon metrics ----
    horizons = [s.horizon for s in all_steps if s.action is not None]
    horizon_metrics = {}
    if horizons:
        horizon_metrics = {
            "average_horizon": statistics.mean(horizons),
            "horizon_std": statistics.stdev(horizons) if len(horizons) > 1 else 0.0,
            "horizon_min": min(horizons),
            "horizon_max": max(horizons),
        }

    # ---- Per-latency metrics ----
    by_latency: dict[int, dict] = {}
    hit_rates_by_latency: list[float] = []

    for lat_ms in latency_options:
        lat_episodes = [ep for ep in episodes if ep.latency_ms == lat_ms]
        lat_steps = [s for ep in lat_episodes for s in ep.steps]
        if not lat_steps:
            continue

        lat_hits = sum(int(s.hit_a) + int(s.hit_b) for s in lat_steps)
        lat_hr = lat_hits / len(lat_steps)
        hit_rates_by_latency.append(lat_hr)

        lat_horizons = [s.horizon for s in lat_steps if s.action is not None]
        by_latency[lat_ms] = {
            "hit_rate": lat_hr,
            "shots": len(lat_steps),
            "hits": lat_hits,
            "average_horizon": statistics.mean(lat_horizons) if lat_horizons else 0.0,
            "average_reward": sum(s.reward for s in lat_steps) / len(lat_steps),
        }

    # ---- Hardware-aware metrics ----
    hw_metrics = {}
    if len(hit_rates_by_latency) >= 2:
        hw_metrics["generalization_gap"] = max(hit_rates_by_latency) - min(hit_rates_by_latency)

        # Adaptation score: correlation between latency and horizon
        lat_horizon_pairs = []
        for lat_ms in latency_options:
            if lat_ms in by_latency:
                lat_horizon_pairs.append(
                    (lat_ms, by_latency[lat_ms]["average_horizon"])
                )
        if len(lat_horizon_pairs) >= 2:
            hw_metrics["adaptation_score"] = _pearson(
                [p[0] for p in lat_horizon_pairs],
                [p[1] for p in lat_horizon_pairs],
            )

    return {
        "core": core,
        "horizon": horizon_metrics,
        "by_latency": by_latency,
        "hardware_aware": hw_metrics,
    }


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = (sx * sy) ** 0.5
    return sxy / denom if denom > 0 else 0.0


# ===================================================================
#  Baselines
# ===================================================================
def run_random_baseline(
    env: DuckHuntEnvWrapper,
    config: FullConfig,
    num_episodes: int = 5,
    max_steps: int = 100,
) -> dict:
    """Random agent: uniform random x, y, horizon."""
    episodes: list[EpisodeResult] = []
    max_horizon = config.environment.max_horizon

    for lat_ms in config.environment.latency_options_ms:
        lat_frames = int(lat_ms / 1000 * config.environment.fps)

        for _ in range(num_episodes):
            env._env.latency_ms = lat_ms
            env._env.latency_frames = lat_frames
            env.reset()

            ep = EpisodeResult(latency_ms=lat_ms)
            for _ in range(max_steps):
                if env.is_done():
                    break
                if env.get_flying_count() == 0:
                    env.advance_frames(15)
                    if env.is_done():
                        break
                    if env.get_flying_count() == 0:
                        continue

                action = Action(
                    x=random.random(),
                    y=random.random() * 0.5,  # upper half
                    horizon=random.randint(0, max_horizon),
                )
                obs = env.step(action.x, action.y, action.horizon)
                result_str = obs.get("last_action_result", "miss")

                hit_a = result_str in ("hit", "double_kill")
                hit_b = result_str == "double_kill"
                had_target = result_str != "no_target"

                reward = compute_reward(
                    {"hit_a": hit_a, "hit_b": hit_b, "had_target": had_target},
                    action, config.reward,
                )
                ep.steps.append(StepResult(
                    action=action, hit_a=hit_a, hit_b=hit_b,
                    had_target=had_target, reward=reward,
                    horizon=action.horizon, latency_frames=lat_frames,
                ))
                ep.total_reward += reward
            episodes.append(ep)

    return _aggregate_metrics(episodes, config.environment.latency_options_ms)


def run_fixed_horizon_baseline(
    env: DuckHuntEnvWrapper,
    config: FullConfig,
    horizon: int = 10,
    num_episodes: int = 5,
    max_steps: int = 100,
) -> dict:
    """Centre-shot agent with fixed horizon."""
    episodes: list[EpisodeResult] = []

    for lat_ms in config.environment.latency_options_ms:
        lat_frames = int(lat_ms / 1000 * config.environment.fps)

        for _ in range(num_episodes):
            env._env.latency_ms = lat_ms
            env._env.latency_frames = lat_frames
            env.reset()

            ep = EpisodeResult(latency_ms=lat_ms)
            for _ in range(max_steps):
                if env.is_done():
                    break
                if env.get_flying_count() == 0:
                    env.advance_frames(15)
                    if env.is_done():
                        break
                    if env.get_flying_count() == 0:
                        continue

                action = Action(x=0.5, y=0.25, horizon=horizon)
                obs = env.step(action.x, action.y, action.horizon)
                result_str = obs.get("last_action_result", "miss")

                hit_a = result_str in ("hit", "double_kill")
                hit_b = result_str == "double_kill"
                had_target = result_str != "no_target"

                reward = compute_reward(
                    {"hit_a": hit_a, "hit_b": hit_b, "had_target": had_target},
                    action, config.reward,
                )
                ep.steps.append(StepResult(
                    action=action, hit_a=hit_a, hit_b=hit_b,
                    had_target=had_target, reward=reward,
                    horizon=action.horizon, latency_frames=lat_frames,
                ))
                ep.total_reward += reward
            episodes.append(ep)

    return _aggregate_metrics(episodes, config.environment.latency_options_ms)


# ===================================================================
#  Pretty-print
# ===================================================================
def print_metrics(metrics: dict, label: str = "Evaluation") -> None:
    """Print metrics in a readable table."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    if "core" in metrics:
        c = metrics["core"]
        print(f"  Hit rate:           {c['hit_rate']:.1%}")
        print(f"  Double kill rate:   {c['double_kill_rate']:.1%}")
        print(f"  Miss rate:          {c['miss_rate']:.1%}")
        print(f"  Invalid action:     {c['invalid_action_rate']:.1%}")
        print(f"  Avg reward:         {c['average_reward']:.3f}")
        print(f"  Total shots:        {c['total_shots']}")

    if "horizon" in metrics and metrics["horizon"]:
        h = metrics["horizon"]
        print(f"\n  Avg horizon:        {h['average_horizon']:.1f}")
        print(f"  Horizon std:        {h['horizon_std']:.1f}")
        print(f"  Horizon range:      [{h['horizon_min']}, {h['horizon_max']}]")

    if "by_latency" in metrics and metrics["by_latency"]:
        print(f"\n  {'Latency':>10}  {'Hit Rate':>10}  {'Avg Horizon':>12}  {'Avg Reward':>11}")
        print(f"  {'-' * 48}")
        for lat_ms, m in sorted(metrics["by_latency"].items()):
            print(
                f"  {lat_ms:>7}ms  {m['hit_rate']:>10.1%}"
                f"  {m['average_horizon']:>12.1f}"
                f"  {m['average_reward']:>11.3f}"
            )

    if "hardware_aware" in metrics and metrics["hardware_aware"]:
        hw = metrics["hardware_aware"]
        if "generalization_gap" in hw:
            print(f"\n  Generalization gap: {hw['generalization_gap']:.1%}")
        if "adaptation_score" in hw:
            print(f"  Adaptation score:   {hw['adaptation_score']:.3f}")

    print(f"{'=' * 60}\n")


# ===================================================================
#  CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Duck Hunt GRPO model")
    parser.add_argument(
        "--config", type=str, action="append", required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint (if omitted, evaluates base model).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=3,
        help="Episodes per latency bucket.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also run random and fixed-horizon baselines.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save metrics JSON.",
    )
    args = parser.parse_args()

    # Config
    if len(args.config) == 1:
        cfg = FullConfig.from_yaml(args.config[0])
    else:
        cfg = FullConfig.from_yamls(*args.config)

    # Activate the right tool-call format based on model name
    set_model_format(cfg.model.model_name)

    # Model
    if args.checkpoint:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoProcessor

        logger.info("Loading checkpoint from %s …", args.checkpoint)
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.checkpoint)
        processor.tokenizer.padding_side = "left"
    else:
        logger.info("No checkpoint — evaluating base model.")
        model, processor = load_model_and_processor(cfg.model)
        model = apply_lora(model, cfg.lora)

    # Environment
    env = DuckHuntEnvWrapper(cfg.environment)

    # Evaluate model
    logger.info("Running model evaluation …")
    model_metrics = evaluate(
        model, processor, env, cfg,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
    )
    print_metrics(model_metrics, "Trained Model" if args.checkpoint else "Base Model")

    all_results = {"model": model_metrics}

    # Baselines
    if args.baselines:
        logger.info("Running random baseline …")
        rand_metrics = run_random_baseline(env, cfg, num_episodes=args.num_episodes, max_steps=args.max_steps)
        print_metrics(rand_metrics, "Random Baseline")
        all_results["random_baseline"] = rand_metrics

        logger.info("Running fixed-horizon baseline (h=10) …")
        fixed_metrics = run_fixed_horizon_baseline(env, cfg, horizon=10, num_episodes=args.num_episodes, max_steps=args.max_steps)
        print_metrics(fixed_metrics, "Fixed Horizon (h=10)")
        all_results["fixed_horizon_baseline"] = fixed_metrics

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Metrics saved to %s", args.output)


if __name__ == "__main__":
    main()
