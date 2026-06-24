"""VLM evaluation for Duck Hunt horizon minimization via served model API.

Queries a VLM served by vLLM or SGLang (OpenAI-compatible API) and evaluates:
  1. Processing time — wall-clock time per number of input frames (frame sweep)
  2. Tool call validity — parseable JSON, valid param names and values
  3. Horizon analysis — is the predicted horizon reasonable, does it vary?
  4. Hit rate — skip frames for processing time + horizon, check if duck is hit
  5. LLM-as-a-judge — qualitative assessment of game understanding

The model is NOT loaded locally — it must be served via ``serve_vlm.sh`` first.

Usage::

    # Start the model server first
    ./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B

    # Run evaluation
    python eval_vlm.py --config configs/liquidai_eval.yaml

    # Single model with custom endpoint
    python eval_vlm.py --config configs/liquidai_eval.yaml \\
        --model LiquidAI/LFM2.5-VL-1.6B --api-base http://localhost:8000/v1

    # With Weave tracking
    python eval_vlm.py --config configs/liquidai_eval.yaml --weave

    # Post-training eval
    python eval_vlm.py --config configs/liquidai_eval.yaml \\
        --checkpoint outputs/lfm25_duckhunt_grpo/best

    # Save results to specific file
    python eval_vlm.py --config configs/liquidai_eval.yaml \\
        --output results/my_eval.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.environment import DuckHuntEnvWrapper
from src.config import EnvironmentConfig, RewardConfig
from src.reward import compute_reward
from src.dataset import simulate_shot, capture_snapshot
from src.utils import Action, _build_action

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
class FrameSweepResult:
    num_frames: int
    processing_time_ms: float
    success: bool
    produced_tool_call: bool
    raw_output: str = ""
    error: str = ""


@dataclass
class ScenarioResult:
    scenario_id: int
    num_frames: int
    action: Action | None = None
    produced_tool_call: bool = False
    valid_params: bool = False
    raw_output: str = ""
    processing_time_ms: float = 0.0
    game_state: dict = field(default_factory=dict)
    judge_score: dict | None = None

    # Shot execution
    hit_a: bool = False
    hit_b: bool = False
    had_target: bool = False
    reward: float = 0.0
    shot_result: str = ""
    latency_frames_actual: int = 0  # ceil(processing_time_ms / 1000 * 30)


@dataclass
class ModelEvalResult:
    model_id: str
    checkpoint: str | None = None
    api_base: str = ""

    # Frame sweep
    max_frames_supported: int = 0
    frame_sweep: list[FrameSweepResult] = field(default_factory=list)

    # Tool call quality
    tool_call_rate: float = 0.0
    valid_action_rate: float = 0.0

    # Horizon stats
    horizon_min: int | None = None
    horizon_max: int | None = None
    horizon_mean: float | None = None
    horizon_std: float | None = None
    horizon_all_same: bool | None = None

    # Shot results
    total_shots: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    double_kills: int = 0
    misses: int = 0
    average_reward: float = 0.0

    # Scenario results
    scenarios: list[ScenarioResult] = field(default_factory=list)

    # Judge
    judge_avg_score: float | None = None

    # Errors
    connection_error: str = ""


# ===================================================================
#  Tool call parsing — model-agnostic
# ===================================================================
def parse_tool_call(output_text: str, tool_call_format: str, max_horizon: int = 30) -> Action | None:
    """Parse a tool call from model output, handling multiple formats."""

    # --- LiquidAI: <|tool_call_start|>[shoot(x=0.3, y=0.2, horizon=5)]<|tool_call_end|> ---
    liq_match = re.search(
        r"(?:<\|tool_call_start\|>|tool_call_start)"
        r".*?shoot\s*\((.*?)\)"
        r".*?(?:<\|tool_call_end\|>|tool_call_end)?",
        output_text, re.DOTALL | re.IGNORECASE,
    )
    if liq_match:
        return _parse_kwargs(liq_match.group(1), max_horizon)

    # --- Pythonic: shoot(x=0.3, y=0.2, horizon=5) ---
    py_match = re.search(
        r"shoot\s*\((.*?)\)", output_text, re.DOTALL | re.IGNORECASE,
    )
    if py_match:
        return _parse_kwargs(py_match.group(1), max_horizon)

    # --- Mistral: [TOOL_CALLS] [{"name": "shoot", "arguments": {...}}] ---
    tc_match = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", output_text, re.DOTALL)
    if tc_match:
        try:
            calls = json.loads(tc_match.group(1))
            if isinstance(calls, list) and len(calls) > 0:
                args = calls[0].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if "x" in args and "y" in args:
                    return _build_action(args["x"], args["y"], args.get("horizon", 0), max_horizon)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # --- JSON object with x/y ---
    try:
        start = output_text.index("{")
        depth, end = 0, start
        for i, ch in enumerate(output_text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        blob = json.loads(output_text[start:end])
        if "arguments" in blob and isinstance(blob["arguments"], dict):
            blob = blob["arguments"]
        elif "arguments" in blob and isinstance(blob["arguments"], str):
            blob = json.loads(blob["arguments"])
        if "x" in blob and "y" in blob:
            return _build_action(blob["x"], blob["y"], blob.get("horizon", 0), max_horizon)
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # --- key=value fallback ---
    kv_match = re.search(
        r"x\s*[=:]\s*([0-9.eE+-]+).*?"
        r"y\s*[=:]\s*([0-9.eE+-]+).*?"
        r"horizon\s*[=:]\s*([0-9]+)",
        output_text, re.DOTALL | re.IGNORECASE,
    )
    if kv_match:
        return _build_action(kv_match.group(1), kv_match.group(2), kv_match.group(3), max_horizon)

    return None


def _parse_kwargs(args_str: str, max_horizon: int) -> Action | None:
    vals = {}
    for match in re.finditer(r"(\w+)\s*=\s*([0-9.eE+-]+)", args_str):
        vals[match.group(1)] = match.group(2)
    if "x" in vals and "y" in vals:
        return _build_action(vals["x"], vals["y"], vals.get("horizon", "0"), max_horizon)
    return None


def _validate_action_params(action: Action | None) -> bool:
    """Check that parsed action has valid parameter values."""
    if action is None:
        return False
    return (0.0 <= action.x <= 1.0 and
            0.0 <= action.y <= 1.0 and
            0 <= action.horizon <= 30)


# ===================================================================
#  Single query method — all tests use this
# ===================================================================
def _pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


SYSTEM_PROMPT = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} game frame(s). Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).
Predict where the duck will be after processing latency plus your chosen horizon frames.

IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."""


SHOOT_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": (
            "Fire at predicted duck position. "
            "Analyze the frame sequence to estimate duck velocity, "
            "then predict where the duck will be after "
            "processing_latency_frames + horizon frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Predicted horizontal position, normalised 0.0-1.0.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": "Predicted vertical position, normalised 0.0-1.0.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": "Additional frames to wait before shooting (0-30).",
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}


def query_model(
    client,
    model_id: str,
    frames: list[Image.Image],
    game_state: dict,
    generation_params: dict | None = None,
) -> tuple[str, float]:
    """Query a served VLM via OpenAI-compatible API.

    Parameters
    ----------
    client : openai.OpenAI
        OpenAI client pointing at the served model.
    model_id : str
        Model ID as registered in the server (e.g. "LiquidAI/LFM2.5-VL-1.6B").
    frames : list[Image.Image]
        Game frames to send as images.
    game_state : dict
        Game state metadata (ducks_flying, latency_frames, etc.).
    generation_params : dict, optional
        Temperature, max_tokens, etc.

    Returns
    -------
    tuple[str, float]
        (response_text, elapsed_ms)
    """
    gen = generation_params or {}
    num_frames = len(frames)
    latency_frames = game_state.get("simulated_latency_frames", 6)

    system_text = SYSTEM_PROMPT.format(num_frames=num_frames)

    # Build user content with base64 images
    user_content = []
    for img in frames:
        b64 = _pil_to_base64(img)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    user_content.append({
        "type": "text",
        "text": (
            f"{num_frames} frames, {game_state.get('ducks_flying', '?')} ducks flying, "
            f"latency {latency_frames} frames. Shoot now."
        ),
    })

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=gen.get("temperature", 0.1),
        max_tokens=gen.get("max_new_tokens", 256),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Extract response text — check both content and tool_calls
    choice = response.choices[0]
    response_text = ""

    if choice.message.tool_calls:
        # Server parsed tool call natively
        tc = choice.message.tool_calls[0]
        response_text = json.dumps({
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
        })
    elif choice.message.content:
        response_text = choice.message.content

    return response_text, elapsed_ms


# ===================================================================
#  Environment helpers
# ===================================================================
def _advance_to_flying(env: DuckHuntEnvWrapper, max_attempts: int = 20) -> None:
    for _ in range(max_attempts):
        if env.get_flying_count() > 0:
            return
        env.advance_frames(15)
        if env.is_done():
            env.reset()


# ===================================================================
#  1. Frame sweep — processing time vs number of input frames
# ===================================================================
def run_frame_sweep(
    client,
    model_id: str,
    model_cfg: dict,
    env: DuckHuntEnvWrapper,
    max_attempts: int = 32,
    max_horizon: int = 30,
) -> list[FrameSweepResult]:
    """Test frame counts 1..N, measuring processing time and tool call success."""
    tool_call_format = model_cfg.get("tool_call_format", "liquidai")
    gen_params = model_cfg.get("generation", {})
    results: list[FrameSweepResult] = []

    env.reset()
    _advance_to_flying(env)

    for n_frames in range(1, max_attempts + 1):
        logger.info("  Frame sweep: %d frame(s) ...", n_frames)

        all_frames = env.get_frames()
        while len(all_frames) < n_frames:
            all_frames.append(all_frames[-1])
        frames = all_frames[:n_frames]

        state = env.get_state()

        try:
            response_text, elapsed_ms = query_model(
                client, model_id, frames, state, gen_params,
            )
            action = parse_tool_call(response_text, tool_call_format, max_horizon)

            results.append(FrameSweepResult(
                num_frames=n_frames,
                processing_time_ms=elapsed_ms,
                success=True,
                produced_tool_call=action is not None,
                raw_output=response_text,
            ))
            logger.info(
                "    %d frames: %.0fms, tool_call=%s",
                n_frames, elapsed_ms, action is not None,
            )

        except Exception as e:
            err_msg = str(e)
            logger.warning("    %d frames: error — %s", n_frames, err_msg[:200])
            results.append(FrameSweepResult(
                num_frames=n_frames, processing_time_ms=0,
                success=False, produced_tool_call=False, error=err_msg[:500],
            ))
            # Stop on context length / OOM errors
            if any(kw in err_msg.lower() for kw in ("context length", "out of memory", "too many tokens", "maximum")):
                break

    return results


# ===================================================================
#  2. Game scenarios — hit rate with processing-time-based latency
# ===================================================================
def run_scenarios(
    client,
    model_id: str,
    model_cfg: dict,
    env: DuckHuntEnvWrapper,
    num_scenarios: int = 10,
    num_frames: int = 4,
    max_horizon: int = 30,
    fps: int = 30,
) -> list[ScenarioResult]:
    """Run game scenarios. For each:
    1. Query served model
    2. Measure processing time -> latency_frames = ceil(processing_time_ms / 1000 * fps)
    3. Use simulate_shot with snapshot (replacing latency_frames with actual)
    4. Compute reward
    """
    tool_call_format = model_cfg.get("tool_call_format", "liquidai")
    gen_params = model_cfg.get("generation", {})
    reward_config = RewardConfig(max_horizon=max_horizon)
    results: list[ScenarioResult] = []

    for i in range(num_scenarios):
        env.reset()
        _advance_to_flying(env)

        frames = env.get_frames()
        if len(frames) > num_frames:
            frames = frames[:num_frames]
        state = env.get_state()

        # Capture snapshot BEFORE querying (deterministic replay)
        snapshot = capture_snapshot(env)

        try:
            response_text, elapsed_ms = query_model(
                client, model_id, frames, state, gen_params,
            )
            action = parse_tool_call(response_text, tool_call_format, max_horizon)
            valid = _validate_action_params(action)

            # Compute actual latency from processing time
            latency_frames_actual = math.ceil(elapsed_ms / 1000.0 * fps)

            # Simulate shot with actual processing latency
            hit_a, hit_b, had_target = False, False, False
            shot_result = "invalid"
            reward = 0.0

            if action is not None:
                # Override snapshot's latency_frames with actual processing latency
                snapshot_copy = dict(snapshot)
                snapshot_copy["latency_frames"] = latency_frames_actual

                sim_result = simulate_shot(snapshot_copy, action)
                hit_a = sim_result["hit_a"]
                hit_b = sim_result["hit_b"]
                had_target = sim_result["had_target"]

                if hit_a and hit_b:
                    shot_result = "double_kill"
                elif hit_a or hit_b:
                    shot_result = "hit"
                elif had_target:
                    shot_result = "miss"
                else:
                    shot_result = "no_target"

                reward = compute_reward(sim_result, action, reward_config)
            else:
                had_target = True
                reward = reward_config.invalid_action

            results.append(ScenarioResult(
                scenario_id=i,
                num_frames=len(frames),
                action=action,
                produced_tool_call=action is not None,
                valid_params=valid,
                raw_output=response_text,
                processing_time_ms=elapsed_ms,
                game_state=state,
                hit_a=hit_a,
                hit_b=hit_b,
                had_target=had_target,
                reward=reward,
                shot_result=shot_result,
                latency_frames_actual=latency_frames_actual,
            ))

            hit_str = shot_result.upper() if hit_a or hit_b else shot_result
            logger.info(
                "  Scenario %d: %s, latency=%d frames (%.0fms), action=%s, reward=%.2f",
                i, hit_str, latency_frames_actual, elapsed_ms,
                f"({action.x:.2f},{action.y:.2f},h={action.horizon})" if action else "None",
                reward,
            )
        except Exception as e:
            logger.warning("  Scenario %d: error — %s", i, str(e)[:200])
            results.append(ScenarioResult(
                scenario_id=i, num_frames=len(frames),
                raw_output=str(e)[:500], game_state=state,
                shot_result="error",
            ))

        # Advance env for diversity
        env.advance_frames(30)

    return results


# ===================================================================
#  3. LLM-as-a-judge
# ===================================================================
JUDGE_PROMPT = """\
You are evaluating a Vision-Language Model's response to a Duck Hunt game scenario.

The model was given {num_frames} game frame(s) showing ducks flying and asked to call shoot(x, y, horizon).

Model's raw output:
---
{raw_output}
---

Game state at the time: {state_summary}
Shot result: {shot_result} (processing latency: {latency_ms:.0f}ms = {latency_frames} frames)

Rate the response on these criteria (1-5 each):
1. TOOL_FORMAT: Did it produce a proper function/tool call? (5=perfect format, 1=no tool call at all)
2. SPATIAL_AWARENESS: Do the x,y coordinates suggest it understood where ducks are? (5=precise, 1=random/nonsense)
3. HORIZON_REASONING: Is the horizon value reasonable given the latency? (5=well-reasoned, 1=nonsensical)
4. INSTRUCTION_FOLLOWING: Did it follow the instruction (tool call only, no explanation)? (5=perfect, 1=ignored instructions)

Respond in JSON only:
{{"tool_format": N, "spatial_awareness": N, "horizon_reasoning": N, "instruction_following": N, "notes": "brief explanation"}}"""


def run_judge(
    scenarios: list[ScenarioResult],
    judge_cfg: dict | None = None,
) -> list[dict]:
    """Run LLM-as-a-judge on scenario outputs via OpenAI API."""
    if judge_cfg is None:
        judge_cfg = {}

    if not judge_cfg.get("enabled", True):
        logger.info("  Judge disabled — skipping")
        return []

    mode = judge_cfg.get("mode", "openai")
    if mode == "self":
        logger.info("  Judge mode 'self' requires locally loaded model — skipping (use 'openai' mode)")
        return []

    if mode != "openai":
        logger.warning("  Judge mode '%s' not supported in API-based eval — use 'openai'", mode)
        return []

    return _run_judge_openai(scenarios, judge_cfg)


def _run_judge_openai(
    scenarios: list[ScenarioResult],
    judge_cfg: dict,
) -> list[dict]:
    """Judge using an OpenAI-compatible API."""
    import os

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("  openai package not installed — run `pip install openai`")
        return []

    max_to_judge = judge_cfg.get("max_scenarios", 5)
    api_key = judge_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    api_base = judge_cfg.get("api_base")
    api_model = judge_cfg.get("api_model", "gpt-4o")

    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base

    client = OpenAI(**client_kwargs)
    logger.info("  Using OpenAI judge: model=%s", api_model)

    results = []
    for sc in scenarios[:max_to_judge]:
        state_summary = (
            f"ducks_flying={sc.game_state.get('ducks_flying', '?')}, "
            f"latency_frames={sc.latency_frames_actual}, "
            f"round={sc.game_state.get('round_number', '?')}"
        )
        judge_text = JUDGE_PROMPT.format(
            num_frames=sc.num_frames,
            raw_output=sc.raw_output[:300],
            state_summary=state_summary,
            shot_result=sc.shot_result,
            latency_ms=sc.processing_time_ms,
            latency_frames=sc.latency_frames_actual,
        )

        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": judge_text}],
                temperature=0,
                max_tokens=256,
            )
            decoded = response.choices[0].message.content or ""

            score = _extract_judge_json(decoded)
            if score:
                sc.judge_score = score
                results.append(score)
                logger.info("  Judge scenario %d: %s", sc.scenario_id, score)
            else:
                logger.warning("  Judge scenario %d: couldn't parse: %s", sc.scenario_id, decoded[:200])
        except Exception as e:
            logger.warning("  Judge scenario %d: error — %s", sc.scenario_id, str(e)[:200])

    return results


def _extract_judge_json(text: str) -> dict | None:
    try:
        start = text.index("{")
        depth, end = 0, start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# ===================================================================
#  Main evaluation orchestrator
# ===================================================================
def evaluate_model(
    client,
    model_cfg: dict,
    eval_cfg: dict,
    env_cfg: dict,
    checkpoint: str | None = None,
) -> ModelEvalResult:
    """Run full evaluation suite for a single model via API."""
    model_id = model_cfg["model_id"]
    max_horizon = eval_cfg.get("max_horizon", 30)
    fps = env_cfg.get("fps", 30)

    result = ModelEvalResult(
        model_id=model_id,
        checkpoint=checkpoint,
        api_base=str(client.base_url),
    )

    # Environment for generating game states + snapshots
    env = DuckHuntEnvWrapper(EnvironmentConfig(
        frame_output_size=tuple(env_cfg.get("frame_output_size", [512, 512])),
        fps=fps,
        max_horizon=max_horizon,
    ))

    # --- 1. Frame sweep ---
    logger.info("[%s] Running frame sweep ...", model_id)
    result.frame_sweep = run_frame_sweep(
        client, model_id, model_cfg, env,
        max_attempts=eval_cfg.get("frame_sweep_max_attempts", 32),
        max_horizon=max_horizon,
    )
    successful = [r for r in result.frame_sweep if r.success]
    result.max_frames_supported = max((r.num_frames for r in successful), default=0)

    # --- 2. Game scenarios (hit rate) ---
    num_scenarios = eval_cfg.get("num_scenarios", 10)
    scenario_frames = min(4, result.max_frames_supported) if result.max_frames_supported > 0 else 4
    logger.info("[%s] Running %d game scenarios (frames=%d) ...", model_id, num_scenarios, scenario_frames)
    result.scenarios = run_scenarios(
        client, model_id, model_cfg, env,
        num_scenarios=num_scenarios, num_frames=scenario_frames,
        max_horizon=max_horizon, fps=fps,
    )

    # Aggregate metrics
    valid_actions = [s.action for s in result.scenarios if s.action is not None]
    result.tool_call_rate = sum(1 for s in result.scenarios if s.produced_tool_call) / max(len(result.scenarios), 1)
    result.valid_action_rate = sum(1 for s in result.scenarios if s.valid_params) / max(len(result.scenarios), 1)

    if valid_actions:
        horizons = [a.horizon for a in valid_actions]
        result.horizon_min = min(horizons)
        result.horizon_max = max(horizons)
        result.horizon_mean = sum(horizons) / len(horizons)
        if len(horizons) > 1:
            mean = result.horizon_mean
            result.horizon_std = (sum((h - mean) ** 2 for h in horizons) / len(horizons)) ** 0.5
        else:
            result.horizon_std = 0.0
        result.horizon_all_same = len(set(horizons)) == 1

    # Shot results
    result.total_shots = len(result.scenarios)
    result.total_hits = sum(int(s.hit_a) + int(s.hit_b) for s in result.scenarios)
    result.hit_rate = result.total_hits / max(result.total_shots, 1)
    result.double_kills = sum(1 for s in result.scenarios if s.hit_a and s.hit_b)
    result.misses = sum(1 for s in result.scenarios if s.had_target and not s.hit_a and not s.hit_b)
    result.average_reward = sum(s.reward for s in result.scenarios) / max(result.total_shots, 1)

    # --- 3. LLM-as-a-judge ---
    judge_cfg = eval_cfg.get("judge", {})
    if judge_cfg.get("enabled", False):
        logger.info("[%s] Running LLM-as-a-judge (mode=%s) ...", model_id, judge_cfg.get("mode", "openai"))
        judge_scores = run_judge(result.scenarios, judge_cfg=judge_cfg)
        if judge_scores:
            all_scores = []
            for s in judge_scores:
                for key in ("tool_format", "spatial_awareness", "horizon_reasoning", "instruction_following"):
                    if key in s:
                        all_scores.append(s[key])
            if all_scores:
                result.judge_avg_score = sum(all_scores) / len(all_scores)

    return result


# ===================================================================
#  Reporting
# ===================================================================
def print_report(results: list[ModelEvalResult]) -> None:
    print(f"\n{'=' * 80}")
    print(f"  VLM EVALUATION REPORT — Duck Hunt Horizon Minimization")
    print(f"{'=' * 80}")

    for r in results:
        label = r.model_id
        if r.checkpoint:
            label += f" (checkpoint: {r.checkpoint})"

        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"  API: {r.api_base}")
        print(f"{'─' * 80}")

        if r.connection_error:
            print(f"  CONNECTION ERROR: {r.connection_error}")
            continue

        # Shot results
        print(f"  Hit rate:                {r.hit_rate:.1%} ({r.total_hits}/{r.total_shots} shots)")
        print(f"  Double kills:            {r.double_kills}")
        print(f"  Misses:                  {r.misses}")
        print(f"  Average reward:          {r.average_reward:.3f}")

        # Capabilities
        print(f"  Max frames supported:    {r.max_frames_supported}")
        print(f"  Tool call rate:          {r.tool_call_rate:.0%}")
        print(f"  Valid action rate:       {r.valid_action_rate:.0%}")

        # Horizon
        if r.horizon_min is not None:
            print(f"  Horizon range:           [{r.horizon_min}, {r.horizon_max}]")
            print(f"  Horizon mean +/- std:    {r.horizon_mean:.1f} +/- {r.horizon_std:.1f}")
            print(f"  Horizon all same:        {r.horizon_all_same}")

        # Frame sweep timing
        if r.frame_sweep:
            print(f"\n  Frame Sweep (processing time):")
            print(f"  {'Frames':>8}  {'Time (ms)':>10}  {'Tool Call':>10}  {'Status':>8}")
            print(f"  {'─' * 42}")
            for fs in r.frame_sweep:
                status = "OK" if fs.success else fs.error[:8]
                tc = "Yes" if fs.produced_tool_call else "No"
                print(f"  {fs.num_frames:>8}  {fs.processing_time_ms:>10.0f}  {tc:>10}  {status:>8}")

        # Judge
        if r.judge_avg_score is not None:
            print(f"\n  LLM-as-Judge avg score:  {r.judge_avg_score:.1f} / 5.0")

        # Sample outputs
        outputs_with_tc = [s for s in r.scenarios if s.produced_tool_call]
        outputs_without = [s for s in r.scenarios if not s.produced_tool_call]
        if outputs_with_tc:
            print(f"\n  Sample output (with tool call):")
            print(f"    {outputs_with_tc[0].raw_output[:200]}")
        if outputs_without:
            print(f"\n  Sample output (NO tool call):")
            print(f"    {outputs_without[0].raw_output[:200]}")

    print(f"\n{'=' * 80}\n")


def save_results(results: list[ModelEvalResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


# ===================================================================
#  Weave integration (optional)
# ===================================================================
_weave_enabled = False


def init_weave(project: str = "duckhunt-vlm-eval") -> None:
    global _weave_enabled
    try:
        import weave
        weave.init(project)
        _weave_enabled = True
        logger.info("Weave initialized: project=%s", project)
    except ImportError:
        logger.warning("Weave not installed — run `pip install weave`")
        _weave_enabled = False


def _weave_log_frame_sweep(model_id: str, results: list[FrameSweepResult]) -> None:
    if not _weave_enabled:
        return
    import weave
    rows = [{
        "model_id": model_id,
        "num_frames": r.num_frames,
        "processing_time_ms": r.processing_time_ms,
        "success": r.success,
        "produced_tool_call": r.produced_tool_call,
        "error": r.error,
    } for r in results]
    weave.publish(weave.Table(rows), name=f"frame_sweep/{model_id.replace('/', '_')}")


def weave_run_evaluation(
    client,
    model_cfg: dict,
    eval_cfg: dict,
    env_cfg: dict,
    checkpoint: str | None = None,
) -> ModelEvalResult:
    """Run evaluation with Weave-tracked scorers."""
    import weave
    from weave import Evaluation
    import asyncio

    model_id = model_cfg["model_id"]
    max_horizon = eval_cfg.get("max_horizon", 30)

    # Run standard evaluation first
    result = evaluate_model(client, model_cfg, eval_cfg, env_cfg, checkpoint)

    if result.connection_error:
        return result

    # Log frame sweep table
    _weave_log_frame_sweep(model_id, result.frame_sweep)

    # Build dataset from scenarios
    dataset_rows = []
    for sc in result.scenarios:
        dataset_rows.append({
            "scenario_id": sc.scenario_id,
            "num_frames": sc.num_frames,
            "raw_output": sc.raw_output,
            "processing_time_ms": sc.processing_time_ms,
            "game_state": sc.game_state,
            "produced_tool_call": sc.produced_tool_call,
            "valid_params": sc.valid_params,
            "action_x": sc.action.x if sc.action else None,
            "action_y": sc.action.y if sc.action else None,
            "action_horizon": sc.action.horizon if sc.action else None,
            "judge_score": sc.judge_score,
            "hit_a": sc.hit_a,
            "hit_b": sc.hit_b,
            "had_target": sc.had_target,
            "reward": sc.reward,
            "shot_result": sc.shot_result,
            "latency_frames_actual": sc.latency_frames_actual,
        })

    # Weave scorers
    @weave.op
    def tool_call_scorer(output: dict) -> dict:
        produced = output.get("produced_tool_call", False)
        valid = output.get("valid_params", False)
        return {
            "produced_tool_call": 1.0 if produced else 0.0,
            "valid_action": 1.0 if valid else 0.0,
        }

    @weave.op
    def accuracy_scorer(output: dict) -> dict:
        hit_a = output.get("hit_a", False)
        hit_b = output.get("hit_b", False)
        had_target = output.get("had_target", False)
        return {
            "is_hit": 1.0 if (hit_a or hit_b) else 0.0,
            "is_double_kill": 1.0 if (hit_a and hit_b) else 0.0,
            "is_miss": 1.0 if (had_target and not hit_a and not hit_b) else 0.0,
            "total_hits": int(hit_a) + int(hit_b),
            "reward": output.get("reward", 0.0),
            "shot_result": output.get("shot_result", ""),
        }

    @weave.op
    def horizon_scorer(output: dict) -> dict:
        h = output.get("action_horizon")
        if h is None:
            return {"horizon_valid": 0.0, "horizon_efficiency": 0.0}
        valid = 0 <= h <= max_horizon
        efficiency = 1.0 - (h / max_horizon) if valid else 0.0
        return {
            "horizon_valid": 1.0 if valid else 0.0,
            "horizon_efficiency": efficiency,
            "horizon_value": h,
        }

    @weave.op
    def timing_scorer(output: dict) -> dict:
        ms = output.get("processing_time_ms", 0)
        speed_score = max(0.0, 1.0 - (ms / 2000.0))
        latency_frames = output.get("latency_frames_actual", 0)
        return {
            "processing_time_ms": ms,
            "speed_score": speed_score,
            "latency_frames": latency_frames,
        }

    @weave.op
    def judge_scorer(output: dict) -> dict:
        js = output.get("judge_score")
        if not js:
            return {"judge_available": 0.0}
        scores = {
            "judge_available": 1.0,
            "judge_tool_format": js.get("tool_format", 0),
            "judge_spatial_awareness": js.get("spatial_awareness", 0),
            "judge_horizon_reasoning": js.get("horizon_reasoning", 0),
            "judge_instruction_following": js.get("instruction_following", 0),
        }
        judge_vals = [v for k, v in scores.items() if k.startswith("judge_") and k != "judge_available"]
        if judge_vals:
            scores["judge_avg"] = sum(judge_vals) / len(judge_vals)
        return scores

    class PrecomputedModel(weave.Model):
        model_id: str = model_id
        checkpoint: str = checkpoint or "base"

        @weave.op
        def predict(self, **kwargs) -> dict:
            return kwargs

    eval_name = model_id.replace("/", "_")
    if checkpoint:
        eval_name += "_checkpoint"

    evaluation = Evaluation(
        name=f"duckhunt_{eval_name}",
        dataset=dataset_rows,
        scorers=[tool_call_scorer, accuracy_scorer, horizon_scorer, timing_scorer, judge_scorer],
    )

    wrapper = PrecomputedModel()
    asyncio.run(evaluation.evaluate(wrapper))
    logger.info("[%s] Weave evaluation published", model_id)

    # Publish summary
    summary = {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "max_frames_supported": result.max_frames_supported,
        "tool_call_rate": result.tool_call_rate,
        "valid_action_rate": result.valid_action_rate,
        "hit_rate": result.hit_rate,
        "total_shots": result.total_shots,
        "total_hits": result.total_hits,
        "double_kills": result.double_kills,
        "average_reward": result.average_reward,
        "horizon_min": result.horizon_min,
        "horizon_max": result.horizon_max,
        "horizon_mean": result.horizon_mean,
        "horizon_std": result.horizon_std,
        "horizon_all_same": result.horizon_all_same,
        "judge_avg_score": result.judge_avg_score,
    }
    weave.publish(summary, name=f"summary/{eval_name}")

    return result


# ===================================================================
#  CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLMs for Duck Hunt (API-based)")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to eval YAML config (e.g. configs/liquidai_eval.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Evaluate only this model ID (default: all models in config).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to fine-tuned checkpoint (post-training eval).",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="Override API base URL (default: from config or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--weave", action="store_true",
        help="Enable Weave tracking (results visible in W&B console).",
    )
    parser.add_argument(
        "--weave-project", type=str, default="duckhunt-vlm-eval",
        help="Weave project name (default: duckhunt-vlm-eval).",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models_cfg = cfg.get("models", [])
    eval_cfg = cfg.get("eval", {})
    env_cfg = cfg.get("environment", {})

    # Weave init
    if args.weave:
        init_weave(args.weave_project)

    # Filter to single model if specified
    if args.model:
        models_cfg = [m for m in models_cfg if m["model_id"] == args.model]
        if not models_cfg:
            logger.error("Model %s not found in config", args.model)
            sys.exit(1)

    # Build OpenAI client
    from openai import OpenAI

    api_base = args.api_base or cfg.get("api", {}).get("base_url", "http://localhost:8000/v1")

    client = OpenAI(
        base_url=api_base,
        api_key="not-needed",  # vLLM/SGLang don't require API keys
    )

    # Verify connection
    try:
        client.models.list()
        logger.info("Connected to API at %s", api_base)
    except Exception as e:
        logger.error("Cannot connect to API at %s: %s", api_base, e)
        logger.error("Start the model server first: ./serve_vlm.sh --model <model_id>")
        sys.exit(1)

    # Run evaluations
    all_results: list[ModelEvalResult] = []
    for model_cfg in models_cfg:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", model_cfg["model_id"])
        logger.info("=" * 60)

        if _weave_enabled:
            result = weave_run_evaluation(
                client, model_cfg, eval_cfg, env_cfg,
                checkpoint=args.checkpoint,
            )
        else:
            result = evaluate_model(
                client, model_cfg, eval_cfg, env_cfg,
                checkpoint=args.checkpoint,
            )
        all_results.append(result)

    # Report
    print_report(all_results)

    # Save
    output_path = args.output or "results/vlm_eval.json"
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
