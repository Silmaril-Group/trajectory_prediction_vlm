"""Dataset generation and TRL-compatible reward function for Duck Hunt GRPO.

The training loop works as follows:

1. **Offline**: ``DuckHuntPromptGenerator`` rolls out the environment and
   collects ``(prompt, images, snapshot)`` tuples.  Each snapshot captures
   enough state to **deterministically** replay the shot later.
2. **Online (TRL)**: ``GRPOTrainer`` generates completions for each prompt,
   then calls our reward function which parses each completion, restores the
   snapshot, simulates the shot, and returns a scalar reward.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import io

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as DatasetImage
from PIL import Image

from .config import EnvironmentConfig, RewardConfig
from .reward import compute_reward
from .utils import Action, build_prompt, parse_tool_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make the server package importable (same as environment.py)
# ---------------------------------------------------------------------------
_SERVER_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "duck_hunt_openenv"
    / "server"
)
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from game_engine import Duck, DuckState, Match  # noqa: E402
from config import SCREEN_WIDTH, SCREEN_HEIGHT, MAX_HORIZON  # noqa: E402


# ===================================================================
#  6.1  Snapshot helpers
# ===================================================================
def _snapshot_duck(duck: Duck) -> dict:
    """Capture the mutable state of a Duck."""
    return {
        "x": duck.x,
        "y": duck.y,
        "dx": duck.dx,
        "dy": duck.dy,
        "state": duck.state.value,
        "sprite_dir": duck.sprite_dir,
    }


def _restore_duck(data: dict, round_number: int) -> Duck:
    """Re-create a Duck from a snapshot dict (bypass __init__)."""
    duck = object.__new__(Duck)
    duck.x = data["x"]
    duck.y = data["y"]
    duck.dx = data["dx"]
    duck.dy = data["dy"]
    duck.state = DuckState(data["state"])
    duck.sprite_dir = data["sprite_dir"]
    return duck


def capture_snapshot(env_wrapper) -> dict:
    """Take a JSON-serialisable snapshot of the current match state.

    Parameters
    ----------
    env_wrapper : DuckHuntEnvWrapper
        The training environment wrapper (from ``src.environment``).

    Returns
    -------
    dict
        Contains duck positions/velocities, round info, latency, and
        the RNG state so the simulation is deterministic.
    """
    inner = env_wrapper._env
    match = inner.round.current_match

    snap = {
        "duck_a": _snapshot_duck(match.duck_a),
        "round_number": inner.round_number,
        "bullets_remaining": match.bullets_remaining,
        "frames_elapsed": match.frames_elapsed,
        "latency_frames": inner.latency_frames,
        "rng_state": json.dumps(random.getstate(), default=_rng_serial),
    }
    if match.duck_b is not None:
        snap["duck_b"] = _snapshot_duck(match.duck_b)
    return snap


def _rng_serial(obj: Any) -> Any:
    """Make random.getstate() JSON-friendly (tuples → lists)."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def _rng_restore(state_json: str) -> None:
    """Restore ``random`` module state from JSON."""
    raw = json.loads(state_json)
    # random.setstate expects (int, tuple[int,...], float|None)
    raw[1] = tuple(raw[1])
    random.setstate(tuple(raw))


def _pil_to_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes for Arrow storage."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _strip_pil_from_messages(messages: list[dict]) -> list[dict]:
    """Remove PIL Image objects from message dicts for JSON serialisation.

    Replaces ``{"type": "image", "image": <PIL>}`` with ``{"type": "image"}``.
    """
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            clean_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    clean_content.append({"type": "image"})
                else:
                    clean_content.append(item)
            out.append({**msg, "content": clean_content})
        else:
            out.append(msg)
    return out


def reconstruct_prompt(example: dict) -> dict:
    """HF Dataset map function: inject PIL images back into prompt messages.

    The dataset stores prompts as JSON strings with ``{"type": "image"}``
    placeholders and images in a separate column.  This function
    reassembles the full multimodal messages that TRL / the processor
    can tokenise.
    """
    messages = json.loads(example["prompt"])
    images = example.get("images", [])

    img_idx = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if img_idx < len(images):
                        img = images[img_idx]
                        # HF Dataset Image feature returns PIL Images
                        if not isinstance(img, Image.Image):
                            img = Image.open(io.BytesIO(img))
                        item["image"] = img
                        img_idx += 1

    example["prompt"] = messages
    return example


# ===================================================================
#  Simulate a shot from a snapshot  (used by the reward function)
# ===================================================================
def simulate_shot(
    snapshot: dict,
    action: Action,
) -> dict:
    """Re-create the match from *snapshot*, advance, and fire.

    Parameters
    ----------
    snapshot : dict
        From ``capture_snapshot``.
    action : Action
        Parsed model output.

    Returns
    -------
    dict
        ``{hit_a: bool, hit_b: bool, had_target: bool}``
    """
    round_number = snapshot["round_number"]
    latency_frames = snapshot["latency_frames"]

    # Restore RNG so bounces are identical to the real env
    _rng_restore(snapshot["rng_state"])

    duck_a = _restore_duck(snapshot["duck_a"], round_number)
    has_duck_b = "duck_b" in snapshot
    duck_b = _restore_duck(snapshot["duck_b"], round_number) if has_duck_b else None

    # Check if ducks were flying at observation time (before processing)
    had_target_at_obs = duck_a.state == DuckState.FLYING
    if duck_b is not None:
        had_target_at_obs = had_target_at_obs or duck_b.state == DuckState.FLYING

    # Advance by latency + horizon (same as DuckHuntEnvironment.step)
    total_advance = latency_frames + action.horizon
    for _ in range(total_advance):
        duck_a.update(round_number)
        if duck_b is not None:
            duck_b.update(round_number)

    # Check duck states AFTER advancing (they might have escaped/fallen)
    had_target_at_shot = duck_a.state == DuckState.FLYING
    if duck_b is not None:
        had_target_at_shot = had_target_at_shot or duck_b.state == DuckState.FLYING

    # Convert normalised coords → pixels
    pixel_x = int(action.x * SCREEN_WIDTH)
    pixel_y = int(action.y * SCREEN_HEIGHT)

    hit_a = duck_a.check_hit(pixel_x, pixel_y)
    hit_b = duck_b.check_hit(pixel_x, pixel_y) if duck_b is not None else False

    # Return duck positions (normalised) for distance-based reward shaping
    duck_a_norm = (duck_a.x / SCREEN_WIDTH, duck_a.y / SCREEN_HEIGHT)
    duck_b_norm = (duck_b.x / SCREEN_WIDTH, duck_b.y / SCREEN_HEIGHT) if duck_b is not None else None

    return {
        "hit_a": hit_a, "hit_b": hit_b,
        "had_target": had_target_at_obs,
        "had_target_at_shot": had_target_at_shot,
        "duck_a_pos": duck_a_norm, "duck_b_pos": duck_b_norm,
        "duck_a_state": duck_a.state.value,
        "duck_b_state": duck_b.state.value if duck_b is not None else "none",
        "shot_pos": (action.x, action.y),
    }


# ===================================================================
#  Lazy transform for set_transform (never writes back to Arrow)
# ===================================================================
def _multimodal_transform(batch: dict) -> dict:
    """Lazy batch transform used by ``Dataset.set_transform``.

    Decodes JSON prompt strings and injects PIL images from the
    ``images`` column into ``{"type": "image"}`` placeholders.
    Called at access time so PIL objects never touch Arrow.
    """
    new_prompts = []
    for prompt_json, images in zip(batch["prompt"], batch["images"]):
        messages = json.loads(prompt_json)
        img_idx = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        if img_idx < len(images):
                            img = images[img_idx]
                            if not isinstance(img, Image.Image):
                                img = Image.open(io.BytesIO(img))
                            item["image"] = img
                            img_idx += 1
        new_prompts.append(messages)

    return {
        "prompt": new_prompts,
        "images": batch["images"],
        "snapshot": batch["snapshot"],
    }


# ===================================================================
#  6.1  Prompt dataset generation
# ===================================================================
class DuckHuntPromptGenerator:
    """Roll out the environment and collect prompt / snapshot pairs.

    Each sample contains:

    * **messages** – chat messages (system + user with images) ready for
      ``processor.apply_chat_template(messages, tools=tools, …)``
    * **images** – list of PIL.Image frames
    * **snapshot** – JSON string storing duck state for deterministic
      reward evaluation
    """

    def __init__(
        self,
        env_config: EnvironmentConfig,
        advance_range: tuple[int, int] = (5, 30),
    ) -> None:
        from .environment import DuckHuntEnvWrapper

        self.env = DuckHuntEnvWrapper(env_config)
        self.env_config = env_config
        self.advance_range = advance_range

    def generate(self, num_samples: int) -> Dataset:
        """Generate *num_samples* prompt samples.

        The environment is reset when it reaches ``done``, and advanced
        by a random number of frames between samples to create diverse
        game states.

        Returns
        -------
        datasets.Dataset
            Columns: ``prompt`` (JSON string), ``images`` (list of PIL
            images via HF Image feature), ``snapshot`` (JSON string).

            A ``set_transform`` is applied so that when TRL reads a row,
            the ``prompt`` is returned as a ``list[dict]`` with PIL
            images injected from the ``images`` column.  This avoids
            Arrow ever having to serialise PIL objects inside nested
            message dicts.
        """
        prompts: list[str] = []
        all_images: list[list[bytes]] = []
        snapshots: list[str] = []

        obs = self.env.reset()

        for i in range(num_samples):
            if self.env.is_done():
                obs = self.env.reset()

            # Capture frames and state
            frames = self.env.get_frames()
            state = self.env.get_state()

            # Only sample states where ducks are flying
            if state.get("ducks_flying", 0) == 0:
                self.env.advance_frames(
                    random.randint(*self.advance_range)
                )
                if self.env.is_done():
                    obs = self.env.reset()
                frames = self.env.get_frames()
                state = self.env.get_state()

            # Build prompt and strip PIL objects for Arrow storage
            messages, _tools = build_prompt(frames, state)
            serialisable_messages = _strip_pil_from_messages(messages)

            # Snapshot for deterministic reward
            snap = capture_snapshot(self.env)

            prompts.append(json.dumps(serialisable_messages))
            all_images.append([_pil_to_bytes(f) for f in frames])
            snapshots.append(json.dumps(snap))

            # Advance environment to create diverse states
            advance = random.randint(*self.advance_range)
            self.env.advance_frames(advance)

            if (i + 1) % 100 == 0:
                logger.info("Generated %d / %d samples", i + 1, num_samples)

        logger.info("Dataset generation complete: %d samples", num_samples)

        ds = Dataset.from_dict(
            {
                "prompt": prompts,
                "images": all_images,
                "snapshot": snapshots,
            },
            features=Features({
                "prompt": Value("string"),
                "images": Sequence(DatasetImage()),
                "snapshot": Value("string"),
            }),
        )

        # Apply a lazy transform so TRL gets fully reconstructed
        # multimodal prompts when it reads rows (never written to Arrow).
        ds.set_transform(_multimodal_transform)
        return ds


# ===================================================================
#  6.2  TRL-compatible reward function
# ===================================================================
def make_reward_function(reward_config: RewardConfig, max_horizon: int = MAX_HORIZON):
    """Return a reward callable compatible with ``GRPOTrainer.reward_funcs``.

    TRL calls the reward function with::

        reward_func(
            completions=["...", "..."],
            prompts=["...", "..."],
            snapshot=["...", "..."],   # extra dataset column
            **kwargs,
        )

    and expects a ``list[float]`` back.
    """

    def reward_func(completions, snapshot, **kwargs) -> list[float]:
        rewards: list[float] = []

        for completion_text, snap_json in zip(completions, snapshot):
            # Handle conversational format (list of message dicts)
            if isinstance(completion_text, list):
                # Extract text from the assistant message
                completion_text = completion_text[0].get("content", "")

            # 1. Parse model output → Action
            action = parse_tool_call(completion_text, max_horizon=max_horizon)

            # 2. Simulate shot from snapshot
            snap = json.loads(snap_json)
            if action is not None:
                result = simulate_shot(snap, action)
            else:
                result = {"hit_a": False, "hit_b": False, "had_target": True}

            # 3. Compute reward
            reward = compute_reward(result, action, reward_config)
            rewards.append(reward)

        return rewards

    # Give the function a name for TRL logging
    reward_func.__name__ = "duckhunt_accuracy"
    return reward_func


def make_format_reward_function(max_horizon: int = MAX_HORIZON):
    """Return a reward function that scores output format quality.

    Scoring:
      - 0.0: no parseable tool call at all
      - 1.0: valid tool call with concise output (just the call, minimal extra text)
      - 0.3–0.9: valid tool call but penalised for verbosity (extra explanation text)

    The verbosity penalty discourages the model from generating explanations,
    commentary, or filler text around the tool call.
    """
    import re

    # Tokens that are not "real" text (padding, special tokens, end markers)
    _STRIP_PAT = re.compile(
        r"<\|[^|]*\|>|<\|im_end\|>|<\|pad\|>|<\|tool_call_start\|>|<\|tool_call_end\|>"
        r"|\[TOOL_CALLS\]|\</s>",
    )

    def _count_extra_chars(text: str) -> int:
        """Count non-tool-call characters (text the model shouldn't be generating)."""
        # Strip special tokens and padding
        cleaned = _STRIP_PAT.sub("", text)
        # Strip the tool call itself (shoot(...) or JSON)
        cleaned = re.sub(r"shoot\s*\([^)]*\)", "", cleaned)
        cleaned = re.sub(r'\[\s*\{[^]]*\}\s*\]', "", cleaned)
        cleaned = re.sub(r'\{[^}]*\}', "", cleaned)
        # Strip whitespace and brackets
        cleaned = re.sub(r'[\s\[\]]', "", cleaned)
        return len(cleaned)

    def format_reward_func(completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion_text in completions:
            if isinstance(completion_text, list):
                completion_text = completion_text[0].get("content", "")

            action = parse_tool_call(completion_text, max_horizon=max_horizon)
            if action is None:
                rewards.append(0.0)
                continue

            # Valid tool call — now penalise verbosity
            extra = _count_extra_chars(completion_text)
            if extra <= 5:
                # Clean output: just the tool call
                rewards.append(1.0)
            else:
                # Penalty scales with extra text length, floors at 0.3
                penalty = min(0.7, extra * 0.01)
                rewards.append(max(0.3, 1.0 - penalty))

        return rewards

    format_reward_func.__name__ = "duckhunt_format"
    return format_reward_func
