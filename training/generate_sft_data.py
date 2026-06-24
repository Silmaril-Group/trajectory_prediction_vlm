"""Generate SFT dataset for Duck Hunt VLM training.

Each sample is ONE continuous game sequence:
  frame_0 → [skip frames] → frame_1 → [latency frames] → future_frame

The model sees frame_0 and frame_1, and must predict the duck hitbox
on the future_frame.

Usage:
    python generate_sft_data.py --num-samples 3000 --output sft_dataset
    python generate_sft_data.py --num-samples 3000 --latency-ms 200 --fixed-speed
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import random
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

# Make server package importable
_SERVER_DIR = Path(__file__).resolve().parent.parent / "duck_hunt_openenv" / "server"
sys.path.insert(0, str(_SERVER_DIR))

from game_engine import Duck, DuckState, Match, Round  # noqa: E402
from renderer import Renderer  # noqa: E402
from config import (  # noqa: E402
    SCREEN_WIDTH, SCREEN_HEIGHT,
    SPRITE_WIDTH, SPRITE_HEIGHT,
    HITBOX_WIDTH, HITBOX_HEIGHT,
    SPEED_BASE, SPEED_VARIANCE,
    FRAME_OUTPUT_SIZE, FPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_duck_visible(duck: Duck) -> bool:
    if duck.state != DuckState.FLYING:
        return False
    if duck.x + SPRITE_WIDTH < 0 or duck.x > SCREEN_WIDTH:
        return False
    if duck.y + SPRITE_HEIGHT < 0 or duck.y > SCREEN_HEIGHT:
        return False
    return True


def _hitbox_on_screen(duck: Duck) -> bool:
    """Check if the duck's hitbox is fully on screen."""
    hx = duck.x + (SPRITE_WIDTH - HITBOX_WIDTH) / 2
    hy = duck.y + (SPRITE_HEIGHT - HITBOX_HEIGHT) / 2
    if hx < 0 or hx + HITBOX_WIDTH > SCREEN_WIDTH:
        return False
    if hy < 0 or hy + HITBOX_HEIGHT > SCREEN_HEIGHT:
        return False
    return True


def _get_hitbox_normalized(duck: Duck) -> tuple[float, float, float, float]:
    """Get hitbox corners in normalized coords."""
    hx = duck.x + (SPRITE_WIDTH - HITBOX_WIDTH) / 2
    hy = duck.y + (SPRITE_HEIGHT - HITBOX_HEIGHT) / 2
    x1 = round(hx / SCREEN_WIDTH, 3)
    y1 = round(hy / SCREEN_HEIGHT, 3)
    x2 = round((hx + HITBOX_WIDTH) / SCREEN_WIDTH, 3)
    y2 = round((hy + HITBOX_HEIGHT) / SCREEN_HEIGHT, 3)
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Generate one sample: frame_0 → skip → frame_1 → latency → future_frame
# ---------------------------------------------------------------------------
def generate_one_sample(
    renderer: Renderer,
    latency_frames: int,
    frame_skip: int,
    fixed_speed: bool,
    single_duck: bool = False,
) -> dict | None:
    """Generate one SFT sample from a fresh match.

    Returns dict with frame_0, frame_1, future_frame, hitbox coords.
    Returns None if validation fails.
    """
    # Use round 1 for fixed speed, random for variable
    round_number = 1 if fixed_speed else random.randint(1, 3)
    match = Match(round_number)

    # Remove duck_b for single duck mode
    if single_duck and match.duck_b is not None:
        match.duck_b.state = DuckState.ESCAPED
        match.duck_b = None

    # Advance many frames so ducks move away from edges toward center
    # Wider range creates diverse positions across the screen
    advance = random.randint(15, 80)
    match.advance_frames(advance)

    # Pick a flying visible duck
    ducks = []
    if _is_duck_visible(match.duck_a):
        ducks.append(match.duck_a)
    if match.duck_b is not None and _is_duck_visible(match.duck_b):
        ducks.append(match.duck_b)

    if not ducks:
        return None

    target = random.choice(ducks)

    # --- Capture frame_0 ---
    game_state_0 = match.get_state()
    frame_0 = renderer.render_and_resize(game_state_0, advance).convert("RGB")
    frame_counter = advance

    # Verify target is visible at frame_0
    if not _is_duck_visible(target):
        return None

    # Current hitbox at frame_0 (for visualization)
    cur_hitbox = _get_hitbox_normalized(target)

    # --- Skip frames ---
    # Save RNG state before advancing (for deterministic replay)
    rng_before_skip = random.getstate()
    match.advance_frames(frame_skip)
    frame_counter += frame_skip

    # --- Capture frame_1 ---
    game_state_1 = match.get_state()
    frame_1 = renderer.render_and_resize(game_state_1, frame_counter).convert("RGB")

    # Verify target still flying and visible at frame_1
    if not _is_duck_visible(target):
        return None

    # --- Advance by latency frames ---
    match.advance_frames(latency_frames)
    frame_counter += latency_frames

    # --- Capture future_frame ---
    game_state_future = match.get_state()
    future_frame = renderer.render_and_resize(game_state_future, frame_counter).convert("RGB")

    # Verify target still flying, visible, hitbox on screen at future time
    if not _is_duck_visible(target):
        return None
    if not _hitbox_on_screen(target):
        return None

    # Ground truth: hitbox at future position
    gt_hitbox = _get_hitbox_normalized(target)

    # Flying duck count at frame_1 time
    flying_count = match.get_flying_count()

    return {
        "frame_0": frame_0,
        "frame_1": frame_1,
        "future_frame": future_frame,
        "gt_hitbox": gt_hitbox,
        "cur_hitbox": cur_hitbox,
        "latency_frames": latency_frames,
        "frame_skip": frame_skip,
        "round_number": round_number,
        "flying_count": max(flying_count, 1),
    }


# ---------------------------------------------------------------------------
# Build SFT training example
# ---------------------------------------------------------------------------
def build_sft_example(sample: dict) -> dict:
    """Format a sample into the SFT training format."""
    lat = sample["latency_frames"]
    x1, y1, x2, y2 = sample["gt_hitbox"]
    flying = sample["flying_count"]

    system_prompt = (
        "You are a Duck Hunt AI. Locate flying ducks by calling the locate tool.\n\n"
        f"You see 2 frames. Latency: {lat} frames.\n"
        "Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).\n"
        "Predict the hitbox of the duck after latency frames.\n"
        "x1,y1 = top-left corner. x2,y2 = bottom-right corner.\n\n"
        "IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."
    )

    # Few-shot
    fs_x1 = round(random.uniform(0.1, 0.7), 2)
    fs_y1 = round(random.uniform(0.1, 0.5), 2)
    fs_x2 = round(fs_x1 + 0.10, 2)
    fs_y2 = round(fs_y1 + 0.14, 2)

    user_text = f"2 frames, {flying} ducks flying, latency {lat} frames. Locate the duck."
    completion = f"<|tool_call_start|>[locate(x1={x1}, y1={y1}, x2={x2}, y2={y2})]<|tool_call_end|>"

    return {
        "system_prompt": system_prompt,
        "user_fewshot": f"Frame sequence: 2 frames. Ducks flying: 2. Latency: {lat} frames. Locate the duck.",
        "assistant_fewshot": f"<|tool_call_start|>[locate(x1={fs_x1}, y1={fs_y1}, x2={fs_x2}, y2={fs_y2})]<|tool_call_end|>",
        "user_text": user_text,
        "completion": completion,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cur_x1": sample["cur_hitbox"][0], "cur_y1": sample["cur_hitbox"][1],
        "cur_x2": sample["cur_hitbox"][2], "cur_y2": sample["cur_hitbox"][3],
        "latency_frames": lat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_dataset(
    num_samples: int = 3000,
    latency_ms: int = 200,
    frame_skip: int = 6,
    fps: int = 30,
    fixed_speed: bool = False,
    single_duck: bool = False,
    output_dir: str = "sft_dataset",
) -> None:
    latency_frames = int(latency_ms / 1000 * fps)
    renderer = Renderer(output_size=FRAME_OUTPUT_SIZE)

    logger.info("Generating SFT dataset:")
    logger.info("  Samples: %d", num_samples)
    logger.info("  Latency: %dms = %d frames", latency_ms, latency_frames)
    logger.info("  Frame skip: %d", frame_skip)
    logger.info("  Fixed speed: %s (round=1 only)", fixed_speed)
    logger.info("  Single duck: %s", single_duck)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    images_dir.mkdir(exist_ok=True)

    records = []
    generated = 0
    attempts = 0
    max_attempts = num_samples * 5

    while generated < num_samples and attempts < max_attempts:
        attempts += 1

        sample = generate_one_sample(renderer, latency_frames, frame_skip, fixed_speed, single_duck)
        if sample is None:
            continue

        idx = generated
        example = build_sft_example(sample)

        # Save 3 images: frame_0, frame_1, future_frame
        f0_path = images_dir / f"{idx:06d}_frame0.png"
        f1_path = images_dir / f"{idx:06d}_frame1.png"
        ff_path = images_dir / f"{idx:06d}_future.png"

        sample["frame_0"].save(str(f0_path))
        sample["frame_1"].save(str(f1_path))
        sample["future_frame"].save(str(ff_path))

        record = {
            "id": idx,
            "image_paths": [str(f0_path), str(f1_path)],
            "future_frame_path": str(ff_path),
            **{k: v for k, v in example.items()},
            "num_images": 2,
        }
        records.append(record)
        generated += 1

        if generated % 500 == 0:
            logger.info("  %d/%d generated (%d attempts)", generated, num_samples, attempts)

    # Save metadata
    meta_path = out_path / "dataset.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info("Done: %d samples in %d attempts (%.1f%% success)",
                generated, attempts, generated / max(attempts, 1) * 100)
    logger.info("Dataset saved to %s", out_path)

    if records:
        s = records[0]
        logger.info("Sample: lat=%df hitbox=(%.3f,%.3f)→(%.3f,%.3f)",
                     s["latency_frames"], s["x1"], s["y1"], s["x2"], s["y2"])


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Duck Hunt")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--output", type=str, default="sft_dataset")
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--latency-ms", type=int, default=200,
                        help="Single latency value in ms")
    parser.add_argument("--fixed-speed", action="store_true",
                        help="All ducks at round 1 speed (no speed increase)")
    parser.add_argument("--single-duck", action="store_true",
                        help="Only 1 duck per match (no ambiguity)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    generate_dataset(
        num_samples=args.num_samples,
        latency_ms=args.latency_ms,
        frame_skip=args.frame_skip,
        fixed_speed=args.fixed_speed,
        single_duck=args.single_duck,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
