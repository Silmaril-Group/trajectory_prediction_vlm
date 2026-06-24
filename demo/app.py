"""Duck Hunt VLM Demo — Gradio application for HuggingFace Spaces.

Runs a pre-rendered episode: the model plays Duck Hunt, frames are collected
with crosshair overlays, then encoded as an MP4 video for playback.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import gradio as gr
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from environment import DuckHuntEnvironment
from game_engine import DuckState
from game_config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    HITBOX_WIDTH,
    HITBOX_HEIGHT,
    FRAMES_PER_OBSERVATION,
    FPS,
)
from dataclasses import dataclass


@dataclass
class Action:
    x: float
    y: float
    horizon: int

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# ---------------------------------------------------------------------------
# Crosshair overlay
# ---------------------------------------------------------------------------
_crosshair_img: Image.Image | None = None


def _load_crosshair() -> Image.Image:
    global _crosshair_img
    if _crosshair_img is None:
        path = ASSETS_DIR / "crosshairs.png"
        _crosshair_img = Image.open(path).convert("RGBA")
        # Resize to 50x50 if not already
        if _crosshair_img.size != (50, 50):
            _crosshair_img = _crosshair_img.resize((50, 50), Image.LANCZOS)
    return _crosshair_img


def overlay_crosshair(
    frame: Image.Image,
    x_norm: float,
    y_norm: float,
) -> Image.Image:
    """Paste crosshair centered at normalised (x, y) on an 800x500 RGBA frame."""
    crosshair = _load_crosshair()
    px = int(x_norm * SCREEN_WIDTH)
    py = int(y_norm * SCREEN_HEIGHT)
    # Center the 50x50 crosshair
    paste_x = px - 25
    paste_y = py - 25
    out = frame.copy()
    out.paste(crosshair, (paste_x, paste_y), crosshair)
    return out


# ---------------------------------------------------------------------------
# Font for HIT/MISS text
# ---------------------------------------------------------------------------
_font: ImageFont.FreeTypeFont | None = None


def _load_font(size: int = 36) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    global _font
    if _font is None:
        font_path = ASSETS_DIR / "arcadeclassic.ttf"
        try:
            _font = ImageFont.truetype(str(font_path), size)
        except (OSError, IOError):
            _font = ImageFont.load_default()
    return _font


def _draw_result_text(frame: Image.Image, text: str, color: tuple) -> Image.Image:
    """Draw HIT/MISS text overlay at the top-center of the frame."""
    out = frame.copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(48)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (SCREEN_WIDTH - tw) // 2
    y = 20
    # Shadow
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 200), font=font)
    # Main text
    draw.text((x, y), text, fill=color, font=font)
    return out


# ---------------------------------------------------------------------------
# Perfect agent: peeks at game state, extrapolates duck position
# ---------------------------------------------------------------------------
def _pick_target_duck(match) -> object | None:
    """Return the best flying duck to aim at (prefer duck_a)."""
    if match.duck_a.state == DuckState.FLYING:
        return match.duck_a
    if match.duck_b.state == DuckState.FLYING:
        return match.duck_b
    return None


def perfect_agent_shot(env: DuckHuntEnvironment) -> Action:
    """Compute a near-perfect shot by reading actual duck state.

    Picks a short horizon, linearly extrapolates the duck's position
    forward by (latency + horizon) frames, aims at hitbox center,
    and adds small Gaussian noise so ~10% of shots miss.
    """
    match = env.round.current_match
    duck = _pick_target_duck(match)

    if duck is None:
        return Action(x=0.5, y=0.25, horizon=0)

    # Choose a small horizon (model-like range)
    horizon = np.random.randint(3, 8)
    total_frames = env.latency_frames + horizon

    # Linear extrapolation of duck position
    future_x = duck.x + duck.dx * total_frames
    future_y = duck.y + duck.dy * total_frames

    # Clamp to screen bounds (duck bounces, so clamp is an approximation)
    future_x = max(0, min(SCREEN_WIDTH - HITBOX_WIDTH, future_x))
    future_y = max(0, min(SCREEN_HEIGHT - HITBOX_HEIGHT, future_y))

    # Aim at hitbox center
    aim_x = future_x + HITBOX_WIDTH / 2
    aim_y = future_y + HITBOX_HEIGHT / 2

    # Add small Gaussian noise for realism (sigma ~15px ≈ ~10% miss)
    aim_x += np.random.normal(0, 15)
    aim_y += np.random.normal(0, 15)

    # Normalize to 0.0–1.0
    x_norm = max(0.0, min(1.0, aim_x / SCREEN_WIDTH))
    y_norm = max(0.0, min(1.0, aim_y / SCREEN_HEIGHT))

    return Action(
        x=round(x_norm, 3),
        y=round(y_norm, 3),
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    latency_ms: int = 300,
    max_shots: int = 30,
) -> tuple[list[Image.Image], dict]:
    """Run a full episode, collecting 800x500 display frames.

    Returns (frames_list, stats_dict).
    """
    env = DuckHuntEnvironment()

    # Force desired latency
    env.latency_ms = latency_ms
    env.latency_frames = int(latency_ms / 1000 * FPS)
    obs = env.reset()

    display_frames: list[Image.Image] = []
    total_hits = 0
    total_shots = 0
    total_double_kills = 0

    for shot_idx in range(max_shots):
        # Check if game is over
        if obs.get("done", False):
            break

        # Check if ducks are flying; if not, advance
        if obs.get("ducks_flying", 0) == 0:
            match = env.round.current_match
            match.advance_frames(15)
            env.frame_counter += 15
            env._update_frame_buffer()
            obs["frames"] = env.frame_buffer.copy()
            obs["ducks_flying"] = match.get_flying_count()

            if obs.get("ducks_flying", 0) == 0:
                match.advance_frames(30)
                env.frame_counter += 30
                env._update_frame_buffer()
                obs["frames"] = env.frame_buffer.copy()
                obs["ducks_flying"] = match.get_flying_count()
                if obs.get("ducks_flying", 0) == 0:
                    continue

        # Perfect agent: peek at game state and aim
        action = perfect_agent_shot(env)

        # Render the 800x500 display frame BEFORE the shot
        game_state = env.round.current_match.get_state()
        display_frame = env.renderer.render_frame(game_state, env.frame_counter)

        # Overlay crosshair at predicted position
        display_frame = overlay_crosshair(display_frame, action.x, action.y)

        # Add a few frames showing the crosshair (for visibility in video)
        for _ in range(3):
            display_frames.append(display_frame)

        # Execute the shot
        shot_action = {"x": action.x, "y": action.y, "horizon": action.horizon}
        obs = env.step(shot_action)
        total_shots += 1

        result = obs.get("last_action_result", "miss")
        ducks_hit = obs.get("last_ducks_hit", 0)

        # Determine result text and color
        if result == "double_kill":
            text, color = "DOUBLE  KILL!", (255, 215, 0, 255)
            total_hits += 2
            total_double_kills += 1
        elif result == "hit":
            text, color = "HIT!", (0, 255, 0, 255)
            total_hits += ducks_hit
        elif result == "no_target":
            text, color = "NO  TARGET", (128, 128, 128, 255)
        else:
            text, color = "MISS", (255, 50, 50, 255)

        # Render post-shot frames with result text
        for i in range(8):
            post_state = env.round.current_match.get_state()
            post_frame = env.renderer.render_frame(post_state, env.frame_counter + i)
            if i < 6:
                post_frame = _draw_result_text(post_frame, text, color)
            display_frames.append(post_frame)

        # Advance a few frames for animation
        env.round.current_match.advance_frames(4)
        env.frame_counter += 4

        logger.info(
            "Shot %d: action=(%.2f, %.2f, h=%d) result=%s",
            shot_idx + 1, action.x, action.y, action.horizon, result,
        )

    hit_rate = total_hits / max(total_shots, 1)
    stats = {
        "total_shots": total_shots,
        "total_hits": total_hits,
        "double_kills": total_double_kills,
        "hit_rate": f"{hit_rate:.1%}",
        "latency_ms": latency_ms,
        "round_reached": obs.get("round_number", 1),
    }

    return display_frames, stats


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------
def frames_to_video(
    frames: list[Image.Image],
    fps: int = 15,
) -> str:
    """Encode PIL frames into an MP4 file, return the file path."""
    if not frames:
        return ""

    # Convert RGBA → RGB numpy arrays
    np_frames = []
    for f in frames:
        rgb = f.convert("RGB")
        np_frames.append(np.array(rgb))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    iio.imwrite(
        tmp.name,
        np_frames,
        fps=fps,
        codec="libx264",
        plugin="pyav",
    )

    logger.info("Video saved: %s (%d frames, %d fps)", tmp.name, len(np_frames), fps)
    return tmp.name


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------
def play_episode(latency_ms: int, max_shots: int) -> tuple[str, str]:
    """Gradio callback: run episode and return (video_path, stats_text)."""
    frames, stats = run_episode(
        latency_ms=int(latency_ms),
        max_shots=int(max_shots),
    )

    video_path = frames_to_video(frames, fps=15)

    stats_text = (
        f"Shots: {stats['total_shots']}  |  "
        f"Hits: {stats['total_hits']}  |  "
        f"Hit Rate: {stats['hit_rate']}  |  "
        f"Double Kills: {stats['double_kills']}  |  "
        f"Latency: {stats['latency_ms']}ms  |  "
        f"Round: {stats['round_reached']}"
    )

    return video_path, stats_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Duck Hunt VLM Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Duck Hunt VLM Demo\n"
            "Watch an AI agent play Duck Hunt!\n\n"
            "The agent observes duck positions, predicts where each duck will be after "
            "accounting for processing latency, and fires. "
            "Crosshairs show the predicted shot position.\n\n"
            "**Trained model**: [`dmayboroda/dh_ministal_gpro`](https://huggingface.co/dmayboroda/dh_ministal_gpro) "
            "(Ministral-3B + LoRA, 60.9% hit rate via GRPO training)"
        )

        with gr.Row():
            latency_slider = gr.Slider(
                minimum=100, maximum=600, value=300, step=100,
                label="Simulated Latency (ms)",
                info="Processing delay the model must predict ahead for",
            )
            shots_slider = gr.Slider(
                minimum=5, maximum=50, value=30, step=5,
                label="Max Shots",
                info="Maximum number of shots per episode",
            )

        play_btn = gr.Button("Play Episode", variant="primary", size="lg")

        video_output = gr.Video(label="Episode Replay", autoplay=True)
        stats_output = gr.Textbox(label="Episode Stats", interactive=False)

        play_btn.click(
            fn=play_episode,
            inputs=[latency_slider, shots_slider],
            outputs=[video_output, stats_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch()
