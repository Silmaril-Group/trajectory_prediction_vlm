"""Duck Hunt OpenEnv Environment"""

import random

from game_engine import Round, DuckState
from renderer import Renderer
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FRAMES_PER_OBSERVATION,
    FRAME_OUTPUT_SIZE,
    FRAME_SKIP,
    LATENCY_OPTIONS_MS,
    FPS,
    MAX_HORIZON,
    MAX_MISSES,
    DUCKS_PER_MATCH,
    REWARD_HIT,
    REWARD_DOUBLE_KILL,
    REWARD_MISS,
    REWARD_SHOOT_NOTHING,
    LAMBDA_HORIZON,
    BONUS_PERFECT_MATCH,
    BONUS_PERFECT_ROUND,
)


class DuckHuntEnvironment:
    """OpenEnv-compatible Duck Hunt environment."""

    def __init__(self, output_size: tuple[int, int] = FRAME_OUTPUT_SIZE, num_frames: int | None = None, frame_skip: int | None = None):
        # Game state
        self.round: Round | None = None
        self.round_number: int = 1
        self.total_misses: int = 0
        self.frame_counter: int = 0

        # Renderer with configurable output size
        self.output_size = output_size
        self.renderer = Renderer(output_size=output_size)

        # Configurable number of observation frames and skip between them
        self.num_frames = num_frames if num_frames is not None else FRAMES_PER_OBSERVATION
        self.frame_skip = frame_skip if frame_skip is not None else FRAME_SKIP

        # Frame buffer (last N frames as base64)
        self.frame_buffer: list[str] = []

        # Latency simulation (set on reset)
        self.latency_ms: int | None = None
        self.latency_frames: int = 0

    def reset(self) -> dict:
        """Reset the environment and return initial observation."""
        # Reset game state
        self.round_number = 1
        self.total_misses = 0
        self.frame_counter = 0
        self.round = Round(self.round_number)

        # Randomly select simulated latency
        self.latency_ms = random.choice(LATENCY_OPTIONS_MS)
        self.latency_frames = int(self.latency_ms / 1000 * FPS)

        # Clear frame buffer
        self.frame_buffer = []

        # Render initial frames (with skip for visible duck displacement)
        for i in range(self.num_frames):
            game_state = self.round.current_match.get_state()
            image = self.renderer.render_and_resize(game_state, self.frame_counter)
            frame_b64 = self.renderer.image_to_base64(image)
            self.frame_buffer.append(frame_b64)
            # Advance by FRAME_SKIP between captures (1 after last to not waste frames)
            advance = self.frame_skip if i < self.num_frames - 1 else 1
            self.round.current_match.advance_frames(advance)
            self.frame_counter += advance

        # Build observation
        return self._build_observation(
            reward=0.0,
            done=False,
            last_action_result=None,
            last_ducks_hit=0,
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        last_action_result: str | None,
        last_ducks_hit: int,
    ) -> dict:
        """Build observation dict from current state."""
        match = self.round.current_match

        return {
            "frames": self.frame_buffer.copy(),
            "num_frames": len(self.frame_buffer),
            "round_number": self.round_number,
            "match_number": self.round.matches_completed + 1,
            "ducks_flying": match.get_flying_count(),
            "bullets_remaining": match.bullets_remaining,
            "match_ducks_hit": match.ducks_hit,
            "round_ducks_hit": self.round.total_ducks_hit + match.ducks_hit,
            "total_misses": self.total_misses,
            "processing_latency_ms": self.latency_ms,
            "last_action_result": last_action_result,
            "last_ducks_hit": last_ducks_hit,
            "reward": reward,
            "done": done,
        }

    def step(self, action: dict) -> dict:
        """Execute action and return observation."""
        # Get normalized coords (0.0-1.0)
        x_norm = action.get("x", 0.0)
        y_norm = action.get("y", 0.0)
        horizon = action.get("horizon", 0)

        # 1. VALIDATE ACTION - clamp normalized coords to valid range
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        horizon = max(0, min(horizon, MAX_HORIZON))

        # Convert to pixel coordinates
        x = int(x_norm * SCREEN_WIDTH)   # 0.0-1.0 -> 0-800
        y = int(y_norm * SCREEN_HEIGHT)  # 0.0-1.0 -> 0-500

        match = self.round.current_match

        # 2. CHECK IF DUCKS FLYING (before advancing)
        flying_before = match.get_flying_count()

        # 3. ADVANCE GAME BY LATENCY + HORIZON
        # Game was paused during VLM processing, now advance
        total_advance = self.latency_frames + horizon
        match.advance_frames(total_advance)
        self.frame_counter += total_advance

        # 4. PROCESS SHOT
        if flying_before == 0:
            # No ducks were flying when observation was taken
            hit_a, hit_b = False, False
            action_result = "no_target"
        else:
            hit_a, hit_b = match.process_shot(x, y)

            if hit_a and hit_b:
                action_result = "double_kill"
            elif hit_a or hit_b:
                action_result = "hit"
            else:
                action_result = "miss"

        ducks_hit = int(hit_a) + int(hit_b)

        # 5. CALCULATE REWARD
        had_target = flying_before > 0
        reward = self._calculate_reward(hit_a, hit_b, had_target, horizon)

        # 6. CHECK MATCH COMPLETION
        done = False
        if match.is_complete:
            # Update misses before advancing
            match_misses = DUCKS_PER_MATCH - match.ducks_hit
            self.total_misses += match_misses

            # Check game over
            if self.total_misses >= MAX_MISSES:
                done = True
            else:
                # Advance to next match (auto-skip dog scene)
                self.round.advance_to_next_match()

                # Check round completion
                if self.round.is_complete:
                    # Bonus for perfect round
                    if self.round.get_misses() == 0:
                        reward += BONUS_PERFECT_ROUND

                    # Start new round
                    self.round_number += 1
                    self.round = Round(self.round_number)

        # 7. UPDATE FRAME BUFFER
        self._update_frame_buffer()

        # 8. BUILD AND RETURN OBSERVATION
        return self._build_observation(
            reward=reward,
            done=done,
            last_action_result=action_result,
            last_ducks_hit=ducks_hit,
        )

    def _calculate_reward(
        self,
        hit_a: bool,
        hit_b: bool,
        had_target: bool,
        horizon: int,
    ) -> float:
        """Calculate reward based on hit results and horizon."""
        # No target was flying
        if not had_target:
            return REWARD_SHOOT_NOTHING

        # Double kill
        if hit_a and hit_b:
            base = REWARD_DOUBLE_KILL
            penalty = LAMBDA_HORIZON * (horizon / MAX_HORIZON)
            return base - penalty

        # Single hit
        if hit_a or hit_b:
            base = REWARD_HIT
            penalty = LAMBDA_HORIZON * (horizon / MAX_HORIZON)
            return base - penalty

        # Miss (no horizon penalty)
        return REWARD_MISS

    def _update_frame_buffer(self):
        """Render new frames and update buffer (with skip for visible displacement)."""
        self.frame_buffer = []

        for i in range(self.num_frames):
            game_state = self.round.current_match.get_state()
            image = self.renderer.render_and_resize(game_state, self.frame_counter)
            frame_b64 = self.renderer.image_to_base64(image)
            self.frame_buffer.append(frame_b64)
            advance = self.frame_skip if i < self.num_frames - 1 else 1
            self.round.current_match.advance_frames(advance)
            self.frame_counter += advance
