"""Training-side wrapper around the Duck Hunt OpenEnv environment.

Imports the server-side ``DuckHuntEnvironment`` directly (no HTTP
overhead) and exposes a clean API for the GRPO training loop.
"""

from __future__ import annotations

import base64
import logging
import random
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

from .config import EnvironmentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Make the server package importable
# ---------------------------------------------------------------------------
_SERVER_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "duck_hunt_openenv"
    / "server"
)
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from environment import DuckHuntEnvironment  # noqa: E402
from game_engine import DuckState  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64_to_pil(b64: str) -> Image.Image:
    """Decode a base-64 PNG string into a PIL Image (RGB)."""
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class DuckHuntEnvWrapper:
    """Training wrapper around ``DuckHuntEnvironment``.

    Differences from the raw environment:
    * Frames are returned as ``PIL.Image`` objects, not base-64 strings.
    * Episode-level statistics are tracked automatically.
    * The latency / observation-count knobs are exposed via
      ``EnvironmentConfig``.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

        self._env = DuckHuntEnvironment(
            output_size=tuple(config.frame_output_size),
            num_frames=config.frames_per_observation,
            frame_skip=config.frame_skip,
        )

        # ---- episode stats (reset each episode) ----
        self._total_shots: int = 0
        self._total_hits: int = 0
        self._total_double_kills: int = 0
        self._total_misses_shot: int = 0  # shots that missed
        self._total_no_target: int = 0
        self._total_reward: float = 0.0
        self._steps: int = 0
        self._done: bool = False

        # ---- latest observation cache ----
        self._last_obs: dict | None = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Reset environment for a new episode.

        Returns the initial observation dict (same schema as ``step``).
        """
        # Override latency selection with our config options
        self._env.latency_ms = random.choice(self.config.latency_options_ms)
        self._env.latency_frames = int(
            self._env.latency_ms / 1000 * self.config.fps
        )

        obs = self._env.reset()

        # Randomize starting round (1-5) so ducks have varied speeds
        start_round = random.randint(1, 5)
        if start_round > 1:
            self._env.round_number = start_round
            from game_engine import Round
            self._env.round = Round(start_round)
            self._env._update_frame_buffer()
            obs["frames"] = self._env.frame_buffer.copy()

        # Reset episode tracking
        self._total_shots = 0
        self._total_hits = 0
        self._total_double_kills = 0
        self._total_misses_shot = 0
        self._total_no_target = 0
        self._total_reward = 0.0
        self._steps = 0
        self._done = False
        self._last_obs = obs

        return obs

    # ------------------------------------------------------------------
    # step  (accepts normalised coords, like the raw env)
    # ------------------------------------------------------------------
    def step(self, x_norm: float, y_norm: float, horizon: int = 0) -> dict:
        """Execute one shot and return the new observation.

        Parameters
        ----------
        x_norm : float
            Normalised horizontal coordinate (0.0 – 1.0).
        y_norm : float
            Normalised vertical coordinate (0.0 – 1.0).
        horizon : int
            Frames to predict ahead (0 – ``config.max_horizon``).

        Returns
        -------
        dict
            Observation from the underlying ``DuckHuntEnvironment.step``.
        """
        action = {
            "x": x_norm,
            "y": y_norm,
            "horizon": horizon,
        }
        obs = self._env.step(action)
        self._last_obs = obs

        # ---- bookkeeping ----
        self._steps += 1
        self._total_shots += 1
        self._total_reward += obs["reward"]

        result = obs.get("last_action_result")
        ducks_hit = obs.get("last_ducks_hit", 0)
        if result == "double_kill":
            self._total_hits += 2
            self._total_double_kills += 1
        elif result == "hit":
            self._total_hits += ducks_hit
        elif result == "miss":
            self._total_misses_shot += 1
        elif result == "no_target":
            self._total_no_target += 1

        self._done = obs.get("done", False)
        return obs

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def get_frames(self, n: int | None = None) -> list[Image.Image]:
        """Return the last *n* frames as PIL Images.

        If *n* is ``None``, return all frames in the current buffer
        (typically ``frames_per_observation``).
        """
        if self._last_obs is None:
            return []

        b64_frames: list[str] = self._last_obs.get("frames", [])
        if n is not None:
            b64_frames = b64_frames[-n:]

        return [_b64_to_pil(f) for f in b64_frames]

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Return a summary of the current game state."""
        if self._last_obs is None:
            return {}

        return {
            "bullets_remaining": self._last_obs.get("bullets_remaining", 0),
            "ducks_flying": self._last_obs.get("ducks_flying", 0),
            "round_number": self._last_obs.get("round_number", 1),
            "match_number": self._last_obs.get("match_number", 1),
            "simulated_latency_ms": self._last_obs.get(
                "processing_latency_ms", 0
            ),
            "simulated_latency_frames": self._env.latency_frames,
            "is_done": self._done,
        }

    def get_flying_count(self) -> int:
        """Return the number of ducks currently flying."""
        if self._last_obs is None:
            return 0
        return self._last_obs.get("ducks_flying", 0)

    def is_done(self) -> bool:
        """Return ``True`` when the episode has ended."""
        return self._done

    # ------------------------------------------------------------------
    # Low-level pass-throughs (useful for custom reward logic)
    # ------------------------------------------------------------------
    def advance_frames(self, n: int) -> None:
        """Advance the underlying game by *n* frames without shooting."""
        match = self._env.round.current_match
        match.advance_frames(n)
        self._env.frame_counter += n
        self._env._update_frame_buffer()
        # refresh cached obs frames
        if self._last_obs is not None:
            self._last_obs["frames"] = self._env.frame_buffer.copy()
            self._last_obs["ducks_flying"] = match.get_flying_count()

    def process_shot(self, pixel_x: int, pixel_y: int) -> dict:
        """Fire at pixel coordinates and return hit info (no frame advance)."""
        match = self._env.round.current_match
        flying_before = match.get_flying_count()
        hit_a, hit_b = match.process_shot(pixel_x, pixel_y)
        return {
            "hit_a": hit_a,
            "hit_b": hit_b,
            "had_target": flying_before > 0,
        }

    def auto_advance_to_next_match(self) -> None:
        """Skip the dog scene and start the next match."""
        rnd = self._env.round
        if rnd is not None and not rnd.is_complete:
            rnd.advance_to_next_match()
            if rnd.is_complete:
                self._env.round_number += 1
                from game_engine import Round  # noqa: E402 (already on path)

                self._env.round = Round(self._env.round_number)
        # Re-render so frames reflect the new match
        self._env._update_frame_buffer()
        if self._last_obs is not None:
            self._last_obs["frames"] = self._env.frame_buffer.copy()

    # ------------------------------------------------------------------
    # Episode statistics
    # ------------------------------------------------------------------
    def get_episode_stats(self) -> dict:
        """Return aggregate statistics for the current episode."""
        hit_rate = (
            self._total_hits / self._total_shots
            if self._total_shots > 0
            else 0.0
        )
        return {
            "total_shots": self._total_shots,
            "total_hits": self._total_hits,
            "total_double_kills": self._total_double_kills,
            "total_misses_shot": self._total_misses_shot,
            "total_no_target": self._total_no_target,
            "hit_rate": hit_rate,
            "total_reward": self._total_reward,
            "steps": self._steps,
            "done": self._done,
            "round_number": self._last_obs.get("round_number", 1)
            if self._last_obs
            else 1,
            "total_misses_game": self._last_obs.get("total_misses", 0)
            if self._last_obs
            else 0,
        }
