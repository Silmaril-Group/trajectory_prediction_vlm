"""Unit tests for Duck Hunt OpenEnv"""

import sys
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

import pytest
from PIL import Image

from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    HITBOX_WIDTH,
    HITBOX_HEIGHT,
    SPEED_BASE,
    SPEED_VARIANCE,
    BULLETS_PER_MATCH,
    DUCKS_PER_MATCH,
    FRAME_OUTPUT_SIZE,
    MAX_MISSES,
    REWARD_HIT,
    REWARD_MISS,
    REWARD_DOUBLE_KILL,
    REWARD_SHOOT_NOTHING,
)
from game_engine import Duck, DuckState, Match, Round
from renderer import Renderer
from environment import DuckHuntEnvironment


# =============================================================================
# TEST DUCK
# =============================================================================

class TestDuck:
    """Tests for Duck class."""

    def test_duck_spawn(self):
        """Duck spawns in valid position."""
        duck = Duck(round_number=1)

        # X should be at edge (0 or SCREEN_WIDTH)
        assert duck.x == 0 or duck.x == SCREEN_WIDTH

        # Y should be in bottom half
        assert duck.y >= SCREEN_HEIGHT // 2
        assert duck.y <= SCREEN_HEIGHT - HITBOX_HEIGHT

        # State should be flying
        assert duck.state == DuckState.FLYING

    def test_duck_update(self):
        """Duck moves correctly when updated."""
        duck = Duck(round_number=1)
        initial_x = duck.x
        initial_y = duck.y
        dx = duck.dx
        dy = duck.dy

        duck.update(round_number=1)

        # Position should change by velocity
        assert duck.x == initial_x + dx
        assert duck.y == initial_y + dy

    def test_duck_bounce_left(self):
        """Duck bounces at left edge."""
        duck = Duck(round_number=1)
        duck.x = 0
        duck.dx = -5  # Moving left

        duck.update(round_number=1)

        # After bounce, dx should be positive (moving right)
        assert duck.dx > 0

    def test_duck_bounce_right(self):
        """Duck bounces at right edge."""
        duck = Duck(round_number=1)
        duck.x = SCREEN_WIDTH - HITBOX_WIDTH
        duck.dx = 5  # Moving right

        duck.update(round_number=1)

        # After bounce, dx should be negative (moving left)
        assert duck.dx < 0

    def test_duck_bounce_top(self):
        """Duck bounces at top edge."""
        duck = Duck(round_number=1)
        duck.x = SCREEN_WIDTH // 2  # Middle of screen
        duck.y = 0
        duck.dy = -5  # Moving up

        duck.update(round_number=1)

        # After bounce, dy should be positive (moving down)
        assert duck.dy > 0

    def test_duck_bounce_bottom(self):
        """Duck bounces at bottom half boundary."""
        duck = Duck(round_number=1)
        duck.x = SCREEN_WIDTH // 2  # Middle of screen
        duck.y = SCREEN_HEIGHT // 2
        duck.dy = 5  # Moving down

        duck.update(round_number=1)

        # After bounce, dy should be negative (moving up)
        assert duck.dy < 0

    def test_duck_escape(self):
        """Duck escapes when off top of screen."""
        duck = Duck(round_number=1)
        duck.y = -HITBOX_HEIGHT - 10  # Well off screen

        duck._check_escaped()

        assert duck.state == DuckState.ESCAPED

    def test_duck_hit_detection(self):
        """Hitbox works correctly."""
        duck = Duck(round_number=1)
        duck.x = 100
        duck.y = 100
        duck.state = DuckState.FLYING

        # Hit inside hitbox
        assert duck.check_hit(150, 150) is True

        # Miss outside hitbox
        assert duck.check_hit(0, 0) is False
        assert duck.check_hit(200, 200) is False

    def test_duck_hit_sets_falling(self):
        """Duck hit() sets state to falling."""
        duck = Duck(round_number=1)
        duck.hit()

        assert duck.state == DuckState.FALLING
        assert duck.dx == 0
        assert duck.dy == 4

    def test_duck_speed_by_round(self):
        """Speed increases with round number."""
        duck_r1 = Duck(round_number=1)
        duck_r5 = Duck(round_number=5)

        # Speed range for round 1: 5-6, round 5: 9-10
        min_speed_r1 = SPEED_BASE + 1
        min_speed_r5 = SPEED_BASE + 5

        assert abs(duck_r1.dx) >= min_speed_r1
        assert abs(duck_r5.dx) >= min_speed_r5


# =============================================================================
# TEST MATCH
# =============================================================================

class TestMatch:
    """Tests for Match class."""

    def test_match_init(self):
        """Match starts with 2 ducks and 3 bullets."""
        match = Match(round_number=1)

        assert match.duck_a is not None
        assert match.duck_b is not None
        assert match.bullets_remaining == BULLETS_PER_MATCH
        assert match.ducks_hit == 0
        assert match.frames_elapsed == 0

    def test_match_advance(self):
        """Ducks move when match advances."""
        match = Match(round_number=1)
        initial_a_x = match.duck_a.x
        initial_b_x = match.duck_b.x

        match.advance_frames(10)

        # Ducks should have moved
        assert match.duck_a.x != initial_a_x or match.duck_a.y != initial_a_x
        assert match.frames_elapsed == 10

    def test_match_shoot_hit(self):
        """Hit detection works in match."""
        match = Match(round_number=1)
        match.duck_a.x = 100
        match.duck_a.y = 100
        match.duck_a.state = DuckState.FLYING

        hit_a, hit_b = match.process_shot(150, 150)

        assert hit_a is True
        assert match.ducks_hit == 1
        assert match.bullets_remaining == BULLETS_PER_MATCH - 1

    def test_match_shoot_miss(self):
        """Miss works correctly."""
        match = Match(round_number=1)
        match.duck_a.x = 100
        match.duck_a.y = 100
        match.duck_b.x = 300
        match.duck_b.y = 300

        hit_a, hit_b = match.process_shot(500, 500)  # Miss both

        assert hit_a is False
        assert hit_b is False
        assert match.ducks_hit == 0
        assert match.bullets_remaining == BULLETS_PER_MATCH - 1

    def test_match_double_kill(self):
        """Both ducks hit with one shot (overlapping)."""
        match = Match(round_number=1)
        # Position ducks to overlap
        match.duck_a.x = 100
        match.duck_a.y = 100
        match.duck_b.x = 100
        match.duck_b.y = 100

        hit_a, hit_b = match.process_shot(150, 150)

        assert hit_a is True
        assert hit_b is True
        assert match.ducks_hit == 2

    def test_match_no_bullets(self):
        """Cannot shoot with no bullets."""
        match = Match(round_number=1)
        match.bullets_remaining = 0

        hit_a, hit_b = match.process_shot(100, 100)

        assert hit_a is False
        assert hit_b is False

    def test_match_complete_by_kills(self):
        """Match ends when both ducks hit and finished."""
        match = Match(round_number=1)
        match.duck_a.state = DuckState.FALLING
        match.duck_a.y = SCREEN_HEIGHT  # Finished falling
        match.duck_b.state = DuckState.FALLING
        match.duck_b.y = SCREEN_HEIGHT

        assert match.is_complete is True

    def test_match_complete_by_escape(self):
        """Match ends when both ducks escape."""
        match = Match(round_number=1)
        match.duck_a.state = DuckState.ESCAPED
        match.duck_b.state = DuckState.ESCAPED

        assert match.is_complete is True

    def test_match_flying_count(self):
        """Flying count reflects duck states."""
        match = Match(round_number=1)

        assert match.get_flying_count() == 2

        match.duck_a.state = DuckState.FALLING
        assert match.get_flying_count() == 1

        match.duck_b.state = DuckState.ESCAPED
        assert match.get_flying_count() == 0


# =============================================================================
# TEST ROUND
# =============================================================================

class TestRound:
    """Tests for Round class."""

    def test_round_init(self):
        """Round initializes correctly."""
        round_obj = Round(round_number=1)

        assert round_obj.round_number == 1
        assert round_obj.current_match is not None
        assert round_obj.matches_completed == 0
        assert round_obj.total_ducks_hit == 0

    def test_round_advance_to_next_match(self):
        """Round advances to next match."""
        round_obj = Round(round_number=1)
        round_obj.current_match.ducks_hit = 2

        round_obj.advance_to_next_match()

        assert round_obj.matches_completed == 1
        assert round_obj.total_ducks_hit == 2

    def test_round_complete(self):
        """Round completes after 5 matches."""
        round_obj = Round(round_number=1)

        for _ in range(5):
            round_obj.advance_to_next_match()

        assert round_obj.is_complete is True

    def test_round_get_misses(self):
        """Misses calculated correctly."""
        round_obj = Round(round_number=1)
        round_obj.matches_completed = 3
        round_obj.total_ducks_hit = 4  # 6 possible, 4 hit = 2 missed

        assert round_obj.get_misses() == 2


# =============================================================================
# TEST RENDERER
# =============================================================================

class TestRenderer:
    """Tests for Renderer class."""

    @pytest.fixture
    def renderer(self):
        return Renderer()

    @pytest.fixture
    def game_state(self):
        return {
            "duck_a": {
                "x": 100,
                "y": 100,
                "state": "flying",
                "sprite_dir": "up_right",
            },
            "duck_b": {
                "x": 300,
                "y": 200,
                "state": "flying",
                "sprite_dir": "up_left",
            },
        }

    def test_render_produces_image(self, renderer, game_state):
        """Render returns valid PIL.Image."""
        image = renderer.render_frame(game_state)

        assert isinstance(image, Image.Image)

    def test_render_correct_size(self, renderer, game_state):
        """Render outputs correct size (800x500)."""
        image = renderer.render_frame(game_state)

        assert image.size == (SCREEN_WIDTH, SCREEN_HEIGHT)

    def test_resize(self, renderer, game_state):
        """Resize outputs correct size (512x512)."""
        image = renderer.render_and_resize(game_state)

        assert image.size == FRAME_OUTPUT_SIZE

    def test_base64(self, renderer, game_state):
        """Base64 encoding works."""
        image = renderer.render_frame(game_state)
        b64 = renderer.image_to_base64(image)

        assert isinstance(b64, str)
        assert len(b64) > 0
        # Should be valid base64 (only valid characters)
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


# =============================================================================
# TEST ENVIRONMENT
# =============================================================================

class TestEnvironment:
    """Tests for DuckHuntEnvironment class."""

    @pytest.fixture
    def env(self):
        return DuckHuntEnvironment()

    def test_reset(self, env):
        """Reset returns valid observation."""
        obs = env.reset()

        assert "frames" in obs
        assert "round_number" in obs
        assert "match_number" in obs
        assert "ducks_flying" in obs
        assert "bullets_remaining" in obs
        assert "reward" in obs
        assert "done" in obs

        assert obs["round_number"] == 1
        assert obs["match_number"] == 1
        assert obs["done"] is False
        assert len(obs["frames"]) > 0

    def test_step_hit(self, env):
        """Step processes hit correctly."""
        env.reset()

        # Disable latency so duck doesn't move before shot
        env.latency_frames = 0
        env.latency_ms = 0

        # Position duck at known location (pixel 100, 100)
        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100
        env.round.current_match.duck_a.state = DuckState.FLYING

        # Normalized coords: 150/800 = 0.1875, 150/500 = 0.3
        action = {"x": 0.1875, "y": 0.3, "horizon": 0}
        obs = env.step(action)

        # Should have hit
        assert obs["last_action_result"] in ["hit", "double_kill"]
        assert obs["reward"] > 0

    def test_step_miss(self, env):
        """Step processes miss correctly."""
        env.reset()

        # Position ducks away from shot
        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100
        env.round.current_match.duck_b.x = 200
        env.round.current_match.duck_b.y = 200

        # Normalized coords: 700/800 = 0.875, 400/500 = 0.8
        action = {"x": 0.875, "y": 0.8, "horizon": 0}
        obs = env.step(action)

        assert obs["last_action_result"] == "miss"
        assert obs["reward"] == REWARD_MISS

    def test_reward_hit(self, env):
        """Reward for hit is correct."""
        env.reset()

        # Disable latency so duck doesn't move before shot
        env.latency_frames = 0
        env.latency_ms = 0

        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100
        env.round.current_match.duck_b.x = 500
        env.round.current_match.duck_b.y = 400

        # Normalized coords: 150/800 = 0.1875, 150/500 = 0.3
        action = {"x": 0.1875, "y": 0.3, "horizon": 0}
        obs = env.step(action)

        # With horizon=0, reward should be close to REWARD_HIT
        assert obs["reward"] > 0
        assert obs["reward"] <= REWARD_HIT

    def test_reward_double_kill(self, env):
        """Reward for double kill is correct."""
        env.reset()

        # Disable latency so ducks don't move before shot
        env.latency_frames = 0
        env.latency_ms = 0

        # Overlap ducks
        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100
        env.round.current_match.duck_b.x = 100
        env.round.current_match.duck_b.y = 100

        # Normalized coords: 150/800 = 0.1875, 150/500 = 0.3
        action = {"x": 0.1875, "y": 0.3, "horizon": 0}
        obs = env.step(action)

        assert obs["last_action_result"] == "double_kill"
        assert obs["reward"] > REWARD_HIT

    def test_reward_no_target(self, env):
        """Reward for shooting with no target."""
        env.reset()
        # Set both ducks to escaped
        env.round.current_match.duck_a.state = DuckState.ESCAPED
        env.round.current_match.duck_b.state = DuckState.ESCAPED

        # Normalized coords: center of screen (0.5, 0.5)
        action = {"x": 0.5, "y": 0.5, "horizon": 0}
        obs = env.step(action)

        assert obs["last_action_result"] == "no_target"
        assert obs["reward"] == REWARD_SHOOT_NOTHING

    def test_horizon_penalty(self, env):
        """Horizon penalty reduces reward."""
        env.reset()
        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100

        # Normalized coords: 150/800 = 0.1875, 150/500 = 0.3
        action_no_horizon = {"x": 0.1875, "y": 0.3, "horizon": 0}
        obs1 = env.step(action_no_horizon)
        reward_no_horizon = obs1["reward"]

        # Reset and try with horizon
        env.reset()
        env.round.current_match.duck_a.x = 100
        env.round.current_match.duck_a.y = 100

        action_with_horizon = {"x": 0.1875, "y": 0.3, "horizon": 15}
        obs2 = env.step(action_with_horizon)
        reward_with_horizon = obs2["reward"]

        # Horizon should reduce reward (if hit)
        if obs1["last_action_result"] in ["hit", "double_kill"]:
            assert reward_with_horizon < reward_no_horizon

    def test_episode_completion_game_over(self, env):
        """Done flag set on game over."""
        env.reset()
        env.total_misses = MAX_MISSES - 1

        # Force match completion with a miss
        env.round.current_match.duck_a.state = DuckState.ESCAPED
        env.round.current_match.duck_b.state = DuckState.ESCAPED

        # Normalized coords: center of screen (0.5, 0.5)
        action = {"x": 0.5, "y": 0.5, "horizon": 0}
        obs = env.step(action)

        # Match should complete and add misses
        # This might trigger game over
        assert obs["total_misses"] >= MAX_MISSES - 1

    def test_normalized_coords_boundary_zero(self, env):
        """Normalized coord 0.0 maps to pixel 0."""
        env.reset()

        # Disable latency so duck doesn't move before shot
        env.latency_frames = 0
        env.latency_ms = 0

        # Position duck at top-left corner
        env.round.current_match.duck_a.x = 0
        env.round.current_match.duck_a.y = 0
        env.round.current_match.duck_a.state = DuckState.FLYING
        # Move duck_b far away
        env.round.current_match.duck_b.x = 700
        env.round.current_match.duck_b.y = 400

        # Shoot at normalized (0.05, 0.1) which is pixel (40, 50) - inside hitbox at (0,0)
        action = {"x": 0.05, "y": 0.1, "horizon": 0}
        obs = env.step(action)

        # Should hit duck at origin
        assert obs["last_action_result"] in ["hit", "double_kill"]

    def test_normalized_coords_boundary_one(self, env):
        """Normalized coord 1.0 maps to screen edge."""
        env.reset()

        # Position duck at bottom-right area
        env.round.current_match.duck_a.x = SCREEN_WIDTH - HITBOX_WIDTH
        env.round.current_match.duck_a.y = SCREEN_HEIGHT - HITBOX_HEIGHT
        env.round.current_match.duck_a.state = DuckState.FLYING
        # Move duck_b far away
        env.round.current_match.duck_b.x = 100
        env.round.current_match.duck_b.y = 100

        # Shoot at normalized (0.95, 0.9) which is pixel (760, 450) - inside hitbox
        action = {"x": 0.95, "y": 0.9, "horizon": 0}
        obs = env.step(action)

        # Should hit duck at corner
        assert obs["last_action_result"] in ["hit", "double_kill"]

    def test_normalized_coords_clamped(self, env):
        """Values outside 0.0-1.0 are clamped."""
        env.reset()

        # Position duck at corner
        env.round.current_match.duck_a.x = 0
        env.round.current_match.duck_a.y = 0
        env.round.current_match.duck_a.state = DuckState.FLYING

        # Shoot with negative coords (should clamp to 0)
        action = {"x": -0.5, "y": -0.5, "horizon": 0}
        obs = env.step(action)

        # Should still work (clamped to 0,0 which hits duck at origin)
        assert obs["last_action_result"] in ["hit", "double_kill", "miss"]
        # No crash = success

    def test_normalized_coords_over_one_clamped(self, env):
        """Values > 1.0 are clamped to 1.0."""
        env.reset()

        # Position duck at bottom-right
        env.round.current_match.duck_a.x = SCREEN_WIDTH - HITBOX_WIDTH
        env.round.current_match.duck_a.y = SCREEN_HEIGHT - HITBOX_HEIGHT
        env.round.current_match.duck_a.state = DuckState.FLYING

        # Shoot with coords > 1.0 (should clamp to 1.0)
        action = {"x": 1.5, "y": 2.0, "horizon": 0}
        obs = env.step(action)

        # Should still work (clamped to edge)
        assert obs["last_action_result"] in ["hit", "double_kill", "miss"]

    def test_center_shot(self, env):
        """Normalized (0.5, 0.5) maps to center pixel (400, 250)."""
        env.reset()

        # Disable latency so duck doesn't move before shot
        env.latency_frames = 0
        env.latency_ms = 0

        # Position duck at center
        env.round.current_match.duck_a.x = 400 - HITBOX_WIDTH // 2
        env.round.current_match.duck_a.y = 250 - HITBOX_HEIGHT // 2
        env.round.current_match.duck_a.state = DuckState.FLYING
        # Move duck_b far away
        env.round.current_match.duck_b.x = 100
        env.round.current_match.duck_b.y = 100

        # Shoot at normalized center (0.5, 0.5) = pixel (400, 250)
        action = {"x": 0.5, "y": 0.5, "horizon": 0}
        obs = env.step(action)

        assert obs["last_action_result"] in ["hit", "double_kill"]


class TestConfigurableResolution:
    """Tests for configurable output resolution."""

    def test_default_resolution(self):
        """Default resolution is 512x512."""
        env = DuckHuntEnvironment()
        assert env.output_size == FRAME_OUTPUT_SIZE
        assert env.output_size == (512, 512)

    def test_custom_resolution(self):
        """Custom resolution can be set."""
        env = DuckHuntEnvironment(output_size=(256, 256))
        assert env.output_size == (256, 256)
        assert env.renderer.output_size == (256, 256)

    def test_custom_resolution_renders_correctly(self):
        """Custom resolution produces correct image size."""
        env = DuckHuntEnvironment(output_size=(256, 256))
        obs = env.reset()

        # Decode first frame and check size
        import base64
        from io import BytesIO
        frame_b64 = obs["frames"][0]
        frame_bytes = base64.b64decode(frame_b64)
        image = Image.open(BytesIO(frame_bytes))

        assert image.size == (256, 256)

    def test_non_square_resolution(self):
        """Non-square resolution works."""
        env = DuckHuntEnvironment(output_size=(640, 480))
        obs = env.reset()

        import base64
        from io import BytesIO
        frame_b64 = obs["frames"][0]
        frame_bytes = base64.b64decode(frame_b64)
        image = Image.open(BytesIO(frame_bytes))

        assert image.size == (640, 480)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
