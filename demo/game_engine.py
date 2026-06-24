"""Duck Hunt Game Engine"""

import random
from enum import Enum

from game_config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    HITBOX_WIDTH,
    HITBOX_HEIGHT,
    SPEED_BASE,
    SPEED_VARIANCE,
    BOUNCE_DY_MIN,
    BOUNCE_DY_MAX,
    BOUNCE_TOP_DY_MIN,
    BOUNCE_TOP_DY_MAX,
    BOUNCE_BOTTOM_DY_MIN,
    BOUNCE_BOTTOM_DY_MAX,
    BULLETS_PER_MATCH,
    MATCH_DURATION_FRAMES,
    MATCHES_PER_ROUND,
    DUCKS_PER_MATCH,
)


class DuckState(Enum):
    FLYING = "flying"
    FALLING = "falling"
    ESCAPED = "escaped"


class Duck:
    """A duck that flies around and can be shot."""

    def __init__(self, round_number: int):
        # Random spawn position (left or right edge, bottom half)
        spawn_left = random.choice([True, False])
        if spawn_left:
            self.x = 0
        else:
            self.x = SCREEN_WIDTH

        self.y = random.randint(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - HITBOX_HEIGHT)

        # Speed based on round number
        speed_range = range(SPEED_BASE + round_number, SPEED_BASE + SPEED_VARIANCE + round_number)
        speed = random.choice(list(speed_range))

        # Initial direction (move toward center)
        if spawn_left:
            self.dx = speed
        else:
            self.dx = -speed

        self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
        # Ensure dy is not 0
        if self.dy == 0:
            self.dy = random.choice([-1, 1])

        self.state = DuckState.FLYING
        self._update_sprite_dir()

    def _update_sprite_dir(self):
        """Update sprite direction based on velocity."""
        if self.dx > 0:
            if self.dy < 0:
                self.sprite_dir = "up_right"
            elif self.dy > 0:
                self.sprite_dir = "down_right"
            else:
                self.sprite_dir = "right"
        else:
            if self.dy < 0:
                self.sprite_dir = "up_left"
            elif self.dy > 0:
                self.sprite_dir = "down_left"
            else:
                self.sprite_dir = "left"

    def update(self, round_number: int):
        """Update duck position and state."""
        if self.state == DuckState.FLYING:
            self.x += self.dx
            self.y += self.dy
            self._check_boundaries(round_number)
            self._check_escaped()

        elif self.state == DuckState.FALLING:
            self.y += 4  # Fall straight down

    def _check_boundaries(self, round_number: int):
        """Check boundaries and apply bounce logic."""
        speed_range = range(SPEED_BASE + round_number, SPEED_BASE + SPEED_VARIANCE + round_number)
        speed = random.choice(list(speed_range))
        coin_toss = random.choice([-1, 1])

        # Left edge
        if self.x <= 0:
            self.dx = speed
            self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
            if self.dy == 0:
                self.dy = random.choice([-1, 1])
            self._update_sprite_dir()

        # Right edge
        elif self.x >= SCREEN_WIDTH - HITBOX_WIDTH:
            self.dx = -speed
            self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
            if self.dy == 0:
                self.dy = random.choice([-1, 1])
            self._update_sprite_dir()

        # Top edge
        elif self.y <= 0:
            self.dx = speed * coin_toss
            self.dy = random.randint(BOUNCE_TOP_DY_MIN, BOUNCE_TOP_DY_MAX)
            self._update_sprite_dir()

        # Bottom half
        elif self.y >= SCREEN_HEIGHT // 2:
            self.dx = speed * coin_toss
            self.dy = random.randint(BOUNCE_BOTTOM_DY_MIN, BOUNCE_BOTTOM_DY_MAX)
            self._update_sprite_dir()

    def _check_escaped(self):
        """Check if duck escaped off top of screen."""
        if self.y + HITBOX_HEIGHT < 0:
            self.state = DuckState.ESCAPED

    def check_hit(self, target_x: int, target_y: int) -> bool:
        """Check if target coordinates hit this duck."""
        if self.state != DuckState.FLYING:
            return False

        # Hitbox check (hitbox is at duck position with size HITBOX_WIDTH x HITBOX_HEIGHT)
        if target_x < self.x or target_x > self.x + HITBOX_WIDTH:
            return False
        if target_y < self.y or target_y > self.y + HITBOX_HEIGHT:
            return False

        return True

    def hit(self):
        """Mark duck as hit and start falling."""
        self.state = DuckState.FALLING
        self.dx = 0
        self.dy = 4
        self.sprite_dir = "falling"

    @property
    def is_finished(self) -> bool:
        """Check if duck is done (fell off screen or escaped)."""
        if self.state == DuckState.ESCAPED:
            return True
        if self.state == DuckState.FALLING and self.y >= SCREEN_HEIGHT // 2:
            return True
        return False


class Match:
    """A single match with two ducks."""

    def __init__(self, round_number: int):
        self.round_number = round_number
        self.duck_a = Duck(round_number)
        self.duck_b = Duck(round_number)
        self.bullets_remaining = BULLETS_PER_MATCH
        self.frames_elapsed = 0
        self.ducks_hit = 0

    def advance_frames(self, n: int):
        """Advance the match by n frames."""
        for _ in range(n):
            if self.is_complete:
                break
            self.duck_a.update(self.round_number)
            self.duck_b.update(self.round_number)
            self.frames_elapsed += 1

    def process_shot(self, x: int, y: int) -> tuple[bool, bool]:
        """Process a shot at (x, y). Returns (hit_a, hit_b)."""
        if self.bullets_remaining <= 0:
            return (False, False)

        self.bullets_remaining -= 1

        hit_a = self.duck_a.check_hit(x, y)
        hit_b = self.duck_b.check_hit(x, y)

        if hit_a:
            self.duck_a.hit()
            self.ducks_hit += 1

        if hit_b:
            self.duck_b.hit()
            self.ducks_hit += 1

        return (hit_a, hit_b)

    @property
    def is_complete(self) -> bool:
        """Check if match is complete."""
        both_resolved = (
            self.duck_a.state != DuckState.FLYING
            and self.duck_b.state != DuckState.FLYING
        )
        time_up = self.frames_elapsed >= MATCH_DURATION_FRAMES
        return both_resolved or time_up

    def get_flying_count(self) -> int:
        """Count ducks that are still flying."""
        count = 0
        if self.duck_a.state == DuckState.FLYING:
            count += 1
        if self.duck_b.state == DuckState.FLYING:
            count += 1
        return count

    def get_state(self) -> dict:
        """Return current match state for observation."""
        return {
            "duck_a": {
                "x": self.duck_a.x,
                "y": self.duck_a.y,
                "state": self.duck_a.state.value,
                "sprite_dir": self.duck_a.sprite_dir,
            },
            "duck_b": {
                "x": self.duck_b.x,
                "y": self.duck_b.y,
                "state": self.duck_b.state.value,
                "sprite_dir": self.duck_b.sprite_dir,
            },
            "bullets_remaining": self.bullets_remaining,
            "frames_elapsed": self.frames_elapsed,
            "ducks_hit": self.ducks_hit,
            "flying_count": self.get_flying_count(),
        }


class Round:
    """A round consisting of multiple matches."""

    def __init__(self, round_number: int):
        self.round_number = round_number
        self.current_match = Match(round_number)
        self.matches_completed = 0
        self.total_ducks_hit = 0

    def advance_to_next_match(self):
        """Complete current match and start next one."""
        self.total_ducks_hit += self.current_match.ducks_hit
        self.matches_completed += 1

        if self.matches_completed < MATCHES_PER_ROUND:
            self.current_match = Match(self.round_number)

    @property
    def is_complete(self) -> bool:
        """Check if round is complete."""
        return self.matches_completed >= MATCHES_PER_ROUND

    def get_misses(self) -> int:
        """Return total missed ducks so far."""
        return (self.matches_completed * DUCKS_PER_MATCH) - self.total_ducks_hit
