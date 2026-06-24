"""Duck Hunt Headless Renderer using PIL"""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from game_config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FRAME_OUTPUT_SIZE,
    HITBOX_WIDTH,
    HITBOX_HEIGHT,
)


ASSETS_DIR = Path(__file__).parent / "assets"

# Sprite sheet layout (from original duck.py)
# All measurements in pixels
FRAME_W = HITBOX_WIDTH   # 81
FRAME_H = HITBOX_HEIGHT  # 75

# Duck sprite positions in sprite sheet
# Using black/green duck (middle column in each row)
SPRITE_MAP = {
    # Flying diagonal up-right: row 3, y=225
    # 3 animation frames at x = 250, 331, 412
    "fly_up_right": {"y": 225, "frames": [250, 331, 412]},

    # Flying diagonal down-right: row 4, y=300
    "fly_down_right": {"y": 300, "frames": [250, 331, 412]},

    # Flying straight up: row 5, y=380
    "fly_up": {"y": 380, "frames": [250, 331, 412]},

    # Shot/Falling: row 6, y=460
    # Frame 0 = just shot (stunned), Frame 1 = falling
    "shot": {"y": 460, "x": 250},
    "falling": {"y": 460, "x": 331},
}


class Renderer:
    """Headless renderer using PIL for frame generation."""

    def __init__(self, output_size: tuple[int, int] = FRAME_OUTPUT_SIZE):
        self.output_size = output_size

        # Load background image
        bg_path = ASSETS_DIR / "background.jpg"
        self.background = Image.open(bg_path).convert("RGBA")
        self.background = self.background.resize((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Load sprite sheet
        sprites_path = ASSETS_DIR / "sprites.png"
        self.sprite_sheet = Image.open(sprites_path).convert("RGBA")

        # Extract duck sprites
        self.duck_sprites = self._extract_duck_sprites()

    def _extract_duck_sprites(self) -> dict[str, list[Image.Image] | Image.Image]:
        """Extract duck sprites from sprite sheet."""
        sprites = {}

        # Flying sprites (3 animation frames each direction)
        for direction in ["fly_up_right", "fly_down_right", "fly_up"]:
            info = SPRITE_MAP[direction]
            y = info["y"]
            frames_right = []
            frames_left = []

            for x in info["frames"]:
                sprite = self.sprite_sheet.crop((x, y, x + FRAME_W, y + FRAME_H))
                frames_right.append(sprite)
                frames_left.append(sprite.transpose(Image.FLIP_LEFT_RIGHT))

            sprites[direction] = frames_right
            # Create left-facing versions
            left_key = direction.replace("right", "left") if "right" in direction else direction + "_left"
            sprites[left_key] = frames_left

        # Shot sprite (just hit, stunned)
        shot_info = SPRITE_MAP["shot"]
        sprites["shot"] = self.sprite_sheet.crop((
            shot_info["x"], shot_info["y"],
            shot_info["x"] + FRAME_W, shot_info["y"] + FRAME_H
        ))

        # Falling sprite
        fall_info = SPRITE_MAP["falling"]
        sprites["falling"] = self.sprite_sheet.crop((
            fall_info["x"], fall_info["y"],
            fall_info["x"] + FRAME_W, fall_info["y"] + FRAME_H
        ))

        return sprites

    def _get_duck_sprite(self, sprite_dir: str, frame: int = 0) -> Image.Image:
        """Get appropriate sprite for duck direction and animation frame."""
        anim_frame = frame % 3

        # Falling states
        if sprite_dir == "falling":
            return self.duck_sprites["falling"]
        if sprite_dir == "shot":
            return self.duck_sprites["shot"]

        # Map sprite_dir to sprite keys
        # sprite_dir can be: up_right, down_right, up_left, down_left, right, left
        if sprite_dir in ("up_right", "right"):
            key = "fly_up_right"
        elif sprite_dir == "down_right":
            key = "fly_down_right"
        elif sprite_dir in ("up_left", "left"):
            key = "fly_up_left"
        elif sprite_dir == "down_left":
            key = "fly_down_left"
        else:
            key = "fly_up_right"  # fallback

        frames = self.duck_sprites.get(key)
        if frames and isinstance(frames, list):
            return frames[anim_frame]

        # Fallback
        return self.duck_sprites["fly_up_right"][0]

    def render_frame(self, game_state: dict, frame_counter: int = 0) -> Image.Image:
        """Render a single frame. Returns 800x500 PIL.Image."""
        # Copy background
        frame = self.background.copy()

        # Render duck_a
        duck_a = game_state.get("duck_a")
        if duck_a:
            self._render_duck(frame, duck_a, frame_counter)

        # Render duck_b
        duck_b = game_state.get("duck_b")
        if duck_b:
            self._render_duck(frame, duck_b, frame_counter)

        return frame

    def _render_duck(self, frame: Image.Image, duck: dict, frame_counter: int):
        """Render a duck onto the frame."""
        state = duck.get("state")
        if state == "escaped":
            return  # Don't render escaped ducks

        x = int(duck.get("x", 0))
        y = int(duck.get("y", 0))
        sprite_dir = duck.get("sprite_dir", "up_right")

        # Get sprite based on state
        if state == "falling":
            # Animation: first few frames show "shot", then "falling"
            if duck.get("just_shot", False):
                sprite = self._get_duck_sprite("shot")
            else:
                sprite = self._get_duck_sprite("falling")
        else:
            # Flying - animate based on frame counter (every 8 frames)
            anim_frame = (frame_counter // 8) % 3
            sprite = self._get_duck_sprite(sprite_dir, anim_frame)

        # Paste sprite with transparency
        if x >= -FRAME_W and x < SCREEN_WIDTH and y >= -FRAME_H and y < SCREEN_HEIGHT:
            frame.paste(sprite, (x, y), sprite)

    def render_and_resize(self, game_state: dict, frame_counter: int = 0) -> Image.Image:
        """Render frame and resize to output size (512x512)."""
        image = self.render_frame(game_state, frame_counter)
        resized = image.resize(self.output_size, Image.LANCZOS)
        return resized

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL.Image to base64 encoded PNG string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        return encoded
