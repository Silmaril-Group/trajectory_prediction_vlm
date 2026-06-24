"""VLM Agent with OpenAI Tool Calls and Weave Tracking."""

import sys
import json
import base64
import random
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

import weave
from PIL import Image
from openai import OpenAI

from .tools import DUCK_HUNT_TOOLS, get_system_prompt
from config import FRAMES_PER_OBSERVATION_MIN, FRAMES_PER_OBSERVATION_MAX


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))


@dataclass
class ShootPrediction:
    """Result of VLM prediction."""
    x: float  # Normalized 0.0-1.0
    y: float  # Normalized 0.0-1.0
    horizon: int
    confidence: str | None = None
    raw_response: dict | None = None


class DuckHuntVLMAgent(weave.Model):
    """VLM Agent that uses OpenAI tool calls to predict shots."""

    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 256
    num_frames: int = 4  # Number of frames for observation (tracked by Weave)

    def __init__(self, **kwargs):
        # If num_frames not specified, randomly select from configured range
        if "num_frames" not in kwargs:
            kwargs["num_frames"] = random.randint(
                FRAMES_PER_OBSERVATION_MIN, FRAMES_PER_OBSERVATION_MAX
            )
        super().__init__(**kwargs)
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @weave.op
    def predict(
        self,
        frames: list[Image.Image],
        game_state: dict | None = None,
    ) -> ShootPrediction:
        """
        Predict where to shoot based on observation frames.

        Args:
            frames: List of PIL Images (observation frames)
            game_state: Optional game state dict for context

        Returns:
            ShootPrediction with x, y, horizon, confidence
        """
        # Build message content with images
        content = []

        # Add frames as images
        for i, frame in enumerate(frames):
            b64 = pil_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })

        # Add text context
        text_prompt = f"Frame sequence showing duck movement (frames 1-{len(frames)}, oldest to newest)."
        if game_state:
            text_prompt += f"""

Current game state:
- Round: {game_state.get('round_number', '?')}
- Match: {game_state.get('match_number', '?')}
- Ducks flying: {game_state.get('ducks_flying', '?')}
- Bullets remaining: {game_state.get('bullets_remaining', '?')}
- Processing latency: {game_state.get('processing_latency_ms', '?')}ms

Analyze the frames and call the shoot tool with your prediction."""
        else:
            text_prompt += "\n\nAnalyze the frames and call the shoot tool with your prediction."

        content.append({"type": "text", "text": text_prompt})

        # Call OpenAI with tool
        system_prompt = get_system_prompt(self.num_frames)
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            tools=DUCK_HUNT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "shoot"}},
        )

        # Parse tool call response
        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            return ShootPrediction(
                x=args.get("x", 0.5),
                y=args.get("y", 0.5),
                horizon=args.get("horizon", 0),
                confidence=args.get("confidence"),
                raw_response=response.model_dump(),
            )
        else:
            # Fallback if no tool call (shouldn't happen with tool_choice)
            return ShootPrediction(
                x=0.5,
                y=0.5,
                horizon=0,
                confidence="low",
                raw_response=response.model_dump(),
            )


# Standalone function version for simpler usage
@weave.op
def vlm_predict_shot(
    frames: list[Image.Image],
    game_state: dict | None = None,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    num_frames: int | None = None,
) -> dict:
    """
    Standalone function to predict shot using VLM.

    Args:
        frames: List of observation frames
        game_state: Optional game state context
        model_name: OpenAI model to use
        temperature: Sampling temperature
        num_frames: Number of frames (if None, randomly selected from config range)

    Returns dict compatible with environment step().
    """
    agent_kwargs = {
        "model_name": model_name,
        "temperature": temperature,
    }
    if num_frames is not None:
        agent_kwargs["num_frames"] = num_frames

    agent = DuckHuntVLMAgent(**agent_kwargs)
    prediction = agent.predict(frames, game_state)

    return {
        "x": prediction.x,
        "y": prediction.y,
        "horizon": prediction.horizon,
        "confidence": prediction.confidence,
        "num_frames": agent.num_frames,  # Include in output for tracking
    }
