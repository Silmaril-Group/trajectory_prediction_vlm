"""OpenAI Tool Schemas for Duck Hunt VLM Agent."""

# Tool schema for the shoot action
SHOOT_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": (
            "Fire at predicted duck position. "
            "Analyze the frame sequence to track flying ducks, "
            "estimate their velocity, and predict where they will be "
            "after processing_latency_frames + horizon frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": (
                        "Predicted horizontal position, normalized 0.0 to 1.0. "
                        "0.0 = left edge, 1.0 = right edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": (
                        "Predicted vertical position, normalized 0.0 to 1.0. "
                        "0.0 = top edge, 1.0 = bottom edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": (
                        "Additional frames to wait before shooting (0-30). "
                        "Total prediction distance = "
                        "processing_latency_frames + horizon."
                    ),
                    "minimum": 0,
                    "maximum": 30,
                },
                "confidence": {
                    "type": "string",
                    "description": "Your confidence level in this shot.",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}

# List of all tools for the agent
DUCK_HUNT_TOOLS = [SHOOT_TOOL]

# System prompt template for the VLM agent
SYSTEM_PROMPT_TEMPLATE = """You are a Duck Hunt game AI agent. Your goal is to shoot flying ducks.

GAME RULES:
- Two ducks fly around the screen at a time
- You have 3 bullets per match
- Ducks bounce off screen edges
- Ducks fly in the upper half of the screen (y: 0.0-0.5 normalized)
- Match lasts 30 seconds

OBSERVATION:
- You receive {num_frames} consecutive frames showing duck movement
- Use frame sequence to estimate duck velocity and direction

COORDINATE SYSTEM:
- All positions use normalized coordinates (0.0 to 1.0)
- x: 0.0 = left edge, 1.0 = right edge, 0.5 = center
- y: 0.0 = top edge, 1.0 = bottom edge, 0.5 = middle
- Ducks typically fly in upper half (y: 0.0-0.5)

STRATEGY:
- Identify duck positions in the frames
- Calculate duck velocity from frame-to-frame movement
- Account for processing_latency_frames (given in game state)
- Predict future position: total frames = processing_latency_frames + horizon
- Lead your shot ahead of the duck's current position
- Use lower horizon (0-5) for slow/stationary ducks
- Use higher horizon (10-20) for fast-moving ducks

ALWAYS call the shoot tool with your best prediction."""


def get_system_prompt(num_frames: int) -> str:
    """Generate system prompt with the specified number of frames."""
    return SYSTEM_PROMPT_TEMPLATE.format(num_frames=num_frames)


# Default system prompt for backward compatibility
SYSTEM_PROMPT = get_system_prompt(4)
