"""Prompt building, tool schema, and output parsing for Duck Hunt GRPO training.

Shared code lives here: ``Action``, ``_build_action``, system prompt template.

Format-specific logic (Mistral vs LiquidAI tool-call tokens, few-shot
examples, primary parser regexes) lives in ``formats.py``.  Call
``set_model_format(model_name)`` to activate the right format; after that,
``build_prompt``, ``parse_tool_call``, and ``TOOLS`` all delegate to it.
"""

from __future__ import annotations

import logging
import string
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .formats import ModelFormat

logger = logging.getLogger(__name__)


# ===================================================================
#  Shared: system prompt
# ===================================================================
SYSTEM_PROMPT_TEMPLATE = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} frames. Latency: {processing_latency_frames} frames.
Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).
Predict where the duck will be after latency + horizon frames.

IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."""

SYSTEM_PROMPT_PHASE1_TEMPLATE = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} frames. Latency: {processing_latency_frames} frames.
Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).
Predict where the duck will be after latency frames.

IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."""


def format_system_prompt(
    *,
    num_frames: int = 4,
    processing_latency_frames: int = 6,
    phase: int = 2,
) -> str:
    """Return the system prompt with placeholders filled in."""
    template = SYSTEM_PROMPT_PHASE1_TEMPLATE if phase == 1 else SYSTEM_PROMPT_TEMPLATE
    return template.format(
        num_frames=num_frames,
        processing_latency_frames=processing_latency_frames,
    )


# ===================================================================
#  Shared: Action dataclass + builder
# ===================================================================
@dataclass
class Action:
    """Parsed shoot action from the model's completion."""

    x: float
    y: float
    horizon: int


def _build_action(
    raw_x: str | float,
    raw_y: str | float,
    raw_horizon: str | int,
    max_horizon: int,
) -> Action:
    """Clamp values and return an ``Action``."""
    x = max(0.0, min(1.0, float(raw_x)))
    y = max(0.0, min(1.0, float(raw_y)))
    horizon = max(0, min(max_horizon, int(float(raw_horizon))))
    return Action(x=x, y=y, horizon=horizon)


# ===================================================================
#  Shared: Mistral call-ID helper
# ===================================================================
_CALL_ID_CHARS = string.ascii_letters + string.digits


def generate_call_id() -> str:
    """Return a random 9-character alphanumeric ID (Mistral requirement)."""
    return "".join(random.choices(_CALL_ID_CHARS, k=9))


# ===================================================================
#  Active format (set via set_model_format)
# ===================================================================
_active_format: ModelFormat | None = None


def set_model_format(model_name: str) -> None:
    """Activate the tool-call format for *model_name*.

    After calling this, ``build_prompt``, ``parse_tool_call``, and ``TOOLS``
    all use the detected format (Mistral or LiquidAI).
    """
    global _active_format, TOOLS
    from .formats import get_format

    _active_format = get_format(model_name)
    TOOLS = _active_format.get_tools()
    logger.info("Active format set to %s", type(_active_format).__name__)


def _ensure_format() -> ModelFormat:
    """Return the active format, defaulting to Mistral for backward compat."""
    global _active_format, TOOLS
    if _active_format is None:
        from .formats import MistralFormat
        _active_format = MistralFormat()
        TOOLS = _active_format.get_tools()
        logger.info("No format set — defaulting to MistralFormat")
    return _active_format


# ===================================================================
#  Delegating functions (same signatures as before)
# ===================================================================
# Initial TOOLS value (Mistral default for backward compat).
# Overwritten by set_model_format().
TOOLS: list[dict] = []


def build_prompt(
    frames: list[Image.Image],
    state: dict,
    num_frames: int | None = None,
    phase: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Build chat messages and tools list — delegates to the active format."""
    fmt = _ensure_format()
    return fmt.build_prompt(frames, state, num_frames, phase=phase)


def parse_tool_call(
    output_text: str,
    max_horizon: int = 30,
    phase: int = 2,
) -> Action | None:
    """Parse model output into an Action — delegates to the active format."""
    fmt = _ensure_format()
    return fmt.parse_tool_call(output_text, max_horizon, phase=phase)
