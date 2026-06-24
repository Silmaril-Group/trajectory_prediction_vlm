"""Model loading, prompt building, and output parsing for the Duck Hunt VLM demo.

Combines logic from:
- training/evaluate.py (model loading with 4-bit quantization)
- training/src/utils.py (prompt construction, tool schema, action parsing)
"""

from __future__ import annotations

import json
import logging
import re
import string
import random
from dataclasses import dataclass

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema (Mistral native function-calling format)
# ---------------------------------------------------------------------------
SHOOT_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": (
            "Fire at predicted duck position. "
            "Analyze the frame sequence to estimate duck velocity, "
            "then predict where the duck will be after "
            "processing_latency_frames + horizon frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": (
                        "Predicted horizontal position, normalised 0.0-1.0. "
                        "0.0 = left edge, 1.0 = right edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": (
                        "Predicted vertical position, normalised 0.0-1.0. "
                        "0.0 = top edge, 1.0 = bottom edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": (
                        "Additional frames to wait before shooting (0-30). "
                        "Total prediction = processing_latency_frames + horizon."
                    ),
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}

TOOLS = [SHOOT_TOOL]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} frames. Latency: {processing_latency_frames} frames.
Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).
Predict where the duck will be after latency + horizon frames.

IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."""


def format_system_prompt(
    *,
    num_frames: int = 4,
    processing_latency_frames: int = 6,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        num_frames=num_frames,
        processing_latency_frames=processing_latency_frames,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(
    frames: list[Image.Image],
    state: dict,
    num_frames: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build chat messages and tools list for apply_chat_template."""
    if num_frames is None:
        num_frames = len(frames)

    latency_frames = state.get("simulated_latency_frames", 6)

    system_prompt = format_system_prompt(
        num_frames=num_frames,
        processing_latency_frames=latency_frames,
    )

    user_content: list[dict] = []
    for img in frames:
        user_content.append({"type": "image", "image": img})

    state_text = (
        f"{num_frames} frames, {state.get('ducks_flying', '?')} ducks flying, "
        f"latency {latency_frames} frames. Shoot now."
    )
    user_content.append({"type": "text", "text": state_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": (
                "Frame sequence: 4 frames. Ducks flying: 2. "
                "Latency: 6 frames. Call the shoot tool now."
            )}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": '[TOOL_CALLS] [{"name": "shoot", "arguments": {"x": 0.45, "y": 0.25, "horizon": 8}, "id": "abc123xyz"}]'}],
        },
        {"role": "user", "content": user_content},
    ]

    return messages, TOOLS


# ---------------------------------------------------------------------------
# Action dataclass + parsing
# ---------------------------------------------------------------------------
@dataclass
class Action:
    x: float
    y: float
    horizon: int


def _build_action(
    raw_x: str | float,
    raw_y: str | float,
    raw_horizon: str | int,
    max_horizon: int,
) -> Action:
    x = max(0.0, min(1.0, float(raw_x)))
    y = max(0.0, min(1.0, float(raw_y)))
    horizon = max(0, min(max_horizon, int(float(raw_horizon))))
    return Action(x=x, y=y, horizon=horizon)


def parse_tool_call(
    output_text: str,
    max_horizon: int = 30,
) -> Action | None:
    """Extract an Action from the model's raw decoded output."""
    # --- attempt 1: Mistral [TOOL_CALLS] format ---
    tc_match = re.search(
        r"\[TOOL_CALLS\]\s*(\[.*\])", output_text, re.DOTALL
    )
    if tc_match:
        try:
            calls = json.loads(tc_match.group(1))
            if isinstance(calls, list) and len(calls) > 0:
                call = calls[0]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if "x" in args and "y" in args:
                    return _build_action(
                        args["x"], args["y"],
                        args.get("horizon", 0), max_horizon,
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # --- attempt 1b: Ministral [TOOL_CALLS]name[ARGS]{...} format ---
    args_match = re.search(
        r"\[TOOL_CALLS\]\s*\w+\s*\[ARGS\]\s*(\{.*?\})", output_text, re.DOTALL
    )
    if args_match:
        try:
            args = json.loads(args_match.group(1))
            if "x" in args and "y" in args:
                return _build_action(
                    args["x"], args["y"],
                    args.get("horizon", 0), max_horizon,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # --- attempt 2: find any JSON object with x/y keys ---
    try:
        start = output_text.index("{")
        depth = 0
        end = start
        for i, ch in enumerate(output_text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        blob = json.loads(output_text[start:end])
        if "arguments" in blob and isinstance(blob["arguments"], dict):
            blob = blob["arguments"]
        elif "arguments" in blob and isinstance(blob["arguments"], str):
            blob = json.loads(blob["arguments"])
        if "x" in blob and "y" in blob:
            return _build_action(
                blob["x"], blob["y"],
                blob.get("horizon", 0), max_horizon,
            )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # --- attempt 3: key=value fallback ---
    kv_match = re.search(
        r"x\s*[=:]\s*([0-9.eE+-]+).*?"
        r"y\s*[=:]\s*([0-9.eE+-]+).*?"
        r"horizon\s*[=:]\s*([0-9]+)",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    if kv_match:
        return _build_action(
            kv_match.group(1), kv_match.group(2),
            kv_match.group(3), max_horizon,
        )

    logger.warning("Failed to parse tool call from: %s", output_text[:200])
    return None


# ---------------------------------------------------------------------------
# Model loading (4-bit QLoRA for T4 16GB)
# ---------------------------------------------------------------------------
def load_model_and_processor(
    adapter_id: str = "dmayboroda/dh_ministal_gpro",
):
    """Load the LoRA adapter with 4-bit quantization.

    Returns (model, processor).
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoProcessor, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading adapter %s with 4-bit quantization...", adapter_id)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(adapter_id)
    processor.tokenizer.padding_side = "left"

    logger.info("Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Predict shot: end-to-end inference
# ---------------------------------------------------------------------------
def predict_shot(
    model,
    processor,
    frames: list[Image.Image],
    state: dict,
    max_new_tokens: int = 128,
) -> Action | None:
    """Run inference and return a parsed Action (or None on failure)."""
    messages, tools = build_prompt(frames, state)

    inputs = processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0, prompt_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=False)
    logger.info("Model output: %s", decoded[:200])

    return parse_tool_call(decoded)
