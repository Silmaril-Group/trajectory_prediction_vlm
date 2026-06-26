"""Unit tests for model-family tool-call formats (src/formats.py).

Covers tool-call parsing for every registered format (Mistral, LiquidAI, Qwen,
Gemma, InternVL), phase-1 horizon clamping, fallbacks, and model-name
auto-detection via the registry.

Run with:  pytest training/tests/test_formats.py
       or:  python training/tests/test_formats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.formats import (  # noqa: E402
    GemmaFormat,
    InternVLFormat,
    LiquidAIFormat,
    MistralFormat,
    QwenFormat,
    get_format,
    get_format_by_name,
)


# ---------------------------------------------------------------------------
# Registry / auto-detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("mistralai/Ministral-3-8B-Instruct-2512-BF16", MistralFormat),
        ("LiquidAI/LFM2.5-VL-1.6B", LiquidAIFormat),
        ("LFM2-VL-3B", LiquidAIFormat),
        ("Qwen/Qwen3-VL-8B-Instruct", QwenFormat),
        ("google/gemma-3-4b-it", GemmaFormat),
        ("OpenGVLab/InternVL3-2B-hf", InternVLFormat),
        ("OpenGVLab/InternVL2_5-8B", InternVLFormat),
    ],
)
def test_auto_detect(model_name, expected):
    assert isinstance(get_format(model_name), expected)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_format("some-unknown/model")


@pytest.mark.parametrize("key", ["mistral", "liquidai", "qwen", "gemma", "internvl"])
def test_get_format_by_name(key):
    assert get_format_by_name(key) is not None


# ---------------------------------------------------------------------------
# Gemma — ```tool_code``` fenced pythonic call
# ---------------------------------------------------------------------------
def test_gemma_tool_code_block():
    fmt = GemmaFormat()
    out = "```tool_code\nshoot(x=0.42, y=0.31, horizon=7)\n```"
    a = fmt.parse_tool_call(out)
    assert (round(a.x, 2), round(a.y, 2), a.horizon) == (0.42, 0.31, 7)


def test_gemma_plain_pythonic_fallback():
    fmt = GemmaFormat()
    a = fmt.parse_tool_call("I think shoot(0.5, 0.2, 9) is best")
    assert (a.x, a.y, a.horizon) == (0.5, 0.2, 9)


def test_gemma_phase1_clamps_horizon():
    fmt = GemmaFormat()
    a = fmt.parse_tool_call("```tool_code\nshoot(x=0.5, y=0.5, horizon=20)\n```", phase=1)
    assert a.horizon == 0


def test_gemma_garbage_returns_none():
    assert GemmaFormat().parse_tool_call("no tool call here") is None


# ---------------------------------------------------------------------------
# InternVL — Qwen backbone (<tool_call>) and InternLM backbone (action tokens)
# ---------------------------------------------------------------------------
def test_internvl_qwen_backbone():
    fmt = InternVLFormat()
    out = '<tool_call>{"name": "shoot", "arguments": {"x": 0.6, "y": 0.4, "horizon": 12}}</tool_call>'
    a = fmt.parse_tool_call(out)
    assert (a.x, a.y, a.horizon) == (0.6, 0.4, 12)


def test_internvl_internlm_backbone():
    fmt = InternVLFormat()
    out = (
        '<|action_start|><|plugin|>{"name": "shoot", '
        '"parameters": {"x": 0.7, "y": 0.5, "horizon": 3}}<|action_end|>'
    )
    a = fmt.parse_tool_call(out)
    assert (a.x, a.y, a.horizon) == (0.7, 0.5, 3)


def test_internvl_internlm_phase1_clamps_horizon():
    fmt = InternVLFormat()
    out = (
        '<|action_start|><|plugin|>{"name": "shoot", '
        '"parameters": {"x": 0.7, "y": 0.5, "horizon": 30}}<|action_end|>'
    )
    assert fmt.parse_tool_call(out, phase=1).horizon == 0


def test_internvl_horizon_clamped_to_max():
    fmt = InternVLFormat()
    out = '<tool_call>{"name": "shoot", "arguments": {"x": 0.6, "y": 0.4, "horizon": 999}}</tool_call>'
    assert fmt.parse_tool_call(out, max_horizon=30).horizon == 30


# ---------------------------------------------------------------------------
# Regression — existing formats still parse after the shared-helper refactor
# ---------------------------------------------------------------------------
def test_liquidai_kwargs_and_positional():
    fmt = LiquidAIFormat()
    a = fmt.parse_tool_call("<|tool_call_start|>[shoot(x=0.3, y=0.2, horizon=5)]<|tool_call_end|>")
    assert (a.x, a.y, a.horizon) == (0.3, 0.2, 5)
    b = fmt.parse_tool_call("shoot(0.3, 0.2, 5)")
    assert (b.x, b.y, b.horizon) == (0.3, 0.2, 5)


def test_liquidai_phase1():
    a = LiquidAIFormat().parse_tool_call("shoot(x=0.3, y=0.2, horizon=5)", phase=1)
    assert a.horizon == 0


def test_mistral_tool_calls():
    out = '[TOOL_CALLS] [{"name": "shoot", "arguments": {"x": 0.4, "y": 0.5, "horizon": 3}}]'
    a = MistralFormat().parse_tool_call(out)
    assert (a.x, a.y, a.horizon) == (0.4, 0.5, 3)


def test_qwen_tool_call_tag():
    out = '<tool_call>{"name": "shoot", "arguments": {"x": 0.1, "y": 0.9, "horizon": 2}}</tool_call>'
    a = QwenFormat().parse_tool_call(out)
    assert (a.x, a.y, a.horizon) == (0.1, 0.9, 2)


# ---------------------------------------------------------------------------
# Coordinate clamping is shared across all formats
# ---------------------------------------------------------------------------
def test_coords_clamped_to_unit_range():
    a = GemmaFormat().parse_tool_call("```tool_code\nshoot(x=1.5, y=-0.2, horizon=3)\n```")
    assert a.x == 1.0 and a.y == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
