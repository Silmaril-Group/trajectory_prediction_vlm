# VLM Evaluation for Duck Hunt

Evaluates Vision-Language Models served via vLLM or SGLang on the Duck Hunt horizon minimization task. The eval script queries the model through an OpenAI-compatible API — no local model loading.

Supports both Mistral and LiquidAI model families (and any future models added to the training pipeline).

## Quick Start

```bash
cd training

# 1. Start the model server
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B

# 2. Run evaluation
python eval_vlm.py --config configs/liquidai_eval.yaml

# Single model
python eval_vlm.py --config configs/liquidai_eval.yaml \
    --model LiquidAI/LFM2.5-VL-1.6B

# Custom API endpoint
python eval_vlm.py --config configs/liquidai_eval.yaml \
    --api-base http://localhost:8000/v1

# With Weave tracking (results in W&B console)
python eval_vlm.py --config configs/liquidai_eval.yaml --weave

# Post-training eval (train first, then serve checkpoint)
./run_training.sh --config configs/liquidai_config.yaml
./serve_vlm.sh --checkpoint outputs/lfm25_duckhunt_grpo/best
python eval_vlm.py --config configs/liquidai_eval.yaml \
    --checkpoint outputs/lfm25_duckhunt_grpo/best
```

## Architecture

```
┌─────────────────┐     OpenAI API      ┌──────────────────┐
│  eval_vlm.py    │ ──── /v1/chat ────► │  vLLM / SGLang   │
│                 │     completions      │  (Docker)        │
│  - frame sweep  │ ◄── response ─────  │                  │
│  - scenarios    │                      │  LFM2.5-VL-1.6B │
│  - judge        │                      └──────────────────┘
│  - hit rate     │
└────────┬────────┘
         │
         ▼
  Duck Hunt Env (local)
  - game state + frames
  - snapshot capture
  - deterministic shot sim
```

## What It Measures

### 1. Frame Sweep — Processing Time & Max Frames

Sends 1, 2, 3, ... N frames to the served model until it fails (context length exceeded). For each frame count, records:

- **Processing time** (wall-clock ms, including network round-trip)
- **Success** (did the server return a response?)
- **Tool call** (did it produce a valid `shoot(x, y, horizon)`?)

### 2. Hit Rate — With Actual Processing Latency

For each game scenario:

1. Capture a deterministic game snapshot (duck positions, velocities, RNG state)
2. Query the served model with game frames
3. Measure **actual processing time** → `latency_frames = ceil(processing_time_ms / 1000 * 30)`
4. Parse the model's shot prediction (`x`, `y`, `horizon`)
5. Simulate the shot: advance ducks by `latency_frames + horizon` frames, check hitbox collision
6. Compute reward using the training reward function

This gives a realistic hit rate that accounts for real inference latency.

### 3. Tool Call Validation

For each model response, checks:

- **Parseable**: Can the output be parsed into a tool call at all?
- **Valid param names**: Does it contain `x`, `y`, `horizon`?
- **Valid param values**: `0 <= x <= 1`, `0 <= y <= 1`, `0 <= horizon <= 30`?

### 4. Horizon Analysis

Across all valid responses:

- **Range**: min / max / mean / std of predicted horizon values
- **All same**: Does the model always predict the same horizon? (sign of no reasoning)

### 5. LLM-as-a-Judge

Qualitative assessment via OpenAI-compatible API. Scores 1–5 on:

| Criterion | What it measures |
|-----------|-----------------|
| `tool_format` | Did it produce a proper function/tool call? |
| `spatial_awareness` | Do x,y coordinates suggest it understood where ducks are? |
| `horizon_reasoning` | Is the horizon value reasonable given the latency? |
| `instruction_following` | Did it respond with only the tool call, no explanation? |

## CLI Reference

```
python eval_vlm.py --config CONFIG [OPTIONS]

Required:
  --config PATH          Path to eval YAML config

Optional:
  --model MODEL_ID       Evaluate only this model (default: all in config)
  --checkpoint PATH      Fine-tuned checkpoint label for results
  --api-base URL         API base URL (default: http://localhost:8000/v1)
  --output PATH          Save JSON results (default: results/vlm_eval.json)
  --weave                Enable Weave tracking
  --weave-project NAME   Weave project name (default: duckhunt-vlm-eval)
```

## Configuration

### Model Entry

```yaml
models:
  - model_id: "LiquidAI/LFM2.5-VL-1.6B"
    tool_call_format: "liquidai"       # or "mistral"
    generation:
      temperature: 0.1
      max_new_tokens: 256
```

### API Endpoint

```yaml
api:
  base_url: "http://localhost:8000/v1"
```

### Eval Settings

```yaml
eval:
  num_scenarios: 10
  max_horizon: 30
  frame_sweep_max_attempts: 32
```

### Judge Configuration

The judge runs via OpenAI-compatible API (GPT-4o, Claude, etc.):

```yaml
eval:
  judge:
    enabled: true
    max_scenarios: 5
    mode: "openai"
    api_model: "gpt-4o"
    # api_base: "https://custom-endpoint.com/v1"  # optional
    # api_key: null                                # or set OPENAI_API_KEY env var
```

## Serving Models

Use `serve_vlm.sh` to deploy models before running eval:

```bash
# Base model with vLLM
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B

# Fine-tuned checkpoint
./serve_vlm.sh --checkpoint outputs/lfm25_duckhunt_grpo/best

# SGLang backend
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B --backend sglang

# Custom GPU memory / quantization
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B --gpu-memory 0.8 --quantization awq

# Stop server
./serve_vlm.sh --stop
```

## Weave Integration

```bash
python eval_vlm.py --config configs/liquidai_eval.yaml --weave
```

### What appears in W&B

**Evaluation tab** — per-model evaluation with scorers:

| Scorer | Metrics |
|--------|---------|
| `tool_call_scorer` | `produced_tool_call`, `valid_action` |
| `accuracy_scorer` | `is_hit`, `is_double_kill`, `is_miss`, `total_hits`, `reward`, `shot_result` |
| `horizon_scorer` | `horizon_valid`, `horizon_efficiency`, `horizon_value` |
| `timing_scorer` | `processing_time_ms`, `speed_score`, `latency_frames` |
| `judge_scorer` | `judge_tool_format`, `judge_spatial_awareness`, `judge_horizon_reasoning`, `judge_instruction_following`, `judge_avg` |

**Published objects:**

- `frame_sweep/{model_id}` — table of processing time vs frame count
- `summary/{model_id}` — top-level metrics dict

### Pre vs Post Training Comparison

```bash
# Before training — evaluate base model
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B
python eval_vlm.py --config configs/liquidai_eval.yaml \
    --weave --weave-project duckhunt-liquidai

# Train (unified script — auto-detects LiquidAI format)
./run_training.sh --config configs/liquidai_config.yaml

# After training — evaluate fine-tuned checkpoint
./serve_vlm.sh --stop
./serve_vlm.sh --checkpoint outputs/lfm25_duckhunt_grpo/best
python eval_vlm.py --config configs/liquidai_eval.yaml \
    --checkpoint outputs/lfm25_duckhunt_grpo/best \
    --weave --weave-project duckhunt-liquidai
```

## Output Format

Results are saved as JSON (default: `results/vlm_eval.json`):

```json
[
  {
    "model_id": "LiquidAI/LFM2.5-VL-1.6B",
    "checkpoint": null,
    "api_base": "http://localhost:8000/v1",
    "max_frames_supported": 12,
    "tool_call_rate": 0.8,
    "valid_action_rate": 0.7,
    "hit_rate": 0.15,
    "total_shots": 10,
    "total_hits": 2,
    "double_kills": 0,
    "misses": 6,
    "average_reward": -0.12,
    "horizon_min": 0,
    "horizon_max": 25,
    "horizon_mean": 8.3,
    "horizon_std": 5.2,
    "horizon_all_same": false,
    "frame_sweep": [
      {"num_frames": 1, "processing_time_ms": 342.5, "success": true, "produced_tool_call": true}
    ],
    "scenarios": [...]
  }
]
```

## Dependencies

```bash
# Core
pip install openai pillow pyyaml

# Weave (optional)
pip install weave
```
