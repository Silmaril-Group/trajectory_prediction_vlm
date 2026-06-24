# Horizon Min — Hardware-Agnostic Trajectory Prediction

![Duck Hunt](duckhunt.webp)

A vision-language model learns to predict where a moving target will be — and to lead its shot by exactly as much as the **hardware it runs on** demands.

Inference latency is not a constant. The same model is faster on a GPU than a CPU, faster at 1.6B than 3B, faster with a smaller batch than a larger one. Any policy that hard-codes a single lead time breaks the moment it is deployed on different hardware. Horizon Min solves trajectory prediction so that the timing is **decoupled from the hardware**: latency is fed to the model as metadata, and the model learns to compensate for whatever delay the current device imposes — from a fast accelerator to a slow CPU.

The model outputs `shoot(x, y, horizon)` where `horizon` controls how many extra frames the game advances before the shot lands. A larger horizon gives more time to predict target movement, but targets bounce randomly off walls, so longer predictions accumulate more error. The training reward penalizes large horizons on hits, pushing the model to find the **minimum prediction window** needed for each shot. Hence the name — *horizon minimization*.

Training in progress — experimenting with GRPO on latency-conditioned VLMs.

## Supported Models

| Model | Params | Tool-call format | VRAM (BF16) |
|-------|--------|------------------|-------------|
| [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) | ~3B | `<\|tool_call_start\|>[shoot(...)]<\|tool_call_end\|>` | 16GB |
| [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | 1.2B LFM + 400M SigLIP2 | `<\|tool_call_start\|>[shoot(...)]<\|tool_call_end\|>` | 8GB |

The model family is **auto-detected** from the config — one training script handles all models. See [training/README.md](training/README.md) for details.

## The Challenge

Duck Hunt is deceptively hard for an AI. Ducks move fast (7–11 px/frame at round 1, increasing each round), spawn from off-screen edges at random heights, bounce unpredictably off walls with speed jitter, and occasionally change direction mid-flight. The model must account for its own processing latency — the time between seeing frames and the shot actually landing. At 300ms latency (9 frames at 30 FPS), a duck traveling at 8 pixels/frame has moved 72 pixels by the time the bullet arrives. The model has to lead its shots.

The horizon tradeoff makes it harder still. The model can wait longer for a clearer trajectory — but every extra frame of prediction is a frame where the duck might bounce and invalidate the prediction. A model that masters this learns to adapt per-shot: short horizons for straight-flying ducks, longer ones near screen edges, and adjusted horizons across different latency conditions.

## How It Works

Two-stage training pipeline:

```
Stage 1 — SFT (Detection):
Game Frames (512x512) -> Vision Encoder -> LLM -> locate(x1, y1, x2, y2)
                                                        |
                                            Cross-entropy vs ground-truth hitbox

Stage 2 — GRPO (Shooting):
Game Frames (512x512) -> Vision Encoder -> LLM -> shoot(x, y, horizon)
                                                        |
                                        Environment simulates shot -> reward
                                                        |
                                                   GRPO update
```

**SFT** teaches the model to **detect** ducks — predicting the hitbox coordinates where the duck will be after processing latency. Ground-truth hitboxes are computed from the physics engine.

**GRPO** teaches the model to **shoot** — using the spatial understanding from SFT to learn an optimal shooting policy through reinforcement learning.

The action space is a single function call with three parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `x` | float | 0.0–1.0 | Horizontal position (0=left, 1=right) |
| `y` | float | 0.0–1.0 | Vertical position (0=top, 1=bottom) |
| `horizon` | integer | 0–30 | Extra frames to wait before firing |

The total prediction distance is `processing_latency_frames + horizon`. The model must learn to lead its shots based on estimated duck velocity and the combined latency.

### Latency-Aware Training (the hardware-agnostic part)

Different hardware imposes different inference latency, so the model is trained across multiple latency buckets and given the current latency as metadata. It never sees a fixed timing to memorize — it must learn the *function* from latency to lead distance, which is exactly what lets the same checkpoint run correctly on a fast GPU or a slow CPU:

| Model | Latency range | Based on |
|-------|--------------|----------|
| LFM2-VL-3B | 150–500ms | Measured processing times |
| LFM2.5-VL-1.6B | 100–300ms | Measured processing times |

At runtime, you measure the device's actual processing time and pass it in; the model adjusts its lead automatically. A horizon penalty (`-0.1 * horizon/30`) on successful hits encourages the model to shoot quickly when it can.

### Reward Function

| Outcome | Reward |
|---------|--------|
| Hit one duck | +1.0 |
| Double kill | +2.5 |
| Miss | -0.3 |
| Shot dead/escaped duck | -0.7 |
| No target | -0.5 |
| Invalid output | -1.0 |
| Horizon penalty | -0.1 x (horizon / 30) on hits |

Two reward signals are combined: **accuracy** (did the shot hit?) and **format** (is the output a valid tool call?). See [training/REWARD.md](training/REWARD.md) for full documentation.

**Format reward with verbosity penalty**: The format reward scores output quality on a scale of 0.0–1.0. A clean tool call with no extra text gets 1.0. If the model generates explanations or filler around a valid tool call, the reward is penalized proportionally (down to a floor of 0.3). No parseable tool call at all gets 0.0.

**Proximity bonus (target-aware)**: On misses, a distance-based bonus gives gradient signal toward **flying** ducks only — aiming near a falling/dead duck gives no bonus. Uses exponential decay (`0.3 * exp(-5.0 * dist)`).

**Hotspot penalty**: Tracks recent shot positions. If the model repeatedly aims at the same spot, hits there get scaled-down rewards — going negative at >40% concentration. Prevents the model from exploiting a single "lucky" position.

**Curriculum training**: Two-phase approach — phase 1 (steps 0–2000) clamps horizon to 0 so the model focuses on learning (x, y) aiming. Phase 2 (steps 2000+) unlocks horizon for temporal prediction optimization.

**Anti-collapse measures**: Several mechanisms prevent the model from collapsing to a single fixed prediction:
- **Entropy bonus** (`-entropy_coeff * H`) in the GRPO loss keeps the policy stochastic
- **Entropy floor** — if entropy drops below a threshold, an extra penalty kicks in as an emergency brake
- **KL penalty** (`beta * KL(policy || reference)`) anchors the model to its original behavior
- **Randomized few-shot examples** — the few-shot values in the prompt change every step so the model can't memorize them
- **New match every step** — fresh ducks with random positions/speeds each training step

### Training Setup

The unified training script auto-detects the model family and applies the right tool-call format:

| | LFM2-VL-3B | LFM2.5-VL-1.6B |
|---|-----------|-----------------|
| **LoRA targets** | q/k/v/out_proj + in_proj + w1/w2/w3 | q/k/v/out_proj + in_proj + w1/w2/w3 |
| **LoRA r/alpha** | 16/16 | 16/16 |
| **Learning rate** | 2e-5 | 1e-5 |
| **Generations per state** | 6 | 6 |
| **Tool format** | LiquidAI Pythonic | LiquidAI Pythonic |

**Input**: 2 observation frames (with frame_skip=6 for visible duck displacement) + latency metadata

**Output**: Native tool calls in the model's format, directly servable via vLLM/SGLang with the OpenAI SDK

**Deterministic replay**: Game snapshots capture duck state + RNG seeds for reproducible reward computation

**W&B visualization**: Training logs observation frames with crosshair overlays showing where each generation predicted the shot, color-coded by generation with hit/miss labels.

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd horizon_min
uv sync --extra training
```

### Train (recommended: SFT → GRPO)

```bash
cd training

# 1. Generate SFT dataset (duck detection labels)
python generate_sft_data.py --num-samples 2000 --output sft_dataset

# 2. SFT: teach model where ducks are
python train_sft.py --dataset sft_dataset --model LiquidAI/LFM2-VL-3B \
    --output outputs/sft_3b

# 3. Merge SFT into base model
python merge_sft_adapter.py --base LiquidAI/LFM2-VL-3B \
    --adapter outputs/sft_3b/final --output outputs/sft_3b_merged

# 4. GRPO: teach model to shoot
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override model.model_name=outputs/sft_3b_merged \
    --override training.max_steps=15000
```

### Train (GRPO only, any model)

```bash
# LFM2-VL-3B (2x GPU)
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000

# Qwen3-VL-8B (2x GPU)
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/qwen3_vl_8b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override model.quantization=null
```

See [training/TRAINING.md](training/TRAINING.md) for full commands, parameters, and pipeline details.

### Evaluate

```bash
# Local evaluation (loads checkpoint directly)
python evaluate.py --config configs/liquidai_config.yaml \
    --checkpoint outputs/lfm25_duckhunt_grpo_v2/best_checkpoint \
    --baselines

# API-based evaluation (served model via vLLM/SGLang)
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B
python eval_vlm.py --config configs/liquidai_eval.yaml
```

### Publish to Hugging Face Hub

```bash
./run_training.sh --config configs/liquidai_config.yaml --custom \
    --push-to-hub --hub-model-id username/duckhunt-lfm25-grpo
```

See [training/README.md](training/README.md) for full training documentation and [training/EVAL_VLM.md](training/EVAL_VLM.md) for API-based evaluation.

## Game Environment

A headless Duck Hunt implementation with no display required — pure Python API with PIL rendering.

| Parameter | Value |
|-----------|-------|
| Screen | 800 x 500 px |
| Model input | 512 x 512 px |
| FPS | 30 |
| Duck speed | 3 + round to 5 + round px/frame |
| Hitbox | 80 x 70 px (centered on 81x75 sprite) |
| Spawn | Off-screen left/right/top edge, Y range 20-70% |
| Mid-flight jitter | Disabled (straight-line flight) |
| Bounce speed jitter | Disabled (predictable bounces) |
| Frame skip | 6 (observation frames spaced apart for visible displacement) |
| Match duration | 30s |
| Ducks per match | 2 |
| Bullets per match | 3 |
| Game over | 4 misses |
| Coordinates | 0.0–1.0 normalized |

### Running the Server

```bash
uv run uvicorn duck_hunt_openenv.server.app:app --reload --port 8000
```

### Serving a Trained Model

```bash
# Serve base model or fine-tuned checkpoint via vLLM
cd training
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B
./serve_vlm.sh --checkpoint outputs/lfm25_duckhunt_grpo_v2/best

# SGLang backend
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B --backend sglang

# Stop server
./serve_vlm.sh --stop
```

### Client Usage

```python
from duck_hunt_openenv.duck_hunt_env.client import DuckHuntEnv
from duck_hunt_openenv.duck_hunt_env.models import ShootAction

env = DuckHuntEnv.from_local(host="localhost", port=8000)
obs = env.reset()

while not obs.done:
    action = ShootAction(x=400, y=250, horizon=5)
    obs = env.step(action)
    print(f"Result: {obs.last_action_result}, Reward: {obs.reward}")

env.close()
```

## Project Structure

```
horizon_min/
├── duck_hunt_openenv/          # Game environment
│   ├── duck_hunt_env/          # Python client
│   ├── server/                 # Game engine + PIL renderer
│   │   ├── game_engine.py      # Duck physics, bouncing, hit detection
│   │   ├── environment.py      # OpenEnv wrapper, frame buffer, latency sim
│   │   ├── renderer.py         # Headless PIL rendering
│   │   └── config.py           # All game parameters
│   ├── experiments/            # VLM agent experiments
│   ├── tests/                  # Unit tests
│   └── assets/                 # Sprites, background, crosshairs
├── training/                   # GRPO training pipeline
│   ├── train.py                # Unified training entry point (all models)
│   ├── run_training.sh         # Unified launch script
│   ├── evaluate.py             # Local evaluation with baselines
│   ├── eval_vlm.py             # API-based VLM evaluation (served models)
│   ├── serve_vlm.sh            # Model serving via vLLM/SGLang
│   ├── configs/                # YAML configs (Ministral, LiquidAI, eval)
│   └── src/
│       ├── formats.py          # Model-specific: tool schemas, prompts, parsers
│       ├── utils.py            # Shared: Action, system prompt, format dispatch
│       ├── config.py           # Configuration dataclasses + YAML loading
│       ├── model.py            # Model/processor loading, LoRA setup
│       ├── environment.py      # Training environment wrapper
│       ├── reward.py           # Reward computation
│       ├── dataset.py          # Dataset generation, snapshots, reward functions
│       └── trainer.py          # Custom GRPO trainer + W&B crosshair visualization
├── demo/                       # HuggingFace Spaces demo
│   ├── app.py                  # Gradio UI + episode runner
│   └── assets/                 # Game assets
└── GAME_PARAMETERS.md          # Game mechanics reference
```

## Why This Matters

This explores whether vision-language models can learn reactive, spatiotemporal reasoning from reinforcement learning — and whether that reasoning can be made **hardware-agnostic**. In the real world the same model is deployed across wildly different devices, each with its own inference latency, yet trajectory prediction must stay correct everywhere. By conditioning on measured latency instead of baking in a single timing, the model compensates for whatever delay its host hardware imposes. Active research — see [training/REWARD.md](training/REWARD.md) for the full reward system documentation.

## License

MIT
