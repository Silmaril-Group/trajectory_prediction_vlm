# Horizon Minimization: Teaching a Vision-Language Model to Play Duck Hunt

## Overview

This project trains a Vision-Language Model (Ministral-3B-8B with Pixtral vision encoder) to play Duck Hunt from raw pixels using reinforcement learning (GRPO — Group Relative Policy Optimization). The core innovation is **horizon minimization**: the model outputs `shoot(x, y, horizon)` where `horizon` controls how many additional frames to wait before firing. The model must learn to balance the tradeoff between waiting for more information (better trajectory estimation) versus shooting sooner (less prediction error from bounces). The trained model achieves **60.9% hit rate** (up from ~0% untrained, ~5% random baseline).

---

## Core Concept: Horizon Minimization

### The Prediction-Action Tradeoff

In real-time aiming tasks, an agent faces a fundamental tension:

- **Waiting longer** gives more frames to observe the target trajectory, improving velocity estimation
- **Shooting sooner** means predicting fewer frames into the future, reducing error from trajectory changes (bounces, direction reversals)

The `horizon` parameter (0–30 frames) makes this tradeoff **explicit and learnable**. The total prediction window is:

```
total_prediction = processing_latency_frames + horizon
```

Where `processing_latency` simulates real-world inference delay (100–600ms, randomly sampled per episode).

### Why This Matters

Traditional approaches either:
1. Act immediately (no planning horizon) — works poorly with latency
2. Use a fixed prediction window — can't adapt to varying conditions

Horizon minimization lets the model **learn** the optimal prediction window for each situation. The reward function penalizes large horizons on successful hits, encouraging the model to be as decisive as possible while still hitting the target.

---

## Architecture

### Model Pipeline

```
Game Frames (4× 512×512 sequential)
        ↓
[Pixtral Vision Encoder (0.4B params)]    ← Mistral's multi-modal vision component
        ↓
[Ministral LLM (8.4B params)]             ← Language model backbone
        ↓
Tool Call: shoot(x, y, horizon)            ← Native Mistral function calling format
        ↓
[Environment Simulator]                     ← Deterministic replay from snapshot
        ↓
Reward Signal → GRPO Policy Update
```

### Action Space

| Parameter | Type    | Range    | Description |
|-----------|---------|----------|-------------|
| `x`       | float   | 0.0–1.0  | Normalized horizontal position |
| `y`       | float   | 0.0–1.0  | Normalized vertical position |
| `horizon` | integer | 0–30     | Additional frames to wait before shooting |

### Observation Space

The model receives per step:
- **4 sequential game frames** (512×512 PNG) — enables velocity estimation from frame-to-frame displacement
- **Metadata**: processing latency (in frames), round number, bullets remaining, ducks flying count

### Fine-tuning Setup

- **Method**: LoRA (Low-Rank Adaptation) — only ~0.2% of parameters are trainable
- **LoRA config**: rank=16, alpha=32, target modules: q_proj, k_proj, v_proj, o_proj, dropout=0.05
- **Precision**: BF16
- **Output format**: Mistral native tool calls — `[TOOL_CALLS] [{"name": "shoot", "arguments": {...}}]`

---

## Training: GRPO (Group Relative Policy Optimization)

### Algorithm

GRPO is a simplified alternative to PPO designed for language model RL. For each training step:

1. **Collect batch**: Find a game state with flying ducks, capture a deterministic snapshot
2. **Generate G=4 completions** per prompt (temperature=1.0, top-p=0.95)
3. **Compute rewards** for each completion via deterministic environment replay
4. **Normalize advantages** within the group: `(rewards - mean) / std`
5. **Compute loss**: Clipped surrogate objective (ε=0.2) + entropy bonus (prevents mode collapse)
6. **Update** via AdamW with cosine LR schedule, gradient accumulation over 8 steps

### Two Training Modes

1. **TRL mode** (default): Uses HuggingFace TRL's `GRPOTrainer` with offline dataset generation
2. **Custom mode**: Manual GRPO loop with online environment interaction, full checkpointing, W&B logging

### Key Hyperparameters

```yaml
training:
  learning_rate: 5.0e-6
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8       # Effective batch = 8

grpo:
  num_generations: 4                    # Completions per prompt
  temperature: 1.0                      # Exploration
  epsilon: 0.2                          # Clip range
  entropy_coeff: 0.01                   # Prevents mode collapse
```

---

## Reward Function

```python
def compute_reward(result, action, config):
    if action is None:          return -1.0    # Invalid output (unparseable)
    if not had_target:          return -0.5    # No ducks flying

    if hit_both_ducks:          base = 2.5     # Double kill
    elif hit_one_duck:          base = 1.0     # Single hit
    else:                       base = -0.3    # Miss

    # Horizon penalty (hits only): encourages shooting quickly
    penalty = 0.1 * (horizon / 30)

    # Proximity bonus (misses only): gradient signal for learning to aim
    proximity = 0.5 * exp(-5.0 * min_distance_to_duck)

    return base - penalty + proximity
```

### Design Rationale

- **Horizon penalty** on hits: Encourages the model to minimize prediction windows — shoot as soon as confident
- **Proximity bonus** on misses: Provides continuous gradient signal even when missing, helping the model learn rough aiming before fine-tuning accuracy
- **Distance-based reward shaping**: `exp(-5.0 * d)` gives strong signal when close to target, vanishes when far

---

## Latency-Aware Training

The model is trained across **6 latency buckets**, randomly sampled per episode:

| Latency | Frames (at 30 FPS) |
|---------|---------------------|
| 100ms   | 3 frames            |
| 200ms   | 6 frames            |
| 300ms   | 9 frames            |
| 400ms   | 12 frames           |
| 500ms   | 15 frames           |
| 600ms   | 18 frames           |

The model sees the latency value in its prompt and must adapt: higher latency requires more aggressive prediction (larger effective horizon). This forces generalization rather than memorization of a single timing.

---

## Deterministic Replay (Snapshot System)

A critical infrastructure component enabling GRPO training. For each training sample:

1. **Capture snapshot**: Duck positions, velocities, sprite directions, RNG state
2. **Restore snapshot**: Reconstruct exact game state deterministically
3. **Simulate shot**: Advance ducks by `latency_frames + horizon`, check hitbox collisions
4. **Return result**: Hit/miss + duck positions for proximity calculation

This ensures **reproducible rewards** regardless of when or how many times the reward function is called — essential for GRPO's multiple completions per prompt.

---

## Game Environment: Duck Hunt

### Game Parameters

| Parameter | Value |
|-----------|-------|
| Screen | 800×500 px (resized to 512×512 for model) |
| FPS | 30 |
| Match duration | 30 seconds (900 frames) |
| Ducks per match | 2 (spawn from edges) |
| Bullets per match | 3 |
| Game over | 4 misses total |
| Duck hitbox | 81×75 px |

### Duck Behavior

- **Spawning**: Left or right edge (50/50), random Y in bottom half, speed scales with round number
- **Movement**: Linear with bounce physics — reverse direction on edge contact with randomized angles
- **States**: FLYING → FALLING (hit) or ESCAPED (left screen top)

### Environment Architecture

- **`DuckHuntEnvironment`** (server-side): Headless game engine, returns base64 PNG frames, supports deterministic snapshots
- **`DuckHuntEnvWrapper`** (training wrapper): Direct Python import (no HTTP), converts frames to PIL Images, tracks episode statistics
- **FastAPI server** (`app.py`): Optional HTTP API for external agents

---

## Project Structure

```
horizon_min/
├── README.md                              # Public-facing project description
├── GAME_PARAMETERS.md                     # Game mechanics reference
├── pyproject.toml                         # Dependencies & metadata
│
├── training/                              # GRPO training pipeline
│   ├── train.py                           # Entry point (TRL + custom modes)
│   ├── evaluate.py                        # Evaluation with baselines
│   ├── run_training.sh                    # Launch script
│   ├── configs/
│   │   ├── base_config.yaml               # Default hyperparameters
│   │   └── ministral_config.yaml          # Ministral-specific overrides
│   └── src/
│       ├── config.py                      # Configuration dataclasses + YAML loading
│       ├── model.py                       # Model loading, LoRA setup, inference
│       ├── environment.py                 # DuckHuntEnvWrapper (direct import)
│       ├── utils.py                       # Tool schema, prompt builder, output parser
│       ├── reward.py                      # Reward computation with proximity bonus
│       ├── dataset.py                     # Dataset generation, snapshot system, TRL reward fns
│       └── trainer.py                     # Custom GRPO trainer with checkpointing + W&B
│
├── duck_hunt_openenv/                     # Game environment package
│   ├── server/                            # Game engine + FastAPI server
│   │   ├── game_engine.py                 # Duck/Match/Round classes
│   │   ├── environment.py                 # DuckHuntEnvironment (headless)
│   │   ├── renderer.py                    # PIL rendering to base64
│   │   ├── config.py                      # Game parameters
│   │   └── app.py                         # FastAPI endpoints
│   ├── duck_hunt_env/                     # Python client
│   │   ├── client.py
│   │   └── models.py
│   ├── experiments/                       # VLM agent experiments
│   │   ├── agent.py, evaluation.py, episode.py
│   │   ├── scorers.py, tools.py, run.py
│   └── assets/                            # Sprites, background, font
│
├── demo/                                  # HuggingFace Spaces Gradio demo
│   ├── app.py                             # Gradio UI + episode runner
│   ├── inference.py                       # Model inference wrapper
│   ├── environment.py, game_engine.py     # Local environment copies
│   ├── renderer.py, game_config.py
│   └── assets/
│
└── duckhunt/                              # Legacy game implementation (pygame)
```

---

## Key Results

| Agent | Hit Rate |
|-------|----------|
| Random baseline | ~5% |
| Fixed center shot (horizon=10) | ~8% |
| **GRPO-trained model** | **60.9%** |

### What the Model Learned

1. **Spatial aiming**: Predict where ducks will be, not where they are now
2. **Velocity estimation**: Use 4-frame sequences to estimate duck speed and direction
3. **Latency compensation**: Adjust prediction horizon based on processing latency
4. **Horizon minimization**: Prefer shorter prediction windows when confident, reducing bounce risk

---

## Technical Stack

- **Model**: Ministral-3B-8B-Instruct-2512-BF16 (Mistral AI)
- **Training**: TRL (HuggingFace), PEFT/LoRA, PyTorch 2.4+, Accelerate
- **Logging**: Weights & Biases
- **Environment**: Custom Duck Hunt (PIL rendering, headless)
- **Demo**: Gradio on HuggingFace Spaces
- **Config**: Layered YAML with CLI overrides

---

## Prompt Format

The model receives a system message explaining the task, followed by a user message with:

```
[4 game frame images — oldest to newest]

Game State:
- Frames: 4
- Ducks flying: 2
- Processing latency: 6 frames
- Round: 1, Match: 1
- Bullets remaining: 3

Predict where to shoot using the shoot tool.
```

The model responds with a native Mistral tool call:

```
[TOOL_CALLS] [{"name": "shoot", "arguments": {"x": 0.45, "y": 0.25, "horizon": 8}, "id": "abc123"}]
```

### Output Parsing

The parser handles 4 fallback formats to maximize reward signal during training:
1. Mistral native `[TOOL_CALLS] [...]`
2. Alternate `[TOOL_CALLS] name [ARGS] {...}`
3. Plain JSON `{"x": ..., "y": ..., "horizon": ...}`
4. Key=value `x=0.3, y=0.2, horizon=5`

---

## Running the Project

### Training

```bash
# TRL mode (default — offline dataset + GRPOTrainer)
python training/train.py --config training/configs/ministral_config.yaml

# Custom GRPO loop (online interaction)
python training/train.py --config training/configs/ministral_config.yaml --custom

# Resume from checkpoint
python training/train.py --config training/configs/ministral_config.yaml --custom \
    --resume outputs/ministral_duckhunt_grpo/checkpoint-500

# With overrides
python training/train.py --config training/configs/ministral_config.yaml \
    --override training.learning_rate=2e-5 \
    --override grpo.num_generations=8
```

### Evaluation

```bash
python training/evaluate.py --model outputs/ministral_duckhunt_grpo/best \
    --episodes 100 --latencies 100 200 300 400 500 600
```

### Dependencies

```bash
pip install -e ".[training]"     # Training dependencies
pip install -e ".[experiments]"  # Experiment dependencies
pip install -e ".[dev]"          # Development tools
```
