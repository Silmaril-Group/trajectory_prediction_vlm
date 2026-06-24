# Training Guide

## Overview

Two training approaches, designed to work together:

1. **SFT (Supervised Fine-Tuning)** — teaches the model to **detect** ducks by predicting hitbox coordinates
2. **GRPO (Group Relative Policy Optimization)** — teaches the model to **shoot** ducks through reinforcement learning

**Recommended pipeline**: SFT first (learn detection), then GRPO (learn shooting).

### Why two steps?

SFT teaches: "given these frames + latency, the duck's hitbox will be at (x1, y1) → (x2, y2)"
GRPO teaches: "given this hitbox knowledge, shoot at the best point to hit the duck"

SFT uses `locate(x1, y1, x2, y2)` — deterministic, one correct answer per frame.
GRPO uses `shoot(x, y)` — any point inside the hitbox is valid, reward-driven optimization.

---

## SFT Training

SFT teaches the model **where ducks are** by predicting hitbox coordinates after latency.

### Output format

```
<|tool_call_start|>[locate(x1=0.55, y1=0.30, x2=0.65, y2=0.44)]<|tool_call_end|>
```

- `(x1, y1)` = top-left corner of the duck's hitbox after latency frames
- `(x2, y2)` = bottom-right corner
- Coordinates are normalized: 0.0 = left/top, 1.0 = right/bottom

### Step 1: Generate dataset

```bash
cd training

# Default: 2000 observations × 5 latencies = 10,000 examples
python generate_sft_data.py --num-samples 2000 --output sft_dataset

# Custom frame skip
python generate_sft_data.py --num-samples 2000 --frame-skip 3 --output sft_dataset

# Custom latencies
python generate_sft_data.py --num-samples 2000 --latencies 100,200,300 --output sft_dataset

# Larger dataset
python generate_sft_data.py --num-samples 5000 --output sft_dataset_large

# Small test run
python generate_sft_data.py --num-samples 50 --output sft_test
```

**How it works:**

For each observation state:
1. Starts a new match, advances random frames so ducks are visible on screen
2. Captures 2 observation frames (with configurable frame_skip between them)
3. Snapshots duck state (position, velocity, RNG state for deterministic bounces)
4. Picks a **random flying duck** as target
5. For **each latency value**: simulates duck forward, computes hitbox position
6. Validates: duck must be flying AND visible on screen in BOTH input frames and prediction frame
7. Hitbox must be fully on screen (not clipped by edges)

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples` | 2000 | Number of observation states (total examples = this × num latencies) |
| `--output` | sft_dataset | Output directory |
| `--frame-skip` | 6 | Game frames between the 2 observation frames |
| `--latencies` | 100,150,200,250,300 | Comma-separated latency values in ms |
| `--seed` | 42 | Random seed |

**Dataset structure:**

```
sft_dataset/
├── images/
│   ├── 000000_frame0.png    # first observation frame
│   ├── 000000_frame1.png    # second observation frame
│   ├── 000001_frame0.png
│   └── ...
└── dataset.json             # metadata with hitbox coordinates
```

**Why RNG is saved:** When a duck hits a wall during the latency simulation, `duck.update()` calls `random.randint()` to decide the bounce direction. Saving and restoring RNG state ensures the simulated trajectory matches what the game engine would actually produce. Without this, the ground-truth hitbox position could be wrong after a bounce.

### Step 2: Train

```bash
# LFM2.5-VL-1.6B
python train_sft.py \
    --dataset sft_dataset \
    --model LiquidAI/LFM2.5-VL-1.6B \
    --output outputs/sft_1.6b \
    --epochs 3

# LFM2-VL-3B
python train_sft.py \
    --dataset sft_dataset \
    --model LiquidAI/LFM2-VL-3B \
    --output outputs/sft_3b \
    --lr 2e-5

# Higher LoRA rank for more capacity
python train_sft.py \
    --dataset sft_dataset \
    --model LiquidAI/LFM2-VL-3B \
    --lora-r 32 --lora-alpha 64 \
    --output outputs/sft_3b_r32

# With W&B logging
python train_sft.py \
    --dataset sft_dataset \
    --model LiquidAI/LFM2-VL-3B \
    --wandb-project duckhunt-sft \
    --wandb-name "sft-3b-v1"
```

**How SFT loss works:**

Standard cross-entropy on completion tokens only. The prompt (system + few-shot + user with images) is masked with label=-100. Only the `locate(x1=..., y1=..., x2=..., y2=...)` tokens contribute to the loss.

Since the hitbox coordinates are **deterministic** (one correct answer per frame + latency), there are no contradictory labels. The model learns a clean mapping from visual input to spatial coordinates.

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Path to SFT dataset directory |
| `--model` | LiquidAI/LFM2.5-VL-1.6B | Base model name |
| `--output` | outputs/sft | Output directory for checkpoints |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha (scaling = alpha/r) |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--save-steps` | 200 | Checkpoint save interval |
| `--wandb-project` | duckhunt-sft | W&B project name |
| `--wandb-name` | None | W&B run name |
| `--seed` | 42 | Random seed |

### Step 3: Merge adapter (for GRPO refinement)

```bash
# Merge SFT LoRA into base model
python merge_sft_adapter.py \
    --base LiquidAI/LFM2-VL-3B \
    --adapter outputs/sft_3b/final \
    --output outputs/sft_3b_merged
```

This creates a standalone model with SFT detection knowledge baked in. GRPO then applies a fresh LoRA on top for shooting policy.

---

## GRPO Training

GRPO teaches the model to **shoot ducks** through reinforcement learning. Uses `shoot(x, y)` tool call format.

Can be run standalone (from base model) or after SFT (recommended).

### Single GPU

```bash
cd training

# LFM2.5-VL-1.6B
python train.py --config configs/liquidai_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000

# LFM2-VL-3B
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override logging.wandb_run_name="lfm2-3b-grpo"

# Qwen3-VL-8B
python train.py --config configs/qwen3_vl_8b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override model.attn_implementation=sdpa \
    --override model.quantization=null \
    --override logging.wandb_run_name="qwen3-8b-grpo"
```

### Multi-GPU (2x A100)

```bash
# LFM2-VL-3B on 2 GPUs
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override logging.wandb_run_name="lfm2-3b-2gpu"

# Qwen3-VL-8B on 2 GPUs with flash attention
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/qwen3_vl_8b_config.yaml --custom \
    --override training.max_steps=25000 \
    --override grpo.curriculum_phase2_step=15000 \
    --override model.quantization=null \
    --override grpo.temperature=1.5 \
    --override grpo.entropy_coeff=0.005 \
    --override grpo.num_generations=12 \
    --override logging.wandb_run_name="qwen3-8b-2gpu-12gen"
```

### GRPO after SFT (refinement)

```bash
# Point GRPO at the merged SFT model
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override model.model_name=outputs/sft_3b_merged \
    --override training.max_steps=15000 \
    --override grpo.curriculum_phase2_step=10000 \
    --override reward.skip_invalid_generations=true \
    --override reward.format_weight=0.0 \
    --override logging.wandb_run_name="grpo-after-sft"
```

### Resume from checkpoint

```bash
python train.py --config configs/liquidai_3b_config.yaml --custom \
    --override training.max_steps=25000 \
    --resume outputs/lfm2_3b_duckhunt_grpo_v2/checkpoint-5000
```

### Push to HuggingFace Hub

```bash
# Add to any training command
    --push-to-hub --hub-model-id dmayboroda/duckhunt-model-name
```

---

## Full Pipeline: SFT → GRPO

```bash
cd training

# 1. Generate SFT dataset (detection: locate hitboxes)
python generate_sft_data.py --num-samples 2000 --output sft_dataset

# 2. SFT training (learn where ducks are)
python train_sft.py \
    --dataset sft_dataset \
    --model LiquidAI/LFM2-VL-3B \
    --output outputs/sft_3b \
    --epochs 3 \
    --wandb-name "sft-3b"

# 3. Merge SFT adapter into base model
python merge_sft_adapter.py \
    --base LiquidAI/LFM2-VL-3B \
    --adapter outputs/sft_3b/final \
    --output outputs/sft_3b_merged

# 4. GRPO refinement (learn to shoot accurately)
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    train.py --config configs/liquidai_3b_config.yaml --custom \
    --override model.model_name=outputs/sft_3b_merged \
    --override training.max_steps=15000 \
    --override grpo.curriculum_phase2_step=10000 \
    --override reward.skip_invalid_generations=true \
    --override reward.format_weight=0.0 \
    --override logging.wandb_run_name="grpo-after-sft" \
    --push-to-hub --hub-model-id dmayboroda/duckhunt-3b-sft-grpo
```

**What each step teaches:**

| Step | Tool | What model learns |
|------|------|-------------------|
| SFT | `locate(x1, y1, x2, y2)` | Where ducks ARE after latency (detection) |
| GRPO | `shoot(x, y)` | Where to shoot to HIT ducks (policy) |

---

## CLI Overrides

Any config value can be overridden from the command line:

```bash
--override grpo.curriculum_phase2_step=10000
--override training.learning_rate=3e-5
--override training.max_steps=15000
--override grpo.num_generations=8
--override grpo.temperature=1.5
--override grpo.entropy_coeff=0.005
--override reward.proximity_decay=5.0
--override reward.hitbox_center_bonus=0.5
--override reward.edge_bonus=0.3
--override reward.skip_invalid_generations=true
--override reward.format_weight=0.0
--override reward.hotspot_enabled=true
--override model.quantization=null
--override model.attn_implementation=sdpa
--override logging.wandb_run_name="my-run"
```

---

## Available Configs

| Config | Model | Notes |
|--------|-------|-------|
| `configs/liquidai_config.yaml` | LFM2.5-VL-1.6B | Small model, fast iteration |
| `configs/liquidai_3b_config.yaml` | LFM2-VL-3B | Medium model, better vision |
| `configs/qwen3_vl_8b_config.yaml` | Qwen3-VL-8B-Instruct | Large model, best spatial understanding |
| `configs/accelerate_2gpu.yaml` | — | Accelerate config for 2x GPU |

---

## W&B Metrics

### SFT

| Metric | Description |
|--------|-------------|
| `sft/loss` | Cross-entropy loss on completion tokens (locate coordinates) |
| `sft/learning_rate` | Current learning rate |
| `sft/gradient_norm` | Gradient norm before clipping |
| `sft/epoch` | Current epoch |
| `sft/samples_processed` | Total samples seen |

### GRPO

| Metric | Description |
|--------|-------------|
| `train/loss` | GRPO policy loss |
| `train/hit_rate` | Cumulative actual duck hits / total shots |
| `train/mean_reward` | Average reward across generations |
| `train/std_reward` | Reward variance (higher = more GRPO signal) |
| `train/mean_entropy` | Token-level entropy (0.3–1.5 is healthy) |
| `train/advantages_mean` | Average advantage (should be ~0) |
| `train/advantages_std` | Advantage spread (higher = better differentiation) |
| `train/advantages_max` | Best generation's advantage |
| `train/advantages_min` | Worst generation's advantage |
| `train/learning_rate` | Current learning rate |
| `train/gradient_norm` | Gradient norm before clipping |
| `train/curriculum_phase` | 1 = no horizon, 2 = horizon unlocked |
| `train/hits` | W&B image: shot frame with green crosshair + hitbox rectangles |
| `train/misses` | W&B image: shot frame with red crosshair + hitbox rectangles |
| `train/all_shots_overlay` | W&B image: all generations on one frame |

---

## Data Validation

The SFT dataset generator enforces strict validation:

| Check | What it prevents |
|-------|-----------------|
| Duck must be flying at observation time | No dead/escaped ducks in input frames |
| Duck must be visible on screen at observation time | No off-screen spawning ducks in input |
| Duck must be flying after latency simulation | No ground-truth for escaped ducks |
| Duck must be visible on screen after latency | No off-screen predictions |
| Hitbox must be fully on screen after latency | No clipped hitbox coordinates |
| At least 1 flying duck required | No empty frames without targets |
