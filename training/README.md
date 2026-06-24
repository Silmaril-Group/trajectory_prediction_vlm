# Duck Hunt GRPO Training

Train Vision-Language Models to play Duck Hunt using **Group Relative Policy Optimization (GRPO)** with the [Duck Hunt OpenEnv](../duck_hunt_openenv/).

Supported model families:

| Model | Params | Tool-call format | VRAM (BF16) | VRAM (4-bit) |
|-------|--------|------------------|-------------|--------------|
| [Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | 8.4B LLM + 0.4B Pixtral vision | `[TOOL_CALLS] [{"name":"shoot",...}]` | 24GB | 12GB |
| [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | 1.2B LFM + 400M SigLIP2 vision | `<\|tool_call_start\|>[shoot(...)]<\|tool_call_end\|>` | 8GB | 4GB |

The model family is **auto-detected** from the config — one script handles all models.

## Quick Start

```bash
cd training

# Mistral with LoRA (default)
./run_training.sh --config configs/ministral_config.yaml

# LiquidAI with LoRA
./run_training.sh --config configs/liquidai_config.yaml

# LiquidAI full fine-tune (no LoRA)
./run_training.sh --config configs/liquidai_nolora_config.yaml

# Without W&B logging
./run_training.sh --config configs/liquidai_config.yaml --no-wandb

# Custom GRPO loop (any model)
./run_training.sh --config configs/liquidai_config.yaml --custom
```

## Requirements

- Python 3.10+
- CUDA GPU (see VRAM table above)
- [uv](https://docs.astral.sh/uv/) for dependency management
- The Duck Hunt OpenEnv game engine (included in this repo)

```bash
# From the project root
uv sync --extra training
```

Key dependencies: `torch>=2.4`, `transformers>=5.1`, `trl>=0.15`, `peft>=0.13`, `wandb>=0.18`, `huggingface_hub>=0.25`.

## Project Structure

```
training/
├── run_training.sh              # Unified launch script (all models)
├── train.py                     # Unified training entry point (TRL + custom modes)
├── evaluate.py                  # Evaluation with baselines
├── eval_vlm.py                  # API-based VLM evaluation (served models)
├── serve_vlm.sh                 # Model serving via vLLM/SGLang
├── configs/
│   ├── base_config.yaml         # Shared defaults (reward, environment, etc.)
│   ├── ministral_config.yaml    # Mistral-specific overrides
│   ├── liquidai_config.yaml     # LiquidAI + LoRA
│   ├── liquidai_nolora_config.yaml  # LiquidAI full fine-tune
│   └── liquidai_eval.yaml       # API-based eval config
└── src/
    ├── config.py                # Configuration dataclasses + YAML loading
    ├── formats.py               # Model-specific: tool schemas, prompt builders, parsers
    ├── utils.py                 # Shared: Action, system prompt, format dispatch
    ├── model.py                 # Model/processor loading, LoRA setup
    ├── environment.py           # DuckHuntEnvWrapper (direct, no HTTP)
    ├── reward.py                # Reward computation
    ├── dataset.py               # Dataset generation, snapshot system, TRL reward fns
    └── trainer.py               # Custom GRPO trainer with checkpointing + W&B
```

## How Format Auto-Detection Works

`train.py` reads the `model.model_name` from config and calls `set_model_format()`:

- `mistralai/*` → `MistralFormat` (JSON tool calls, `[TOOL_CALLS]` tokens)
- `LiquidAI/*` or `LFM*` → `LiquidAIFormat` (Pythonic tool calls, `<|tool_call_start|>` tokens)

All downstream code (`dataset.py`, `trainer.py`, `evaluate.py`) uses the active format transparently through `build_prompt()` and `parse_tool_call()` from `src/utils.py`.

## Training Modes

### TRL GRPOTrainer (default)

Uses HuggingFace TRL's `GRPOTrainer`. Generates an offline dataset of `(prompt, images, snapshot)` tuples, then trains with two reward functions:

```bash
python train.py --config configs/liquidai_config.yaml --num-samples 1000
```

### Custom GRPO Loop

Runs the full GRPO algorithm manually with online environment interaction:

```bash
python train.py --config configs/liquidai_config.yaml --custom
```

Supports resuming from checkpoints:

```bash
python train.py --config configs/liquidai_config.yaml --custom \
    --resume outputs/lfm25_duckhunt_grpo/checkpoint-500
```

## How It Works

### Observation

The model receives consecutive 512x512 game frames (oldest to newest) plus game state metadata (round, bullets remaining, latency). This lets it estimate duck velocity from frame-to-frame displacement.

### Action Space

A single tool call with three parameters:

| Parameter | Type    | Range   | Description                              |
|-----------|---------|---------|------------------------------------------|
| `x`       | float   | 0.0-1.0 | Horizontal position (0=left, 1=right)    |
| `y`       | float   | 0.0-1.0 | Vertical position (0=top, 1=bottom)      |
| `horizon` | integer | 0-30    | Extra frames to wait before shooting     |

The total prediction distance is `processing_latency_frames + horizon`. The model must learn to lead its shots based on estimated duck velocity and the combined latency.

### Reward Function

| Outcome         | Reward |
|-----------------|--------|
| Hit one duck    | +1.0   |
| Double kill     | +2.5   |
| Miss            | -0.3   |
| No target       | -0.5   |
| Invalid output  | -1.0   |

Hits incur a horizon penalty: `-0.1 * (horizon / 30)` to encourage faster shots when possible.

Two reward signals are combined during training:
- **Accuracy reward** (weight 1.0): Environment simulation determines if the shot hits
- **Format reward** (weight 0.3): Bonus for producing a valid tool call structure

### GRPO Algorithm

For each game state:

1. Sample `G` completions from the current policy (4 for Mistral, 6 for LiquidAI)
2. Score each completion against the environment (deterministic replay via snapshot)
3. Normalise advantages within the group
4. Compute the clipped surrogate objective (PPO-style, epsilon=0.2)
5. Optionally add KL penalty against a frozen reference model (beta=0.0 by default)

### Deterministic Replay

To compute rewards for the TRL path (offline dataset), the system captures a **snapshot** of the game state at each training sample — duck positions, velocities, RNG state. During reward evaluation, it restores this snapshot and deterministically simulates the shot, ensuring reproducible outcomes regardless of when the reward function is called.

### Curriculum Training

Training is split into two phases:

1. **Phase 1** (steps 0–2000): horizon clamped to 0, shorter completions — model learns tool call format and (x, y) aiming
2. **Phase 2** (steps 2000+): horizon unlocked — model learns to optimize prediction timing

Configure with `grpo.curriculum_phase2_step: 2000` (set to 0 to disable).

### Environment Diversity

Each training step uses a **new match** with fresh ducks (different spawn positions, speeds, directions). Ducks spawn from left, right, or top edges with per-duck speed variation. Random starting rounds (1-5) ensure varied difficulty.

### Hotspot Penalty

An anti-exploit mechanism tracks recent shot positions. If the model repeatedly aims at the same spot, hits at that position receive scaled-down rewards (going negative at >40% concentration). This prevents the model from finding a single "lucky" position and spamming it.

See [REWARD.md](REWARD.md) for full reward system documentation.

## Configuration

Configs use a layered YAML system. The base config sets defaults; model-specific configs override them:

```bash
# Single config (includes model-specific defaults)
python train.py --config configs/liquidai_config.yaml

# Layered (base + overrides)
python train.py \
    --config configs/base_config.yaml \
    --config configs/liquidai_config.yaml

# CLI overrides (dot-separated)
python train.py --config configs/liquidai_config.yaml \
    --override training.learning_rate=3e-5 \
    --override grpo.num_generations=8
```

### Key Hyperparameters

| Parameter | Ministral | LiquidAI (LoRA) | LiquidAI (Full) |
|-----------|-----------|-----------------|------------------|
| `training.learning_rate` | 5e-6 | 1e-5 | 5e-6 |
| `training.per_device_train_batch_size` | 1 | 2 | 1 |
| `training.gradient_accumulation_steps` | 8 | 4 | 8 |
| `grpo.num_generations` | 4 | 6 | 4 |
| `grpo.max_completion_length` | 256 | 40 | 40 |
| `grpo.curriculum_phase2_step` | 0 | 2000 | 2000 |
| `lora.r` | 16 | 16 | — |
| `lora.lora_alpha` | 32 | 16 | — |

### LoRA

LoRA target modules differ by architecture:

| Model | Target modules | Trainable % |
|-------|---------------|-------------|
| Ministral | `q_proj`, `k_proj`, `v_proj`, `o_proj` | ~0.2% |
| LiquidAI | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, `w1`, `w2`, `w3` | ~1.5% |

LiquidAI's hybrid architecture (10 LIV convolution blocks + 6 GQA blocks) exposes additional MLP gates (`w1`, `w2`, `w3`) and a convolution input projection (`in_proj`).

### Latency Simulation

The environment simulates real-world inference latency. During training, `processing_latency_ms` is randomly sampled:

| Model | Latency options (ms) | Rationale |
|-------|---------------------|-----------|
| Ministral | 100, 200, 300, 400, 500, 600 | Larger model, slower inference |
| LiquidAI | 100, 150, 200, 250, 300 | Measured: 2 frames=100ms, 10 frames=200ms, 20 frames=300ms |

## Checkpointing

The custom trainer saves full checkpoints at every `save_steps`:

```
outputs/lfm25_duckhunt_grpo/
├── checkpoint-100/
│   ├── adapter_model.safetensors    # (or model.safetensors for full fine-tune)
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-200/
├── best_checkpoint/          # Highest eval hit_rate
└── final/
```

- Old checkpoints rotate out based on `save_total_limit` (default: 3)
- `best_checkpoint/` is preserved regardless of rotation
- Resume with `--resume outputs/.../checkpoint-N`

## Logging (W&B)

When `report_to: wandb` is set (default):

**Per step**: loss, learning_rate, gradient_norm, mean_reward, std_reward, KL divergence

**Per eval**: hit_rate, double_kill_rate, miss_rate, average_reward, horizon distribution histogram, sample outputs table, hit rate by latency chart

**Once**: full config (model, LoRA, GRPO, training, reward, environment parameters)

Disable with `--no-wandb` or `--override logging.report_to=none`.

## Pushing to Hugging Face Hub

Upload trained checkpoints to HF Hub after training:

```bash
# Via CLI flags
./run_training.sh --config configs/liquidai_config.yaml \
    --push-to-hub --hub-model-id username/duckhunt-lfm25-grpo

# Via config override
python train.py --config configs/liquidai_config.yaml \
    --override hub.push_to_hub=true \
    --override hub.hub_model_id=username/duckhunt-lfm25-grpo
```

Or set it in your YAML config:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "username/duckhunt-lfm25-grpo"
  hub_private: true  # default
```

What gets uploaded:
- LoRA adapter weights (or full model weights for no-LoRA training)
- Processor/tokenizer files
- Auto-generated model card with training details and tool format info
- Optimizer and scheduler states are excluded to keep the repo lightweight

Requires authentication: `huggingface-cli login` before training.

## Evaluation

### Local Evaluation (checkpoint loaded directly)

```bash
# Evaluate a checkpoint (any model family)
python evaluate.py --config configs/liquidai_config.yaml \
    --checkpoint outputs/lfm25_duckhunt_grpo/best_checkpoint \
    --num-episodes 5

# With baselines comparison
python evaluate.py --config configs/ministral_config.yaml \
    --checkpoint outputs/ministral_duckhunt_grpo/best_checkpoint \
    --baselines --output results.json
```

### API-Based Evaluation (served model)

See [EVAL_VLM.md](EVAL_VLM.md) for full documentation on evaluating models served via vLLM/SGLang.

```bash
# 1. Serve the model
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B

# 2. Run eval
python eval_vlm.py --config configs/liquidai_eval.yaml
```

Evaluation runs across all latency buckets and reports:

- **Core**: hit_rate, double_kill_rate, miss_rate, invalid_action_rate, average_reward
- **Horizon**: average, std, min, max
- **Per-latency**: hit_rate and average_horizon broken down by latency bucket
- **Hardware-aware**: generalization_gap (best - worst latency), adaptation_score

Baselines:
- **Random**: uniform random x, y (upper half), horizon
- **Fixed horizon**: centre shot (0.5, 0.25) with horizon=10

## Serving the Trained Model

The trained model produces tool calls in its native format, compatible with any OpenAI-SDK-compatible server:

```bash
# Serve with vLLM
./serve_vlm.sh --model LiquidAI/LFM2.5-VL-1.6B
# Or with a fine-tuned checkpoint
./serve_vlm.sh --checkpoint outputs/lfm25_duckhunt_grpo/best

# Client (OpenAI SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-VL-1.6B",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "shoot",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "horizon": {"type": "integer"}
                },
                "required": ["x", "y", "horizon"]
            }
        }
    }],
    tool_choice="required",
)
```
