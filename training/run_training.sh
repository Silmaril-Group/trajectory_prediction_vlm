#!/usr/bin/env bash
set -euo pipefail

# ===================================================================
#  Duck Hunt GRPO Training — Launch Script
# ===================================================================
#
#  Supports all model families (Mistral, LiquidAI, etc.) via --config.
#
#  Usage:
#    # Mistral (default)
#    ./run_training.sh
#    ./run_training.sh --config configs/ministral_config.yaml
#
#    # LiquidAI with LoRA
#    ./run_training.sh --config configs/liquidai_config.yaml
#
#    # LiquidAI full fine-tune (no LoRA)
#    ./run_training.sh --config configs/liquidai_nolora_config.yaml
#
#    # Custom GRPO loop
#    ./run_training.sh --config configs/liquidai_config.yaml --custom
#
#    # Resume from checkpoint
#    ./run_training.sh --custom --resume outputs/lfm25_duckhunt_grpo/checkpoint-500
#
#    # Push to Hub
#    ./run_training.sh --push-to-hub --hub-model-id user/duckhunt-grpo
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------------------------------------------------
#  Configuration
# -------------------------------------------------------------------
CONFIG="configs/ministral_config.yaml"
NUM_SAMPLES=1000
MODE="trl"                   # "trl" or "custom"
RESUME_FROM=""
WANDB_PROJECT="duckhunt-grpo"
PUSH_TO_HUB=false
HUB_MODEL_ID=""
EXTRA_OVERRIDES=()

# -------------------------------------------------------------------
#  Parse arguments
# -------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --custom)
            MODE="custom"
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --override)
            EXTRA_OVERRIDES+=("--override" "$2")
            shift 2
            ;;
        --no-wandb)
            EXTRA_OVERRIDES+=("--override" "logging.report_to=none")
            shift
            ;;
        --push-to-hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --hub-model-id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH         Config YAML (default: configs/ministral_config.yaml)"
            echo "  --custom              Use custom GRPO loop instead of TRL"
            echo "  --resume PATH         Resume from checkpoint (custom mode only)"
            echo "  --num-samples N       Training samples to generate (default: 1000, TRL only)"
            echo "  --wandb-project NAME  W&B project name (default: duckhunt-grpo)"
            echo "  --override KEY=VAL    Override config value (repeatable)"
            echo "  --no-wandb            Disable W&B logging"
            echo "  --push-to-hub         Push final model to Hugging Face Hub"
            echo "  --hub-model-id ID     HF repo id (e.g. user/duckhunt-grpo)"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --config configs/ministral_config.yaml           # Mistral + LoRA"
            echo "  $0 --config configs/liquidai_config.yaml            # LiquidAI + LoRA"
            echo "  $0 --config configs/liquidai_nolora_config.yaml     # LiquidAI full fine-tune"
            echo "  $0 --config configs/liquidai_config.yaml --custom   # LiquidAI custom loop"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -------------------------------------------------------------------
#  Environment setup
# -------------------------------------------------------------------

# Install dependencies via uv from root pyproject.toml
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Syncing dependencies (uv sync --extra training) …"
uv sync --project "$PROJECT_ROOT" --extra training

# W&B
export WANDB_PROJECT="$WANDB_PROJECT"

# -------------------------------------------------------------------
#  Create output directory
# -------------------------------------------------------------------
mkdir -p outputs

# -------------------------------------------------------------------
#  Launch training
# -------------------------------------------------------------------

# Extract model name from config for display
MODEL_NAME=$(grep 'model_name:' "$CONFIG" 2>/dev/null | head -1 | sed 's/.*: *"\?\([^"]*\)"\?/\1/' || echo "unknown")
LORA_STATUS=$(grep -q 'enabled: true' "$CONFIG" 2>/dev/null && echo 'yes' || echo 'no')

echo ""
echo "============================================================"
echo "  Duck Hunt GRPO Training"
echo "============================================================"
echo "  Model:       $MODEL_NAME"
echo "  Config:      $CONFIG"
echo "  Mode:        $MODE"
echo "  LoRA:        $LORA_STATUS"
if [ "$MODE" = "trl" ]; then
    echo "  Samples:     $NUM_SAMPLES"
fi
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from: $RESUME_FROM"
fi
if [ "$PUSH_TO_HUB" = true ]; then
    echo "  Push to Hub: $HUB_MODEL_ID"
fi
echo "============================================================"
echo ""

CMD=(uv run --project "$PROJECT_ROOT" python train.py --config "$CONFIG")

if [ "$MODE" = "custom" ]; then
    CMD+=(--custom)
    if [ -n "$RESUME_FROM" ]; then
        CMD+=(--resume "$RESUME_FROM")
    fi
else
    CMD+=(--num-samples "$NUM_SAMPLES")
fi

# HF Hub
if [ "$PUSH_TO_HUB" = true ]; then
    CMD+=(--push-to-hub)
    if [ -n "$HUB_MODEL_ID" ]; then
        CMD+=(--hub-model-id "$HUB_MODEL_ID")
    fi
fi

# Append any extra overrides
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
