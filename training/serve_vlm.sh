#!/usr/bin/env bash
# =============================================================================
# Serve a VLM for evaluation via vLLM or SGLang (OpenAI-compatible API)
# =============================================================================
#
# Usage:
#   ./serve_vlm.sh                                     # defaults: vLLM, LFM2.5-VL-1.6B
#   ./serve_vlm.sh --model LiquidAI/LFM2-VL-450M
#   ./serve_vlm.sh --backend sglang
#   ./serve_vlm.sh --backend vllm --quantization awq --gpu-memory 0.8
#   ./serve_vlm.sh --checkpoint ./outputs/lfm25_grpo/best  # fine-tuned model
#   ./serve_vlm.sh --stop                              # stop running container

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND="vllm"                                    # vllm | sglang
MODEL="LiquidAI/LFM2.5-VL-1.6B"
CHECKPOINT=""                                     # path to fine-tuned checkpoint
PORT=8000
HOST="0.0.0.0"
CONTAINER_NAME="duckhunt-vlm"
GPU_MEMORY_UTILIZATION=0.90                       # fraction of GPU VRAM to use
MAX_MODEL_LEN=32768                               # max context length
MAX_NUM_SEQS=4                                    # max concurrent sequences
TENSOR_PARALLEL=1                                 # number of GPUs for TP
QUANTIZATION=""                                   # awq | gptq | squeezellm | "" (none)
DTYPE="bfloat16"                                  # float16 | bfloat16 | auto
KV_CACHE_DTYPE="auto"                             # auto | fp8 | fp8_e5m2
ENABLE_CHUNKED_PREFILL=true
MAX_NUM_BATCHED_TOKENS=4096
IMAGE_INPUT_TYPE="pixel_values"
EXTRA_ARGS=""                                     # any extra args to pass through
STOP_MODE=false
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"    # HF cache for model weights

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)          BACKEND="$2";                    shift 2 ;;
        --model)            MODEL="$2";                      shift 2 ;;
        --checkpoint)       CHECKPOINT="$2";                 shift 2 ;;
        --port)             PORT="$2";                       shift 2 ;;
        --host)             HOST="$2";                       shift 2 ;;
        --container-name)   CONTAINER_NAME="$2";             shift 2 ;;
        --gpu-memory)       GPU_MEMORY_UTILIZATION="$2";     shift 2 ;;
        --max-model-len)    MAX_MODEL_LEN="$2";              shift 2 ;;
        --max-num-seqs)     MAX_NUM_SEQS="$2";               shift 2 ;;
        --tensor-parallel)  TENSOR_PARALLEL="$2";            shift 2 ;;
        --quantization)     QUANTIZATION="$2";               shift 2 ;;
        --dtype)            DTYPE="$2";                      shift 2 ;;
        --kv-cache-dtype)   KV_CACHE_DTYPE="$2";             shift 2 ;;
        --max-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2";   shift 2 ;;
        --extra)            EXTRA_ARGS="$2";                 shift 2 ;;
        --stop)             STOP_MODE=true;                  shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend <vllm|sglang>       Serving backend (default: vllm)"
            echo "  --model <model_id>            HuggingFace model ID"
            echo "  --checkpoint <path>           Fine-tuned checkpoint path"
            echo "  --port <port>                 API port (default: 8000)"
            echo "  --gpu-memory <0.0-1.0>        GPU memory fraction (default: 0.90)"
            echo "  --max-model-len <int>         Max context length (default: 32768)"
            echo "  --max-num-seqs <int>          Max concurrent sequences (default: 4)"
            echo "  --tensor-parallel <int>       Number of GPUs (default: 1)"
            echo "  --quantization <method>       Quantization: awq, gptq, squeezellm"
            echo "  --dtype <dtype>               float16, bfloat16, auto"
            echo "  --kv-cache-dtype <dtype>      auto, fp8, fp8_e5m2"
            echo "  --max-batched-tokens <int>    Max batched tokens (default: 4096)"
            echo "  --extra '<args>'              Extra arguments passed to server"
            echo "  --stop                        Stop the running container"
            echo "  --help                        Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────────
if $STOP_MODE; then
    echo "Stopping container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "Done."
    exit 0
fi

# ── Resolve model path ───────────────────────────────────────────────────────
SERVE_MODEL="$MODEL"
VOLUME_MOUNTS="-v $HF_HOME:/root/.cache/huggingface"

if [[ -n "$CHECKPOINT" ]]; then
    # Mount the checkpoint directory into the container
    ABS_CHECKPOINT="$(cd "$CHECKPOINT" && pwd)"
    SERVE_MODEL="/checkpoint"
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $ABS_CHECKPOINT:/checkpoint"
    echo "Serving fine-tuned checkpoint: $ABS_CHECKPOINT"
fi

# ── Stop existing container ──────────────────────────────────────────────────
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# ── Build server command ─────────────────────────────────────────────────────
echo "============================================================"
echo "  Backend:         $BACKEND"
echo "  Model:           $MODEL"
echo "  Checkpoint:      ${CHECKPOINT:-none}"
echo "  Port:            $PORT"
echo "  GPU memory:      $GPU_MEMORY_UTILIZATION"
echo "  Max context:     $MAX_MODEL_LEN"
echo "  Quantization:    ${QUANTIZATION:-none}"
echo "  Dtype:           $DTYPE"
echo "  KV cache dtype:  $KV_CACHE_DTYPE"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo "============================================================"

if [[ "$BACKEND" == "vllm" ]]; then
    # ── vLLM ──────────────────────────────────────────────────────────────
    DOCKER_IMAGE="vllm/vllm-openai:latest"

    CMD="$SERVE_MODEL"
    CMD="$CMD --host $HOST --port $PORT"
    CMD="$CMD --dtype $DTYPE"
    CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
    CMD="$CMD --max-model-len $MAX_MODEL_LEN"
    CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"
    CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL"
    CMD="$CMD --kv-cache-dtype $KV_CACHE_DTYPE"
    CMD="$CMD --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
    CMD="$CMD --trust-remote-code"

    if [[ -n "$QUANTIZATION" ]]; then
        CMD="$CMD --quantization $QUANTIZATION"
    fi

    if $ENABLE_CHUNKED_PREFILL; then
        CMD="$CMD --enable-chunked-prefill"
    fi

    # VLM-specific: enable image input (dot-notation for vLLM 0.17+)
    CMD="$CMD --limit-mm-per-prompt.image 32"

    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Starting vLLM server..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --ipc=host \
        -p "$PORT:$PORT" \
        $VOLUME_MOUNTS \
        --entrypoint /bin/bash \
        "$DOCKER_IMAGE" \
        -c "pip install 'transformers>=5.0.0' --quiet && vllm serve $CMD"

elif [[ "$BACKEND" == "sglang" ]]; then
    # ── SGLang ────────────────────────────────────────────────────────────
    DOCKER_IMAGE="lmsysorg/sglang:latest"

    CMD="python -m sglang.launch_server"
    CMD="$CMD --model-path $SERVE_MODEL"
    CMD="$CMD --host $HOST --port $PORT"
    CMD="$CMD --dtype $DTYPE"
    CMD="$CMD --mem-fraction-static $GPU_MEMORY_UTILIZATION"
    CMD="$CMD --context-length $MAX_MODEL_LEN"
    CMD="$CMD --tp-size $TENSOR_PARALLEL"
    CMD="$CMD --trust-remote-code"

    if [[ -n "$QUANTIZATION" ]]; then
        CMD="$CMD --quantization $QUANTIZATION"
    fi

    if [[ -n "$EXTRA_ARGS" ]]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Starting SGLang server..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --ipc=host \
        -p "$PORT:$PORT" \
        $VOLUME_MOUNTS \
        "$DOCKER_IMAGE" \
        $CMD

else
    echo "Unknown backend: $BACKEND (use vllm or sglang)"
    exit 1
fi

echo ""
echo "Container '$CONTAINER_NAME' started."
echo "Waiting for server to be ready..."

# ── Wait for server ──────────────────────────────────────────────────────────
MAX_WAIT=300  # 5 minutes
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo ""
        echo "Server ready at http://localhost:$PORT"
        echo "Models available:"
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        echo ""
        echo "To stop: $0 --stop"
        exit 0
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    printf "."
done

echo ""
echo "WARNING: Server did not become ready in ${MAX_WAIT}s"
echo "Check logs: docker logs $CONTAINER_NAME"
exit 1
