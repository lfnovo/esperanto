#!/bin/bash
# Start llama.cpp server with specified model
# Usage: ./scripts/start_server.sh <model-name>
#
# Available models:
#   qwen-2.5-7b-instruct
#   qwen-2.5-3b-instruct
#   llama-3.1-8b-instruct
#   llama-3.2-3b-instruct
#   mistral-7b-instruct
#   phi-4-mini

set -e

MODEL=$1
MODELS_DIR="models"
PORT=8765
HOST="127.0.0.1"

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-name>"
    echo ""
    echo "Available models:"
    echo "  qwen-2.5-7b-instruct    - Qwen 2.5 7B (Tier 1: 8K context, GPU)"
    echo "  qwen-2.5-3b-instruct    - Qwen 2.5 3B (Tier 2: 4K context)"
    echo "  llama-3.1-8b-instruct   - Llama 3.1 8B (Tier 1: 8K context, GPU)"
    echo "  llama-3.2-3b-instruct   - Llama 3.2 3B (Tier 2: 4K context)"
    echo "  mistral-7b-instruct     - Mistral 7B (Tier 2: 4K context)"
    echo "  phi-4-mini-instruct     - Phi-4 Mini Instruct (Tier 2: 4K context)"
    echo "  phi-4-reasoning         - Phi-4 Reasoning (Tier 2: 4K context)"
    echo ""
    echo "Example:"
    echo "  $0 qwen-2.5-7b-instruct"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Error: Port $PORT is already in use!"
    echo "Stop the existing server first:"
    echo "  kill \$(lsof -t -i:$PORT)"
    echo "Or run:"
    echo "  ./scripts/stop_server.sh"
    exit 1
fi

case $MODEL in
    qwen-2.5-7b-instruct)
        MODEL_FILE="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=8192
        GPU_LAYERS=-1  # Use all GPU layers
        THREADS=8
        TIER="Tier 1 (High Performance)"
        ;;

    qwen-2.5-3b-instruct)
        MODEL_FILE="$MODELS_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 2 (Balanced)"
        ;;

    llama-3.1-8b-instruct)
        MODEL_FILE="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        CONTEXT=8192
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 1 (High Performance)"
        ;;

    llama-3.2-3b-instruct)
        MODEL_FILE="$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 2 (Balanced)"
        ;;

    mistral-7b-instruct)
        MODEL_FILE="$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        CHAT_FORMAT="mistral-instruct"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 2 (Balanced)"
        ;;

    phi-4-mini-instruct)
        MODEL_FILE="$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 2 (Balanced)"
        ;;

    phi-4-reasoning)
        MODEL_FILE="$MODELS_DIR/phi-4-reasoning-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        TIER="Tier 2 (Balanced - Reasoning)"
        ;;

    *)
        echo "Error: Unknown model '$MODEL'"
        echo "Run '$0' without arguments to see available models."
        exit 1
        ;;
esac

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo ""
    echo "Download it first:"
    echo "  ./scripts/download_models.sh $MODEL"
    exit 1
fi

echo "=========================================="
echo "Starting llama.cpp Server"
echo "=========================================="
echo "Model:        $MODEL"
echo "Tier:         $TIER"
echo "Model file:   $MODEL_FILE"
echo "Chat format:  $CHAT_FORMAT"
echo "Context size: $CONTEXT tokens"
echo "GPU layers:   $GPU_LAYERS"
echo "Threads:      $THREADS"
echo "Host:         $HOST"
echo "Port:         $PORT"
echo "=========================================="
echo ""
echo "Server will be available at: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Make sure we're in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: Not in a conda environment!"
    echo "Run: conda activate briodocs"
    echo ""
fi

# Start the server
python -m llama_cpp.server \
    --model "$MODEL_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    --n_ctx "$CONTEXT" \
    --n_gpu_layers "$GPU_LAYERS" \
    --use_mlock True \
    --n_threads "$THREADS" \
    --chat_format "$CHAT_FORMAT"
