#!/bin/bash
# Start llama.cpp server with tier-based configuration
# Usage: ./scripts/start_server_v2.sh --tier TIER_NUM --model MODEL_NUM
#
# Example:
#   ./scripts/start_server_v2.sh --tier 1 --model 3
#   (Runs Llama 3.1 8B with Tier 1 settings: 8K context, GPU)

set -e

MODELS_DIR="models"
PORT=8765
HOST="127.0.0.1"
CONFIG_FILE="fixtures/briodocs_config.yaml"

# Parse arguments
TIER=""
MODEL_NUM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --model)
            MODEL_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$TIER" ] || [ -z "$MODEL_NUM" ]; then
    echo "Usage: $0 --tier TIER_NUM --model MODEL_NUM"
    echo ""
    echo "Tiers:"
    echo "  1  - High Performance (8K context, GPU)"
    echo "  2  - Balanced (4K context, GPU)"
    echo "  3  - Fast (2K context, CPU-only)"
    echo ""
    echo "Models:"
    echo "  1  - Qwen 2.5 7B Instruct"
    echo "  2  - Qwen 2.5 3B Instruct"
    echo "  3  - Llama 3.1 8B Instruct"
    echo "  4  - Llama 3.2 3B Instruct"
    echo "  5  - Mistral 7B Instruct v0.3"
    echo "  6  - Phi-4 Mini Instruct"
    echo "  7  - Phi-4 Reasoning"
    echo ""
    echo "Examples:"
    echo "  $0 --tier 1 --model 3    # Llama 3.1 8B with Tier 1 settings"
    echo "  $0 --tier 2 --model 7    # Phi-4 Reasoning with Tier 2 settings"
    echo "  $0 --tier 3 --model 6    # Phi-4 Mini with Tier 3 settings"
    exit 1
fi

# Model configuration
case $MODEL_NUM in
    1)
        MODEL_NAME="Qwen 2.5 7B Instruct"
        MODEL_FILE="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        ;;
    2)
        MODEL_NAME="Qwen 2.5 3B Instruct"
        MODEL_FILE="$MODELS_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
        CHAT_FORMAT="chatml"
        ;;
    3)
        MODEL_NAME="Llama 3.1 8B Instruct"
        MODEL_FILE="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        ;;
    4)
        MODEL_NAME="Llama 3.2 3B Instruct"
        MODEL_FILE="$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        ;;
    5)
        MODEL_NAME="Mistral 7B Instruct v0.3"
        MODEL_FILE="$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        CHAT_FORMAT="mistral-instruct"
        ;;
    6)
        MODEL_NAME="Phi-4 Mini Instruct"
        MODEL_FILE="$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        ;;
    7)
        MODEL_NAME="Phi-4 Reasoning"
        MODEL_FILE="$MODELS_DIR/phi-4-reasoning-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        ;;
    *)
        echo "Error: Invalid model number '$MODEL_NUM'. Must be 1-7."
        exit 1
        ;;
esac

# Read tier configuration from YAML
# This is a simple parser - for production use a proper YAML parser
case $TIER in
    1)
        TIER_NAME="High Performance"
        CONTEXT=8192
        GPU_LAYERS=-1
        THREADS=8
        ;;
    2)
        TIER_NAME="Balanced"
        CONTEXT=4096
        GPU_LAYERS=-1
        THREADS=8
        ;;
    3)
        TIER_NAME="Fast"
        CONTEXT=2048
        GPU_LAYERS=0
        THREADS=8
        ;;
    *)
        echo "Error: Invalid tier '$TIER'. Must be 1, 2, or 3."
        exit 1
        ;;
esac

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Error: Port $PORT is already in use!"
    echo "Stop the existing server first:"
    echo "  ./scripts/stop_server.sh"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo ""
    echo "Download it first:"
    echo "  ./scripts/download_models.sh"
    exit 1
fi

echo "=========================================="
echo "Starting llama.cpp Server"
echo "=========================================="
echo "Tier:         $TIER_NAME (Tier $TIER)"
echo "Model:        $MODEL_NAME"
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

# Check if llama_cpp is installed
if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "Error: llama-cpp-python not installed!"
    echo ""
    echo "Install it with:"
    echo "  pip install llama-cpp-python[server]"
    echo ""
    exit 1
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
