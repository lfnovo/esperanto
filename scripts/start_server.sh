#!/bin/bash
# Start llama.cpp server with specified model
# Usage: ./scripts/start_server.sh <number>

set -e

CHOICE=$1
MODELS_DIR="models"
PORT=8765
HOST="127.0.0.1"
THREADS=8

if [ -z "$CHOICE" ]; then
    echo "Usage: $0 <number>"
    echo ""
    echo "Available models:"
    echo "  1  qwen-2.5-7b-instruct     - Qwen 2.5 7B Instruct (Tier 1: 8K context, GPU)"
    echo "  2  qwen-2.5-3b-instruct     - Qwen 2.5 3B Instruct (Tier 2: 4K context)"
    echo "  3  llama-3.1-8b-instruct    - Llama 3.1 8B Instruct (Tier 1: 8K context, GPU)"
    echo "  4  llama-3.2-3b-instruct    - Llama 3.2 3B Instruct (Tier 2: 4K context)"
    echo "  5  mistral-7b-instruct      - Mistral 7B Instruct v0.3 (Tier 2: 4K context)"
    echo "  6  phi-4-mini-instruct      - Phi-4 Mini Instruct (Tier 3: 2K context, CPU-only)"
    echo "  7  phi-4-reasoning           - Phi-4 Reasoning (Tier 2: 4K context, GPU)"
    echo ""
    echo "Example:"
    echo "  $0 1    # Start Qwen 2.5 7B Instruct"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Error: Port $PORT is already in use!"
    echo "Stop the existing server first:"
    echo "  ./scripts/stop_server.sh"
    exit 1
fi

# Configure based on choice
case $CHOICE in
    1)
        MODEL_NAME="Qwen 2.5 7B Instruct"
        MODEL_FILE="$MODELS_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=8192
        GPU_LAYERS=-1
        TIER="Tier 1 (High Performance)"
        ;;
    2)
        MODEL_NAME="Qwen 2.5 3B Instruct"
        MODEL_FILE="$MODELS_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=4096
        GPU_LAYERS=-1
        TIER="Tier 2 (Balanced)"
        ;;
    3)
        MODEL_NAME="Llama 3.1 8B Instruct"
        MODEL_FILE="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        CONTEXT=8192
        GPU_LAYERS=-1
        TIER="Tier 1 (High Performance)"
        ;;
    4)
        MODEL_NAME="Llama 3.2 3B Instruct"
        MODEL_FILE="$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        CHAT_FORMAT="llama-3"
        CONTEXT=4096
        GPU_LAYERS=-1
        TIER="Tier 2 (Balanced)"
        ;;
    5)
        MODEL_NAME="Mistral 7B Instruct v0.3"
        MODEL_FILE="$MODELS_DIR/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        CHAT_FORMAT="mistral-instruct"
        CONTEXT=4096
        GPU_LAYERS=-1
        TIER="Tier 2 (Balanced)"
        ;;
    6)
        MODEL_NAME="Phi-4 Mini Instruct"
        MODEL_FILE="$MODELS_DIR/Phi-4-mini-instruct-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=2048
        GPU_LAYERS=0  # CPU-only for Tier 3
        TIER="Tier 3 (Fast - CPU Only)"
        ;;
    7)
        MODEL_NAME="Phi-4 Reasoning"
        MODEL_FILE="$MODELS_DIR/phi-4-reasoning-Q4_K_M.gguf"
        CHAT_FORMAT="chatml"
        CONTEXT=4096  # Increased for reasoning tasks
        GPU_LAYERS=-1  # Use GPU for reasoning model performance
        TIER="Tier 2 (Reasoning - GPU Accelerated)"
        ;;
    *)
        echo "Error: Invalid choice '$CHOICE'. Must be 1-7."
        echo "Run '$0' without arguments to see available models."
        exit 1
        ;;
esac

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
echo "Model:        $MODEL_NAME"
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

# Check if llama_cpp is installed
if ! python -c "import llama_cpp" 2>/dev/null; then
    echo "Error: llama-cpp-python not installed!"
    echo ""
    echo "Install it with:"
    echo "  pip install llama-cpp-python[server]"
    echo ""
    echo "Or if using conda:"
    echo "  conda activate base"
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
