#!/bin/bash
# Download GGUF models for manual testing
# Usage: ./scripts/download_models.sh [model-name]
#
# Model URLs are sourced from registry.briodocs.ai
#
# Available models:
#   qwen-2.5-7b-instruct    - Qwen 2.5 7B Instruct (Q4_K_M) - 4.4GB
#   qwen-2.5-3b-instruct    - Qwen 2.5 3B Instruct (Q4_K_M) - 2.0GB
#   llama-3.1-8b-instruct   - Llama 3.1 8B Instruct (Q4_K_M) - 4.9GB
#   llama-3.2-3b-instruct   - Llama 3.2 3B Instruct (Q4_K_M) - 2.0GB
#   mistral-7b-instruct     - Mistral 7B Instruct v0.3 (Q4_K_M) - 4.4GB
#   phi-4-mini-instruct     - Phi-4 Mini Instruct (Q4_K_M) - 2.7GB
#   phi-4-reasoning         - Phi-4 Reasoning (Q4_K_M) - 2.7GB
#   all                     - Download all models (~25GB total)

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

download_model() {
    local name=$1
    local url=$2
    local filename=$3

    echo "=========================================="
    echo "Downloading: $name"
    echo "=========================================="

    if [ -f "$MODELS_DIR/$filename" ]; then
        echo "✓ Model already exists: $MODELS_DIR/$filename"
        echo "  To re-download, delete the file first."
        return 0
    fi

    echo "Source: $url"
    echo "Destination: $MODELS_DIR/$filename"
    echo ""

    cd "$MODELS_DIR"
    wget -c "$url" -O "$filename"
    cd ..

    echo ""
    echo "✓ Download complete: $filename"
    echo ""
}

case "$1" in
    qwen-2.5-7b-instruct)
        download_model \
            "Qwen 2.5 7B Instruct" \
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        ;;

    qwen-2.5-3b-instruct)
        download_model \
            "Qwen 2.5 3B Instruct" \
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
            "qwen2.5-3b-instruct-q4_k_m.gguf"
        ;;

    llama-3.1-8b-instruct)
        download_model \
            "Llama 3.1 8B Instruct" \
            "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        ;;

    llama-3.2-3b-instruct)
        download_model \
            "Llama 3.2 3B Instruct" \
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        ;;

    mistral-7b-instruct)
        download_model \
            "Mistral 7B Instruct v0.3" \
            "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
        ;;

    phi-4-mini-instruct)
        download_model \
            "Phi-4 Mini Instruct" \
            "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf" \
            "Phi-4-mini-instruct-Q4_K_M.gguf"
        ;;

    phi-4-reasoning)
        download_model \
            "Phi-4 Reasoning" \
            "https://huggingface.co/unsloth/Phi-4-reasoning-GGUF/resolve/main/phi-4-reasoning-Q4_K_M.gguf" \
            "phi-4-reasoning-Q4_K_M.gguf"
        ;;

    all)
        echo "Downloading all models (~25GB total)..."
        echo ""
        $0 qwen-2.5-7b-instruct
        $0 qwen-2.5-3b-instruct
        $0 llama-3.1-8b-instruct
        $0 llama-3.2-3b-instruct
        $0 mistral-7b-instruct
        $0 phi-4-mini-instruct
        $0 phi-4-reasoning
        echo ""
        echo "=========================================="
        echo "✓ All models downloaded!"
        echo "=========================================="
        ;;

    "")
        echo "Usage: $0 <model-name>"
        echo ""
        echo "Available models (from registry.briodocs.ai):"
        echo "  qwen-2.5-7b-instruct     - Qwen 2.5 7B Instruct (Q4_K_M) - 4.4GB"
        echo "  qwen-2.5-3b-instruct     - Qwen 2.5 3B Instruct (Q4_K_M) - 2.0GB"
        echo "  llama-3.1-8b-instruct    - Llama 3.1 8B Instruct (Q4_K_M) - 4.9GB"
        echo "  llama-3.2-3b-instruct    - Llama 3.2 3B Instruct (Q4_K_M) - 2.0GB"
        echo "  mistral-7b-instruct      - Mistral 7B Instruct v0.3 (Q4_K_M) - 4.4GB"
        echo "  phi-4-mini-instruct      - Phi-4 Mini Instruct (Q4_K_M) - 2.7GB"
        echo "  phi-4-reasoning          - Phi-4 Reasoning (Q4_K_M) - 2.7GB"
        echo "  all                      - Download all models (~25GB total)"
        echo ""
        echo "Examples:"
        echo "  $0 qwen-2.5-7b-instruct    # Start with this one"
        echo "  $0 all                      # Download everything"
        exit 1
        ;;

    *)
        echo "Error: Unknown model '$1'"
        echo "Run '$0' without arguments to see available models."
        exit 1
        ;;
esac
