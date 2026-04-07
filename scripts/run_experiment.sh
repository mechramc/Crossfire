#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE — Experiment Runner
# Runs a single benchmark configuration and saves results
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model PATH        Path to GGUF model file (required)
    --context SIZE      Context size in tokens (default: 2048)
    --kv-compress ALG   KV cache compression algorithm (none|turbo3|turbo4)
    --distributed       Enable distributed mode (prefill/decode split)
    --output DIR        Output directory (default: results/)
    -h, --help          Show this help

Examples:
    # Single-node baseline
    $(basename "$0") --model models/qwen3.5-27b-q8_0.gguf --context 2048

    # With KV cache compression
    $(basename "$0") --model models/qwen3.5-27b-tq4_1s.gguf --context 8192 --kv-compress turbo3

    # Distributed mode
    $(basename "$0") --model models/qwen3.5-27b-tq4_1s.gguf --distributed
EOF
    exit 0
}

# --- Defaults ---
MODEL=""
CONTEXT=2048
KV_COMPRESS="none"
DISTRIBUTED=false
OUTPUT_DIR="$PROJECT_ROOT/results"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --context) CONTEXT="$2"; shift 2 ;;
        --kv-compress) KV_COMPRESS="$2"; shift 2 ;;
        --distributed) DISTRIBUTED=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    usage
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model file not found: $MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="$(basename "$MODEL" .gguf)"
RUN_ID="${MODEL_NAME}_ctx${CONTEXT}_kv${KV_COMPRESS}_${TIMESTAMP}"

echo "=== CROSSFIRE Experiment ==="
echo "Run ID:      $RUN_ID"
echo "Model:       $MODEL"
echo "Context:     $CONTEXT"
echo "KV Compress: $KV_COMPRESS"
echo "Distributed: $DISTRIBUTED"
echo "Output:      $OUTPUT_DIR"
echo ""

# TODO: Implement actual benchmark execution
echo "Experiment runner scaffold — implementation pending."
echo "Run ID $RUN_ID would be saved to $OUTPUT_DIR/$RUN_ID.json"
