#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE v2 — Experiment Runner
# Runs a single benchmark configuration from the ablation matrix (C0-C6)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --config CONFIG     Ablation config (c0|c1|c2|c3|c4|c5|c6) (required)
    --model PATH        Path to GGUF model file (required)
    --context SIZE      Context size in tokens (default: 8192)
    --draft-model PATH  Path to ANE draft model (for c1/c5/c6)
    --output DIR        Output directory (default: results/)
    -h, --help          Show this help

Ablation Configs:
    c0  Pure EXO baseline (5090 prefill + Mac GPU decode)
    c1  ANE speculative decode (draft 0.6B on ANE)
    c2  ANE prefill offload
    c3  Mac-only disaggregated (no 5090)
    c4  TQ+ compression only (EXO + TQ4_1S + turbo3)
    c5  Full stack: EXO + ANE draft + TQ+ (headline result)
    c6  72B stretch goal

Examples:
    # EXO baseline
    $(basename "$0") --config c0 --model models/qwen3.5-27b-q8_0.gguf

    # ANE speculative decode
    $(basename "$0") --config c1 --model models/qwen3.5-27b-q8_0.gguf \\
        --draft-model models/qwen3.5-0.6b-ane/

    # Full stack (headline)
    $(basename "$0") --config c5 --model models/qwen3.5-27b-tq4_1s.gguf \\
        --draft-model models/qwen3.5-0.6b-ane/ --context 32768
EOF
    exit 0
}

# --- Defaults ---
CONFIG=""
MODEL=""
CONTEXT=8192
DRAFT_MODEL=""
OUTPUT_DIR="$PROJECT_ROOT/results"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --context) CONTEXT="$2"; shift 2 ;;
        --draft-model) DRAFT_MODEL="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required"
    usage
fi

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required"
    usage
fi

if [ ! -f "$MODEL" ] && [ ! -d "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# Validate ANE configs have draft model
case "$CONFIG" in
    c1|c5|c6)
        if [ -z "$DRAFT_MODEL" ]; then
            echo "ERROR: Config $CONFIG requires --draft-model (ANE draft model path)"
            exit 1
        fi
        ;;
esac

mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="$(basename "$MODEL" .gguf)"
RUN_ID="${CONFIG}_${MODEL_NAME}_ctx${CONTEXT}_${TIMESTAMP}"

echo "=== CROSSFIRE v2 Experiment ==="
echo "Run ID:      $RUN_ID"
echo "Config:      $CONFIG"
echo "Model:       $MODEL"
echo "Context:     $CONTEXT"
echo "Draft Model: ${DRAFT_MODEL:-none}"
echo "Output:      $OUTPUT_DIR"
echo ""

# TODO: Implement actual benchmark execution per ablation config
echo "Experiment runner scaffold — implementation pending."
echo "Run ID $RUN_ID would be saved to $OUTPUT_DIR/$RUN_ID.json"
