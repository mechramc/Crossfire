#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE — PC/WSL2 Environment Setup
# Installs llama.cpp (TurboQuant+ fork) with CUDA support
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE PC Setup ==="
echo "Project root: $PROJECT_ROOT"

# --- Check CUDA availability ---
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: CUDA toolkit not found. Install CUDA before running this script."
    exit 1
fi
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"

# --- Clone llama.cpp (TurboQuant+ fork) ---
LLAMA_DIR="$PROJECT_ROOT/vendor/llama.cpp"
if [ -d "$LLAMA_DIR" ]; then
    echo "llama.cpp already exists at $LLAMA_DIR, pulling latest..."
    git -C "$LLAMA_DIR" pull
else
    echo "Cloning llama.cpp (TurboQuant+ fork)..."
    mkdir -p "$PROJECT_ROOT/vendor"
    git clone https://github.com/TheTom/llama.cpp.git "$LLAMA_DIR"
fi

# --- Build with CUDA ---
echo "Building llama.cpp with CUDA support..."
cd "$LLAMA_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"

echo ""
echo "=== PC setup complete ==="
echo "Binaries: $LLAMA_DIR/build/bin/"
echo "Next: download models to $PROJECT_ROOT/models/"
