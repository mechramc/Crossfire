#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE v2 — PC/WSL2 Environment Setup
# Installs EXO + llama.cpp (TurboQuant+ fork) with CUDA support
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE v2 PC Setup ==="
echo "Project root: $PROJECT_ROOT"

# --- Check CUDA availability ---
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: CUDA toolkit not found. Install CUDA before running this script."
    exit 1
fi
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"

# --- Install EXO ---
echo ""
echo "--- Installing EXO 1.0 ---"
if command -v exo &>/dev/null; then
    echo "EXO already installed: $(exo --version 2>/dev/null || echo 'version unknown')"
else
    echo "Installing EXO..."
    pip install exo-inference
fi

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
echo ""
echo "--- Building llama.cpp with CUDA support ---"
cd "$LLAMA_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"

echo ""
echo "=== PC setup complete ==="
echo "EXO:       $(command -v exo || echo 'install manually')"
echo "llama.cpp: $LLAMA_DIR/build/bin/"
echo ""
echo "Next steps:"
echo "  1. Download models to $PROJECT_ROOT/models/"
echo "  2. Connect Thunderbolt 5 cable to Mac Studio"
echo "  3. Run: exo discover  (verify Mac is visible)"
