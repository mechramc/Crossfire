#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE — Mac (Apple Silicon) Environment Setup
# Installs llama.cpp with Metal support
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE Mac Setup ==="
echo "Project root: $PROJECT_ROOT"

# --- Check Apple Silicon ---
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: Expected Apple Silicon (arm64), got $ARCH"
fi

# --- Check Xcode CLI tools ---
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode command line tools..."
    xcode-select --install
    echo "Re-run this script after Xcode CLI tools finish installing."
    exit 0
fi

# --- Clone llama.cpp ---
LLAMA_DIR="$PROJECT_ROOT/vendor/llama.cpp"
if [ -d "$LLAMA_DIR" ]; then
    echo "llama.cpp already exists at $LLAMA_DIR, pulling latest..."
    git -C "$LLAMA_DIR" pull
else
    echo "Cloning llama.cpp..."
    mkdir -p "$PROJECT_ROOT/vendor"
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# --- Build with Metal ---
echo "Building llama.cpp with Metal support..."
cd "$LLAMA_DIR"
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)"

echo ""
echo "=== Mac setup complete ==="
echo "Binaries: $LLAMA_DIR/build/bin/"
echo "Next: download models to $PROJECT_ROOT/models/"
