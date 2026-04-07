#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE v2 — Mac (Apple Silicon) Environment Setup
# Installs EXO + llama.cpp + ANEMLL + Rustane with Metal/ANE support
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE v2 Mac Setup ==="
echo "Project root: $PROJECT_ROOT"

# --- Check Apple Silicon ---
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: Expected Apple Silicon (arm64), got $ARCH"
fi

# --- Check macOS version (need 26.2+ for RDMA) ---
MACOS_VERSION="$(sw_vers -productVersion)"
echo "macOS version: $MACOS_VERSION"

# --- Check Xcode CLI tools ---
if ! xcode-select -p &>/dev/null; then
    echo "Installing Xcode command line tools..."
    xcode-select --install
    echo "Re-run this script after Xcode CLI tools finish installing."
    exit 0
fi

# --- Set Metal memory limit ---
echo ""
echo "--- Metal Memory Configuration ---"
CURRENT_LIMIT="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 'not set')"
echo "Current iogpu.wired_limit_mb: $CURRENT_LIMIT"
if [ "$CURRENT_LIMIT" != "58982" ]; then
    echo "Setting iogpu.wired_limit_mb=58982 (unlocks ~90% of 64GB for Metal)..."
    sudo sysctl iogpu.wired_limit_mb=58982
fi

# --- Check RDMA status ---
echo ""
echo "--- RDMA Status ---"
if command -v rdma_ctl &>/dev/null; then
    echo "RDMA tools available. Enable via Recovery mode: rdma_ctl enable"
else
    echo "WARNING: rdma_ctl not found. RDMA requires macOS Tahoe 26.2+."
    echo "Enable via Recovery mode after upgrading."
fi

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

# --- Build with Metal ---
echo ""
echo "--- Building llama.cpp with Metal support ---"
cd "$LLAMA_DIR"
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)"

# --- Clone ANEMLL ---
echo ""
echo "--- Setting up ANEMLL (ANE inference) ---"
ANEMLL_DIR="$PROJECT_ROOT/vendor/anemll"
if [ -d "$ANEMLL_DIR" ]; then
    echo "ANEMLL already exists at $ANEMLL_DIR, pulling latest..."
    git -C "$ANEMLL_DIR" pull
else
    echo "Cloning ANEMLL..."
    git clone https://github.com/Anemll/Anemll.git "$ANEMLL_DIR"
fi
echo "Install ANEMLL deps: pip install -r $ANEMLL_DIR/requirements.txt"

# --- Clone Rustane ---
echo ""
echo "--- Setting up Rustane (Rust ANE engine) ---"
RUSTANE_DIR="$PROJECT_ROOT/vendor/rustane"
if [ -d "$RUSTANE_DIR" ]; then
    echo "Rustane already exists at $RUSTANE_DIR, pulling latest..."
    git -C "$RUSTANE_DIR" pull
else
    echo "Cloning Rustane..."
    git clone https://github.com/ncdrone/rustane.git "$RUSTANE_DIR"
fi
if command -v cargo &>/dev/null; then
    echo "Rust toolchain available. Build with: cd $RUSTANE_DIR && cargo build --release"
else
    echo "WARNING: Rust not installed. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

echo ""
echo "=== Mac setup complete ==="
echo "EXO:       $(command -v exo || echo 'install manually')"
echo "llama.cpp: $LLAMA_DIR/build/bin/"
echo "ANEMLL:    $ANEMLL_DIR"
echo "Rustane:   $RUSTANE_DIR"
echo ""
echo "Next steps:"
echo "  1. Download models to $PROJECT_ROOT/models/"
echo "  2. Convert Qwen3.5-0.6B to ANE format via ANEMLL"
echo "  3. Connect Thunderbolt 5 cable to PC"
echo "  4. Enable RDMA: boot to Recovery mode -> rdma_ctl enable"
echo "  5. Run: exo discover  (verify PC is visible)"
