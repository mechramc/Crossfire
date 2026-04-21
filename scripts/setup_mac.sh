#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE-X — Mac (Apple Silicon) Environment Setup
# Installs EXO (from source, via uv) + llama.cpp + ANEMLL + Rustane with
# Metal/ANE support.
# Interconnect: TCP/IP over USB4 (primary) with 5GbE fallback (no RDMA).
#
# NOTE: EXO lives at EXO_DIR (default ~/crossfire/exo) as a source clone
# managed by uv. The runtime binary is $EXO_DIR/.venv/bin/exo -- no global
# `exo` is installed. On macOS no extra is required (mlx comes from the
# darwin-gated git source in EXO's pyproject.toml).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE-X Mac Setup ==="
echo "Project root: $PROJECT_ROOT"

# --- Check Apple Silicon ---
ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: Expected Apple Silicon (arm64), got $ARCH"
fi

# --- Check macOS version ---
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

# --- Thunderbolt IP bridge status ---
echo ""
echo "--- Interconnect (Thunderbolt IP bridge + 5GbE) ---"
# Thunderbolt IP presents as an en* interface once the cable is connected and
# macOS negotiates the bridge. Surface its state to the user.
if command -v networksetup &>/dev/null; then
    echo "Network services:"
    networksetup -listallnetworkservices | grep -i -E "thunderbolt|ethernet|usb" || true
    echo ""
    echo "If the Thunderbolt Bridge service is missing or inactive, create it under:"
    echo "  System Settings -> Network -> ... -> Add Service -> Thunderbolt Bridge"
fi

# --- Install / update EXO (from source, managed by uv) ---
echo ""
echo "--- Installing / updating EXO ---"
EXO_DIR="${EXO_DIR:-$HOME/crossfire/exo}"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ -d "$EXO_DIR/.git" ]; then
    echo "EXO clone already exists at $EXO_DIR, pulling latest..."
    git -C "$EXO_DIR" pull --ff-only
else
    echo "Cloning EXO to $EXO_DIR..."
    mkdir -p "$(dirname "$EXO_DIR")"
    git clone https://github.com/exo-explore/exo.git "$EXO_DIR"
fi

echo "Syncing EXO venv (this may be multi-GB on first run)..."
(cd "$EXO_DIR" && uv sync)
echo "EXO binary: $EXO_DIR/.venv/bin/exo"

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

# --- Interconnect probe ---
echo ""
echo "--- Interconnect probe ---"
PC_HOST="${PC_HOST:-192.168.1.100}"
PC_PORT="${PC_PORT:-8080}"
if command -v iperf3 &>/dev/null; then
    echo "iperf3 available. To measure throughput, run on the PC:"
    echo "    iperf3 -s"
    echo "Then on this Mac:"
    echo "    iperf3 -c $PC_HOST -t 10 -P 4"
    echo "Target: ~4-5 GB/s over USB4 (Thunderbolt IP bridge), ~600 MB/s over 5GbE."
else
    echo "iperf3 not installed. Install with: brew install iperf3"
fi

if command -v nc &>/dev/null; then
    if nc -z -w 3 "$PC_HOST" "$PC_PORT" 2>/dev/null; then
        echo "PC reachable on $PC_HOST:$PC_PORT."
    else
        echo "PC NOT reachable on $PC_HOST:$PC_PORT yet. Set PC_HOST to the correct address."
    fi
fi

echo ""
echo "=== Mac setup complete ==="
echo "EXO:       $EXO_DIR/.venv/bin/exo"
echo "llama.cpp: $LLAMA_DIR/build/bin/"
echo "ANEMLL:    $ANEMLL_DIR"
echo "Rustane:   $RUSTANE_DIR"
echo ""
echo "Next steps:"
echo "  1. Download models to $PROJECT_ROOT/models/"
echo "  2. Convert Qwen3.5-0.6B to ANE format via ANEMLL"
echo "  3. Connect USB4 40 Gbps active cable to PC"
echo "  4. Ensure Thunderbolt Bridge service is active in System Settings -> Network"
echo "  5. Verify: iperf3 between nodes (see above)"
echo "  6. Run: $EXO_DIR/.venv/bin/exo  (verify PC is visible)"
