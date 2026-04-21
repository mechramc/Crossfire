#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE-X — PC/WSL2 Environment Setup
# Installs EXO + llama.cpp (TurboQuant+ fork) with CUDA support.
# Interconnect: TCP/IP over USB4 (primary) with 5GbE fallback (no RDMA).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CROSSFIRE-X PC Setup ==="
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

# --- Interconnect probe ---
echo ""
echo "--- Interconnect probe (TCP/IP over USB4 / 5GbE) ---"
MAC_HOST="${MAC_HOST:-192.168.1.101}"
MAC_PORT="${MAC_PORT:-8080}"
if command -v iperf3 &>/dev/null; then
    echo "iperf3 available. To measure USB4 throughput, run on the Mac:"
    echo "    iperf3 -s"
    echo "Then on this PC:"
    echo "    iperf3 -c $MAC_HOST -t 10 -P 4"
    echo "Target: ~4-5 GB/s over USB4 (Thunderbolt IP bridge), ~600 MB/s over 5GbE."
else
    echo "iperf3 not installed. Install with: apt-get install iperf3  (or brew on Mac)."
fi

# --- Connectivity check ---
if command -v nc &>/dev/null; then
    if nc -z -w 3 "$MAC_HOST" "$MAC_PORT" 2>/dev/null; then
        echo "Mac reachable on $MAC_HOST:$MAC_PORT."
    else
        echo "Mac NOT reachable on $MAC_HOST:$MAC_PORT yet. Set MAC_HOST to the correct address."
    fi
fi

echo ""
echo "=== PC setup complete ==="
echo "EXO:       $(command -v exo || echo 'install manually')"
echo "llama.cpp: $LLAMA_DIR/build/bin/"
echo ""
echo "Next steps:"
echo "  1. Download models to $PROJECT_ROOT/models/"
echo "  2. Connect USB4 40 Gbps active cable to Mac Studio"
echo "  3. Configure the Thunderbolt IP bridge on both machines (static IPs or link-local)"
echo "  4. Verify: iperf3 between nodes (see above)"
echo "  5. Run: exo discover  (verify Mac is visible)"
