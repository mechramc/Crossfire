#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE-X — PC/WSL2 Environment Setup
# Installs EXO (from source, via uv) + llama.cpp (TurboQuant+ fork) with CUDA.
# Interconnect: TCP/IP over USB4 (primary) with 5GbE fallback (no RDMA).
#
# NOTE: This script is intended to run INSIDE WSL2 (Ubuntu). EXO lives at
# EXO_DIR (default ~/crossfire/exo) as a source clone managed by uv -- there
# is no global `exo` on PATH; the runtime binary is $EXO_DIR/.venv/bin/exo.
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

# --- Install / update EXO (from source, managed by uv) ---
echo ""
echo "--- Installing / updating EXO ---"
EXO_DIR="${EXO_DIR:-$HOME/crossfire/exo}"

# Pick the CUDA extra based on the installed toolkit. EXO's pyproject.toml
# requires an explicit --extra (cuda12 / cuda13 / cpu); a plain `uv sync`
# will UNINSTALL all nvidia-* and mlx-cuda-* packages.
CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9]*\).*/\1/p')"
case "$CUDA_MAJOR" in
    13) EXO_EXTRA="cuda13" ;;
    12) EXO_EXTRA="cuda12" ;;
    *)  echo "WARNING: unrecognized CUDA major version '$CUDA_MAJOR'; defaulting to cpu extra"
        EXO_EXTRA="cpu" ;;
esac
echo "EXO extra: $EXO_EXTRA"

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

echo "Syncing EXO venv with --extra $EXO_EXTRA (this may be multi-GB on first run)..."
(cd "$EXO_DIR" && uv sync --extra "$EXO_EXTRA")

# EXO's exo.shared.constants imports dashboard assets at module load -- without
# a dashboard build, the exo binary fails at import time. Must run after every
# fresh clone / upstream pull.
if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found. Install Node 22 LTS and re-run."
    exit 1
fi
echo "Building EXO dashboard assets (npm install && npm run build)..."
(cd "$EXO_DIR/dashboard" && npm install && npm run build)
echo "EXO binary: $EXO_DIR/.venv/bin/exo"

# --- Clone llama.cpp (TurboQuant+ fork) ---
LLAMA_DIR="$PROJECT_ROOT/vendor/llama.cpp"
if [ -d "$LLAMA_DIR" ]; then
    echo "llama.cpp already exists at $LLAMA_DIR, pulling latest..."
    git -C "$LLAMA_DIR" pull
else
    echo "Cloning llama.cpp (TurboQuant+ fork)..."
    mkdir -p "$PROJECT_ROOT/vendor"
    git clone https://github.com/TheTom/llama-cpp-turboquant.git "$LLAMA_DIR"
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
echo "EXO:       $EXO_DIR/.venv/bin/exo"
echo "llama.cpp: $LLAMA_DIR/build/bin/"
echo ""
echo "Next steps:"
echo "  1. Download models to $PROJECT_ROOT/models/"
echo "  2. Connect USB4 40 Gbps active cable to Mac Studio"
echo "  3. Configure the Thunderbolt IP bridge on both machines (static IPs or link-local)"
echo "  4. Verify: iperf3 between nodes (see above)"
echo "  5. Run: $EXO_DIR/.venv/bin/exo  (verify Mac is visible)"
