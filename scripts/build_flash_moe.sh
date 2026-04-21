#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CROSSFIRE-X — anemll-flash-llama.cpp Build
# Clones (or updates) the Anemll Flash-MoE fork of llama.cpp and builds it with
# the slot-bank GPU-bank feature enabled. Required before execution policy P6
# (Flash-MoE slot-bank) can be selected by AutoPilot.
#
# Detects the platform and sets the correct accelerator flags:
#   macOS (Apple Silicon)  -> Metal
#   Linux (CUDA toolkit)   -> CUDA
#   Linux (no CUDA)        -> CPU only (P6 will run, but without GPU bank)
#
# Outputs:
#   $PROJECT_ROOT/vendor/anemll-flash-llama.cpp/build/bin/llama-cli (with slot-bank)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

FLASH_DIR="${FLASH_DIR:-$PROJECT_ROOT/vendor/anemll-flash-llama.cpp}"
FLASH_REMOTE="${FLASH_REMOTE:-https://github.com/Anemll/anemll-flash-llama.cpp.git}"
FLASH_REF="${FLASH_REF:-main}"

echo "=== CROSSFIRE-X anemll-flash-llama.cpp build ==="
echo "Target dir: $FLASH_DIR"
echo "Remote:     $FLASH_REMOTE ($FLASH_REF)"

# --- Clone or update ---
if [ -d "$FLASH_DIR/.git" ]; then
    echo "Repo exists, pulling latest..."
    git -C "$FLASH_DIR" fetch origin "$FLASH_REF"
    git -C "$FLASH_DIR" checkout "$FLASH_REF"
    git -C "$FLASH_DIR" pull --ff-only origin "$FLASH_REF"
else
    echo "Cloning anemll-flash-llama.cpp..."
    mkdir -p "$(dirname "$FLASH_DIR")"
    git clone "$FLASH_REMOTE" "$FLASH_DIR"
    if [ "$FLASH_REF" != "main" ]; then
        git -C "$FLASH_DIR" checkout "$FLASH_REF"
    fi
fi

# --- Detect platform / accelerator ---
OS_NAME="$(uname -s)"
ACCEL_FLAGS=()
JOBS="${JOBS:-}"

case "$OS_NAME" in
    Darwin)
        ARCH="$(uname -m)"
        if [ "$ARCH" != "arm64" ]; then
            echo "WARNING: Expected Apple Silicon (arm64), got $ARCH. Metal build may fail."
        fi
        echo "Platform: macOS $ARCH -> Metal backend"
        ACCEL_FLAGS+=("-DGGML_METAL=ON" "-DLLAMA_FLASH_MOE_GPU_BANK=ON")
        [ -z "$JOBS" ] && JOBS="$(sysctl -n hw.ncpu)"
        ;;
    Linux)
        if command -v nvcc &>/dev/null; then
            echo "Platform: Linux + CUDA $(nvcc --version | grep release | awk '{print $6}')"
            ACCEL_FLAGS+=("-DGGML_CUDA=ON" "-DLLAMA_FLASH_MOE_GPU_BANK=ON")
        else
            echo "Platform: Linux (no CUDA detected) -> CPU-only build"
            echo "WARNING: P6 slot-bank will work but without GPU-bank acceleration."
            ACCEL_FLAGS+=("-DLLAMA_FLASH_MOE_GPU_BANK=OFF")
        fi
        [ -z "$JOBS" ] && JOBS="$(nproc)"
        ;;
    *)
        echo "ERROR: unsupported platform: $OS_NAME"
        exit 1
        ;;
esac

# --- Build ---
echo ""
echo "--- Configuring with cmake ---"
cd "$FLASH_DIR"
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    "${ACCEL_FLAGS[@]}"

echo ""
echo "--- Building (-j $JOBS) ---"
cmake --build build --config Release -j "$JOBS"

# --- Locate llama-cli binary ---
CLI_BIN=""
for candidate in \
    "$FLASH_DIR/build/bin/llama-cli" \
    "$FLASH_DIR/build/bin/main" \
    "$FLASH_DIR/build/llama-cli"; do
    if [ -x "$candidate" ]; then
        CLI_BIN="$candidate"
        break
    fi
done

echo ""
echo "=== Build complete ==="
if [ -n "$CLI_BIN" ]; then
    echo "llama-cli: $CLI_BIN"
else
    echo "WARNING: could not locate llama-cli binary under $FLASH_DIR/build/"
    echo "         Check the CMake output above for build errors."
fi
echo ""
echo "Next steps:"
echo "  1. Extract MoE sidecar (run Anemll's sidecar extraction tool against the model)"
echo "  2. Set flash_moe_available=True in HardwareAvailability"
echo "  3. AutoPilot P6 policy is now selectable (see tests/test_policy.py)"
