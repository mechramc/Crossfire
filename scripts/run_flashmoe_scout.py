#!/usr/bin/env python3
"""Scout Flash-MoE sidecar viability for a MoE GGUF.

Inspects the GGUF MoE tensor layout, optionally extracts + verifies a sidecar,
and can run a single smoke inference through the vendored
anemll-flash-llama.cpp binary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crossfire.flashmoe import FlashMoEMode, FlashMoERuntime, SlotBankConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="path to the source MoE GGUF")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("vendor/anemll-flash-llama.cpp/build/bin/llama-completion"),
        help=(
            "path to the anemll-flash-llama.cpp batch-completion binary "
            "(llama-completion; llama-cli forces an interactive REPL that "
            "cannot be disabled in this fork)"
        ),
    )
    parser.add_argument(
        "--sidecar-out",
        type=Path,
        default=Path("results/flashmoe_sidecar"),
        help="output directory for sidecar extraction",
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="include shared expert tensors during extraction",
    )
    parser.add_argument("--layers", type=str, help="optional layer filter, e.g. 1,2,4-7")
    parser.add_argument("--families", type=str, help="optional family filter")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract and verify the sidecar after inspection",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a single llama-cli smoke inference after extraction",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Make a poem about Apple Neural Engine in 4 lines.",
        help="prompt used for --smoke",
    )
    parser.add_argument("--ctx-size", type=int, default=4096, help="context size for --smoke")
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="max generated tokens for --smoke"
    )
    parser.add_argument("--slot-bank", type=int, default=16, help="slot-bank size for --smoke")
    parser.add_argument("--topk", type=int, default=4, help="routed top-k override for --smoke")
    parser.add_argument(
        "--mode",
        choices=("stock", "resident-bank", "slot-bank", "oracle-all-hit"),
        default="slot-bank",
        help="Flash-MoE runtime mode for --smoke",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    runtime = FlashMoERuntime(
        binary_path=args.binary,
        mode=FlashMoEMode.STOCK,
        slot_bank=SlotBankConfig(slots_per_layer=args.slot_bank, topk=args.topk),
    )

    inspection = runtime.inspect_sidecar(
        args.model,
        include_shared=args.include_shared,
        layers=args.layers,
        families=args.families,
    )
    print(json.dumps(inspection, indent=2, sort_keys=False))

    if not args.extract and not args.smoke:
        return 0

    sidecar = runtime.extract_sidecar(
        args.model,
        args.sidecar_out,
        include_shared=args.include_shared,
        force=True,
        layers=args.layers,
        families=args.families,
        verify=True,
    )
    print(f"\nsidecar extracted to: {sidecar.sidecar_path}")

    if not args.smoke:
        return 0

    smoke_runtime = FlashMoERuntime(
        binary_path=args.binary,
        mode=FlashMoEMode(args.mode),
        sidecar=sidecar,
        slot_bank=SlotBankConfig(
            slots_per_layer=args.slot_bank,
            topk=args.topk,
            prefetch_temporal=True,
        ),
    )
    stats = smoke_runtime.run_inference(
        args.model,
        prompt=args.prompt,
        context_size=args.ctx_size,
        max_tokens=args.max_tokens,
    )
    print("\nsmoke stats:")
    print(json.dumps(stats.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
