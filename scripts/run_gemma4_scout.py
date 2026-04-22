#!/usr/bin/env python3
"""Gemma 4 E2B chunked ANE scout — coherent-text + TTFT + tok/s on M4 Max.

Loads the chunked CoreML bundle, generates text for a prompt, reports
timing. Replaces the throwaway `/tmp/crossfire_gemma4_scout.py` used in
Session 17. Target acceptance bar for T-0609a: coherent English output,
decode >= 20 tok/s, no NaN/crashes.

Example:
    python scripts/run_gemma4_scout.py \\
        --bundle models/gemma-4-E2B-coreml \\
        --prompt "The capital of France is" \\
        --max-tokens 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crossfire.ane.gemma4_chunked import Gemma4ChunkedEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("models/gemma-4-E2B-coreml"),
        help="path to the pre-converted Gemma 4 E2B CoreML bundle",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="prompt text (BOS is prepended automatically)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="maximum new tokens to generate",
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="cpu_and_ne",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
        help="CoreML compute unit selection",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="keep decoding past EOS up to --max-tokens",
    )
    args = parser.parse_args()

    if not args.bundle.is_dir():
        print(f"error: bundle not found: {args.bundle}", file=sys.stderr)
        return 2

    print(f"loading bundle from {args.bundle} (compute_units={args.compute_units})...")
    engine = Gemma4ChunkedEngine.load(args.bundle, compute_units=args.compute_units)
    print(
        f"  chunks={engine.num_chunks} "
        f"effective_context={engine.effective_context} "
        f"vocab={engine.config.vocab_size} "
        f"layers={engine.config.num_layers}"
    )

    print(f"\nprompt: {args.prompt!r}")
    result = engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        stop_on_eos=not args.no_stop_on_eos,
    )
    print(f"\ngenerated ({result.generated_tokens} tokens):")
    print(result.text)
    print(
        "\n---\n"
        f"prompt_tokens:    {result.prompt_tokens}\n"
        f"generated_tokens: {result.generated_tokens}\n"
        f"ttft_ms:          {result.ttft_ms:.1f}\n"
        f"decode_tok_s:     {result.decode_tok_s:.2f}\n"
        f"total_tok_s:      {result.total_tok_s:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
