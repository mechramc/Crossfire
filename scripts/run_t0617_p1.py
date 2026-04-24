#!/usr/bin/env python3
"""T-0617 P1 distributed baseline runner.

Issues streaming chat completions against an EXO cluster at three context
sizes (default 8K / 16K / 32K) and records TTFT + decode tok/s per run.

Assumes the model instance is already placed. Source text for prompts is
`datasets/wiki.test.raw` (committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
WIKI = REPO / "datasets" / "wiki.test.raw"

DEFAULT_TARGETS = [8192, 16384, 32768]
DEFAULT_DECODE_TOKENS = 64
DEFAULT_MODEL = "mlx-community/gemma-4-31b-it-4bit"
DEFAULT_ENDPOINT = "http://localhost:52415/v1/chat/completions"


def build_prompt_text(target_tokens: int, chars_per_token: float = 4.2) -> str:
    """Pull ~target_tokens of English prose from wiki.test.raw.

    4.2 chars/token is an empirical average for Gemma tokenizers on English
    Wikipedia text; the real count is asserted post-request against the
    `usage.prompt_tokens` field of the response.
    """
    if not WIKI.exists():
        raise FileNotFoundError(f"Wikitext corpus not found: {WIKI}")
    # 15% cushion; server asserts real token count via usage.prompt_tokens
    target_chars = int(target_tokens * chars_per_token * 1.15)
    text = WIKI.read_text(encoding="utf-8", errors="ignore")[:target_chars]
    return text


def stream_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    timeout_s: int = 900,
) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "seed": 1,
    }
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_send = time.perf_counter()
    t_first = None
    t_end = None
    chunks = 0
    completion_text: list[str] = []
    usage_fields: dict[str, object] = {}

    with urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if t_first is None:
                t_first = time.perf_counter()
            chunks += 1
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    completion_text.append(content)
            usage = obj.get("usage")
            if usage:
                usage_fields = usage
    t_end = time.perf_counter()

    ttft_ms = (t_first - t_send) * 1000 if t_first else None
    decode_s = (t_end - t_first) if t_first else None
    completion_tokens = usage_fields.get("completion_tokens") or 0
    prompt_tokens = usage_fields.get("prompt_tokens") or 0
    decode_tok_s = (completion_tokens / decode_s) if (decode_s and completion_tokens) else None

    return {
        "ttft_ms": ttft_ms,
        "decode_sec": decode_s,
        "total_sec": t_end - t_send,
        "chunks": chunks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tok_s": decode_tok_s,
        "text_tail": "".join(completion_text)[-200:] if completion_text else "",
        "usage": usage_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=DEFAULT_TARGETS,
        help="target prompt-token sizes to sweep",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "t0617_p1_distributed_baseline.json",
    )
    args = parser.parse_args()

    runs = []
    for target in args.targets:
        prompt = build_prompt_text(target)
        print(f"[{target} target] prompt_chars={len(prompt)} max_tokens={args.max_tokens}")
        try:
            result = stream_completion(
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            runs.append({"target_tokens": target, "error": str(exc)})
            continue
        result["target_tokens"] = target
        runs.append(result)
        ttft = result["ttft_ms"]
        tok_s = result["decode_tok_s"]
        pt = result["prompt_tokens"]
        ct = result["completion_tokens"]
        print(
            f"  TTFT={ttft:.1f}ms  decode={tok_s:.2f}tok/s  prompt_toks={pt} completion_toks={ct}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": "T-0617",
        "policy": "P1",
        "endpoint": args.endpoint,
        "model": args.model,
        "max_tokens_per_run": args.max_tokens,
        "runs": runs,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
