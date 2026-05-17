#!/usr/bin/env python3
"""Run a multi-turn chat from a file of inputs, printing KV cache metrics after each reply."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# FlashInfer sampler requires CUDA 12+ CCCL headers; disable it so the script
# works on CUDA 11.x systems. Must be set before vLLM is imported so the
# EngineCore subprocess inherits the value.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from collector import KVCacheMetricsCollector, TurnMetrics

_SUPPORTED_TYPES = {"P"}


def _parse_inputs(path: Path) -> list[tuple[str, str]]:
    """Return a list of (type, content) pairs from the input file.

    Each non-blank, non-comment line must start with a known type identifier
    followed by a space and the content:
        P Tell me about prefix caching.
    """
    inputs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) < 3 or line[1] != " ":
            sys.exit(
                f"{path}:{lineno}: expected '<TYPE> <content>', got: {raw!r}"
            )
        kind, content = line[0].upper(), line[2:].strip()
        if kind not in _SUPPORTED_TYPES:
            sys.exit(
                f"{path}:{lineno}: unknown input type {kind!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_TYPES))}"
            )
        if not content:
            sys.exit(f"{path}:{lineno}: empty content after type identifier")
        inputs.append((kind, content))
    return inputs


def _format_metrics(m: TurnMetrics) -> str:
    hit_pct = m.cache_hit_rate * 100
    kv_pct = m.kv_cache_utilization * 100
    gpu_pct = m.gpu_mem_utilization * 100
    return (
        f"\n{'─' * 60}\n"
        f"  [Turn {m.turn} metrics]\n"
        f"  Tokens  : {m.prompt_tokens} prompt  "
        f"({m.cached_tokens} cached / {m.computed_tokens} computed)  "
        f"| {m.generated_tokens} generated\n"
        f"  Cache   : {hit_pct:.1f}% prefix hit rate\n"
        f"  KV pool : {m.kv_cache_tokens_used:,} / {m.kv_cache_total_tokens:,} tokens "
        f"({kv_pct:.4f}% peak)\n"
        f"  Latency : e2e {m.e2e_latency_s:.2f}s  "
        f"| TTFT {m.ttft_s:.3f}s  "
        f"| decode {m.decode_time_s:.3f}s\n"
        f"  GPU mem : {m.gpu_mem_used_gb:.2f} GB / {m.gpu_mem_total_gb:.2f} GB  "
        f"({gpu_pct:.1f}%)\n"
        f"{'─' * 60}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a scripted multi-turn chat and report KV cache metrics."
    )
    parser.add_argument(
        "inputs",
        metavar="INPUTS_FILE",
        type=Path,
        help=(
            "Text file of inputs. Each non-blank line: '<TYPE> <content>'. "
            "Supported types: P (paragraph text). Lines starting with # are comments."
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--dump-json",
        metavar="FILE",
        help="Write session metrics JSON to this file on exit",
    )
    args = parser.parse_args()

    if not args.inputs.is_file():
        sys.exit(f"Input file not found: {args.inputs}")

    turns = _parse_inputs(args.inputs)
    if not turns:
        sys.exit(f"No inputs found in {args.inputs}")

    print(f"Loading model: {args.model}")
    collector = KVCacheMetricsCollector(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"Running {len(turns)} turn(s) from {args.inputs}\n")

    for kind, content in turns:
        # P: send content as a user paragraph
        print(f"You: {content}")
        reply, metrics = collector.chat(content)
        print(f"\nAssistant: {reply}")
        print(_format_metrics(metrics))
        print()

    summary = collector.session.summary()
    print("\n=== Session Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.dump_json:
        data = {
            "model": collector.model,
            "summary": summary,
            "turns": [vars(t) for t in collector.session.turns],
        }
        with open(args.dump_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nMetrics written to {args.dump_json}")


if __name__ == "__main__":
    main()
