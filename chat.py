#!/usr/bin/env python3
"""Interactive multi-turn chat that prints KV cache metrics after each reply."""

from __future__ import annotations

import argparse
import json
import sys

from collector import KVCacheMetricsCollector, TurnMetrics


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
    parser = argparse.ArgumentParser(description="KV-cache metrics chat demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--dump-json",
        metavar="FILE",
        help="Write session metrics JSON to this file on exit",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    collector = KVCacheMetricsCollector(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("Ready. Type your message (Ctrl-C or 'quit' to exit).\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in {"quit", "exit"}:
                break

            reply, metrics = collector.chat(user_input)
            print(f"\nAssistant: {reply}")
            print(_format_metrics(metrics))
            print()

    except KeyboardInterrupt:
        pass

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
            "turns": [
                {k: v for k, v in vars(t).items()}
                for t in collector.session.turns
            ],
        }
        with open(args.dump_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nMetrics written to {args.dump_json}")


if __name__ == "__main__":
    main()
