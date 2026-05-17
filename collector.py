"""KV cache metrics collector for vLLM multi-turn chat sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from vllm import LLM, SamplingParams

_GiB = 1024 ** 3

# Bytes per element for each supported KV cache dtype.
_KV_DTYPE_BYTES: dict[str, int] = {
    "auto": 0,      # resolved at runtime from the model dtype
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}

# Map torch dtype strings to byte widths.
_TORCH_DTYPE_BYTES: dict[str, int] = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float8_e4m3fn": 1,
}


@dataclass
class MemoryBreakdown:
    """GPU memory breakdown captured once after the engine initialises."""
    gpu_total_gb: float
    gpu_free_gb: float           # free after full init
    vllm_footprint_gb: float     # total taken by vLLM (free_before - free_after)
    kv_cache_gb: float           # bytes reserved for the KV block pool
    kv_cache_tokens: int         # total token capacity of the pool
    kv_block_size: int           # tokens per block (CacheConfig.block_size)
    other_gb: float              # model weights + activations + CUDA graph + NCCL


@dataclass
class TurnMetrics:
    turn: int
    role: str  # "user" or "assistant"
    prompt_tokens: int
    generated_tokens: int
    cached_tokens: int          # prefix-cache hits
    computed_tokens: int        # tokens that required actual compute
    cache_hit_rate: float       # cached / prompt_tokens
    ttft_s: float               # time-to-first-token (engine internal)
    decode_time_s: float        # engine decode phase duration
    e2e_latency_s: float        # wall-clock from send to receive
    kv_cache_utilization: float # peak block-pool fraction used this turn (0–1)
    kv_cache_tokens_used: int   # absolute tokens held in the pool at peak
    kv_cache_total_tokens: int  # total pool capacity in tokens
    gpu_mem_used_gb: float      # CUDA driver: all GPU memory in use
    gpu_mem_total_gb: float
    gpu_mem_utilization: float  # gpu_mem_used / total


@dataclass
class SessionMetrics:
    model: str
    turns: list[TurnMetrics] = field(default_factory=list)

    def add(self, m: TurnMetrics) -> None:
        self.turns.append(m)

    def summary(self) -> dict:
        if not self.turns:
            return {}
        total_prompt = sum(t.prompt_tokens for t in self.turns)
        total_cached = sum(t.cached_tokens for t in self.turns)
        return {
            "total_turns": len(self.turns),
            "total_prompt_tokens": total_prompt,
            "total_generated_tokens": sum(t.generated_tokens for t in self.turns),
            "total_cached_tokens": total_cached,
            "session_cache_hit_rate": total_cached / total_prompt if total_prompt else 0.0,
            "mean_e2e_latency_s": sum(t.e2e_latency_s for t in self.turns) / len(self.turns),
        }


class KVCacheMetricsCollector:
    """Wraps vLLM LLM to capture per-turn KV cache and latency metrics."""

    def __init__(
        self,
        model: str,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        num_gpu_blocks_override: int | None = None,
        **llm_kwargs,
    ) -> None:
        self.model = model

        free_before, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)

        self._llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            num_gpu_blocks_override=num_gpu_blocks_override,
            enable_prefix_caching=True,
            disable_log_stats=False,  # populate RequestOutput.metrics
            **llm_kwargs,
        )

        free_after, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)

        self._kv_tracker = _patch_kv_usage_capture(self._llm)
        cc = self._llm.llm_engine.vllm_config.cache_config
        self._kv_total_tokens: int = (cc.num_gpu_blocks or 0) * cc.block_size
        self.memory = _compute_memory_breakdown(
            self._llm, free_before, free_after, gpu_total
        )

        self._history: list[dict] = []
        self.session = SessionMetrics(model=model)
        self._turn = 0

    def chat(
        self,
        user_message: str,
        sampling_params: SamplingParams | None = None,
    ) -> tuple[str, TurnMetrics]:
        """Send a user message, get a reply, and capture metrics for this turn."""
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

        self._history.append({"role": "user", "content": user_message})
        self._turn += 1

        self._kv_tracker.reset()
        t0 = time.perf_counter()
        outputs = self._llm.chat(
            messages=self._history,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        e2e = time.perf_counter() - t0

        result = outputs[0]
        reply = result.outputs[0].text.strip()
        self._history.append({"role": "assistant", "content": reply})

        prompt_tokens = len(result.prompt_token_ids) if result.prompt_token_ids else 0
        generated_tokens = len(result.outputs[0].token_ids)
        cached_tokens = result.num_cached_tokens or 0
        computed_tokens = prompt_tokens - cached_tokens

        # Timing from engine internals (available when disable_log_stats=False)
        stats = result.metrics
        if stats is not None:
            ttft = stats.first_token_latency
            # last_token_ts and first_token_ts are monotonic engine timestamps
            decode_time = max(0.0, stats.last_token_ts - stats.first_token_ts)
        else:
            ttft = 0.0
            decode_time = 0.0

        kv_util = self._kv_tracker.peak()
        kv_tokens_used = round(kv_util * self._kv_total_tokens) if kv_util == kv_util else 0
        gpu_used, gpu_total = _gpu_memory_gb()

        metrics = TurnMetrics(
            turn=self._turn,
            role="assistant",
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            cached_tokens=cached_tokens,
            computed_tokens=computed_tokens,
            cache_hit_rate=cached_tokens / prompt_tokens if prompt_tokens else 0.0,
            ttft_s=ttft,
            decode_time_s=decode_time,
            e2e_latency_s=e2e,
            kv_cache_utilization=kv_util,
            kv_cache_tokens_used=kv_tokens_used,
            kv_cache_total_tokens=self._kv_total_tokens,
            gpu_mem_used_gb=gpu_used,
            gpu_mem_total_gb=gpu_total,
            gpu_mem_utilization=gpu_used / gpu_total if gpu_total else 0.0,
        )
        self.session.add(metrics)
        return reply, metrics

    def reset_history(self) -> None:
        """Start a new conversation while keeping the same model loaded."""
        self._history.clear()
        self._turn = 0
        self.session = SessionMetrics(model=self.model)


class _KVUsageTracker:
    """Captures peak kv_cache_usage across all engine steps for one turn.

    kv_cache_usage is a live snapshot of blocks held by running requests.
    It drops to 0 once the request finishes and blocks are released, so
    reading it after llm.chat() returns always yields 0. We track the
    maximum seen during each turn instead.
    """

    def __init__(self) -> None:
        self._peak: float = float("nan")

    def reset(self) -> None:
        self._peak = float("nan")

    def observe(self, value: float) -> None:
        if self._peak != self._peak:  # nan check
            self._peak = value
        else:
            self._peak = max(self._peak, value)

    def peak(self) -> float:
        return self._peak


def _patch_kv_usage_capture(llm: LLM) -> _KVUsageTracker:
    """Wrap StatLoggerManager.record() to track peak kv_cache_usage per turn.

    The engine core runs in a separate process (SyncMPClient), so we can't
    read the scheduler's block pool directly. Instead we wrap the logger
    manager's record() call, which receives scheduler_stats from the engine
    on every step — the same data source vLLM's own log line uses.
    """
    lm = llm.llm_engine.logger_manager
    tracker = _KVUsageTracker()

    if lm is None:
        return tracker

    orig_record = lm.record

    def _record(
        scheduler_stats=None,
        iteration_stats=None,
        mm_cache_stats=None,
        engine_idx=None,
    ):
        if scheduler_stats is not None:
            tracker.observe(scheduler_stats.kv_cache_usage)
        return orig_record(
            scheduler_stats=scheduler_stats,
            iteration_stats=iteration_stats,
            mm_cache_stats=mm_cache_stats,
            **({"engine_idx": engine_idx} if engine_idx is not None else {}),
        )

    lm.record = _record
    return tracker


def _compute_memory_breakdown(
    llm: LLM,
    free_before: int,
    free_after: int,
    gpu_total: int,
) -> MemoryBreakdown:
    cc = llm.llm_engine.vllm_config.cache_config
    mc = llm.llm_engine.vllm_config.model_config

    kv_cache_tokens = (cc.num_gpu_blocks or 0) * cc.block_size

    # Resolve bytes per KV element.
    dtype_key = cc.cache_dtype if cc.cache_dtype != "auto" else str(mc.hf_config.torch_dtype)
    elem_bytes = _KV_DTYPE_BYTES.get(cc.cache_dtype) or _TORCH_DTYPE_BYTES.get(dtype_key, 2)

    hf = mc.hf_config
    num_layers = hf.num_hidden_layers
    num_kv_heads = hf.num_key_value_heads
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    # Factor of 2: one tensor for K, one for V.
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * elem_bytes

    kv_cache_bytes = kv_cache_tokens * bytes_per_token
    vllm_footprint = free_before - free_after

    return MemoryBreakdown(
        gpu_total_gb=gpu_total / _GiB,
        gpu_free_gb=free_after / _GiB,
        vllm_footprint_gb=vllm_footprint / _GiB,
        kv_cache_gb=kv_cache_bytes / _GiB,
        kv_cache_tokens=kv_cache_tokens,
        kv_block_size=cc.block_size,
        other_gb=(vllm_footprint - kv_cache_bytes) / _GiB,
    )


def _gpu_memory_gb() -> tuple[float, float]:
    """Return (used_gb, total_gb) from the CUDA driver.

    torch.cuda.mem_get_info() queries the driver directly and captures all
    GPU memory in use, including vLLM's pre-allocated KV cache pool that
    torch.cuda.memory_allocated() misses.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    return used / 1024**3, total / 1024**3
