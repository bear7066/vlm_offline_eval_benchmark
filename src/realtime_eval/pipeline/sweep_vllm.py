"""vLLM backend for the real-time sweep.

Requires the optional ``vllm`` dependency group (``uv sync --extra vllm``);
importing this module is safe without it — vLLM is only imported when a
model is actually loaded.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from realtime_eval.core.config import SweepConfig
from realtime_eval.pipeline.sweep import run_sweep

logger = logging.getLogger(__name__)


def _mem_available_bytes() -> int | None:
    """Return Linux ``MemAvailable`` (free + reclaimable cache), or ``None``."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _default_gpu_memory_utilization() -> float:
    """Size vLLM's memory claim from memory that is *actually* obtainable.

    vLLM validates ``gpu_memory_utilization * total <= cudaMemGetInfo free``
    at startup, which breaks twice on unified-memory machines (GB10/Spark):
    the OS and other processes share the GPU pool (so vLLM's ~0.9-of-total
    default can never fit), and ``cudaMemGetInfo`` excludes reclaimable page
    cache (so after reading big checkpoints "free" can drop to ~2 GiB even
    though the kernel would reclaim tens of GiB for a real allocation).

    When Linux reports more available than CUDA reports free, touch a CUDA
    allocation of that size to force the reclaim, then measure again. Claim
    90% of the resulting free memory, capped at vLLM's usual 0.9.

    Returns:
        A ``gpu_memory_utilization`` fraction safe for the current machine
        state.
    """
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    available = _mem_available_bytes()
    if available is not None and available > free_bytes:
        try:
            block = torch.empty(int(available * 0.9), dtype=torch.uint8, device="cuda")
            block.fill_(0)  # touch every page so the kernel really reclaims
            del block
        except RuntimeError:
            pass  # can't reclaim that much (e.g. discrete GPU); use what's free
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return min(0.9, 0.9 * free_bytes / total_bytes)


class VLLMModel:
    """A vLLM-backed VLM with the same contract as ``HuggingFaceVLM``.

    Exposes ``generate_from_frames`` returning the same dict keys as
    :class:`vlm_eval.inference.gemma.HuggingFaceVLM`, so it can be passed to
    :func:`realtime_eval.pipeline.runner.run_config` unchanged. The prompt is
    formatted with the model's own HuggingFace chat template so both backends
    see identical inputs.

    Args:
        model_id: HuggingFace model ID to load.
        hf_token: Optional HuggingFace access token.
        max_images: Upper bound on images per prompt, i.e. the largest frame
            count the sweep will request (vLLM defaults to 1 otherwise).
        gpu_memory_utilization: Fraction of total GPU memory vLLM may claim.
            Defaults to 90% of the memory currently free — vLLM's own default
            (~0.9 of total) fails on unified-memory machines (e.g. GB10/Spark)
            where the OS shares the same pool.

    Raises:
        ImportError: If vLLM is not installed.
    """

    def __init__(
        self,
        model_id: str,
        hf_token: str | None = None,
        max_images: int = 32,
        gpu_memory_utilization: float | None = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Install the optional backend with "
                "`uv sync --extra vllm`."
            ) from exc
        from transformers import AutoProcessor

        if gpu_memory_utilization is None:
            gpu_memory_utilization = _default_gpu_memory_utilization()
            logger.info("vLLM gpu_memory_utilization: %.2f", gpu_memory_utilization)

        self.model_id = model_id
        self._sampling_params_cls = SamplingParams
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": max_images},
            gpu_memory_utilization=gpu_memory_utilization,
            # V1 only attaches per-request stats (our TTFT source) to
            # RequestOutput.metrics when stats logging is on; the offline LLM
            # API disables it by default.
            disable_log_stats=False,
        )

    def generate_from_frames(
        self,
        frames: list[Any],
        prompt_text: str,
        max_new_tokens: int = 150,
    ) -> dict[str, Any]:
        """Run one greedy generation over sampled frames and time it.

        Mirrors ``HuggingFaceVLM.generate_from_frames``: one image placeholder
        per frame plus the prompt text, greedy decoding capped at
        ``max_new_tokens``.

        Args:
            frames: Sampled PIL frames.
            prompt_text: Instruction text sent with the frames.
            max_new_tokens: Generation cap.

        Returns:
            Dict with the keys ``response``, ``elapsed_sec``, ``elapsed_ms``,
            ``ttft_ms``, ``tokens``, ``throughput_tps``, ``ttft_source`` and
            ``average_power_watts``. ``ttft_ms`` is taken from vLLM's request
            metrics (populated because the engine is built with
            ``disable_log_stats=False``), else ``None``.

            ``preprocess_ms`` and ``prefill_ms`` are **not** reported: the
            engine preprocesses images inside ``generate`` and its TTFT is
            measured from request arrival, so the phase split that the
            transformers backend guarantees does not hold here. ``ttft_source``
            records the difference; do not compare TTFT across backends without
            checking it.
        """
        # Timed from chat templating onward, to cover the same span as the
        # transformers backend (which includes preprocessing).
        start_time = time.perf_counter()

        content_items: list[dict[str, Any]] = [{"type": "image"} for _ in range(len(frames))]
        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampling = self._sampling_params_cls(temperature=0.0, max_tokens=max_new_tokens)

        outputs = self.llm.generate(
            {"prompt": formatted_prompt, "multi_modal_data": {"image": list(frames)}},
            sampling_params=sampling,
            use_tqdm=False,
        )
        elapsed_sec = time.perf_counter() - start_time

        completion = outputs[0].outputs[0]
        response = completion.text.strip()
        num_tokens = len(completion.token_ids)

        # TTFT from engine-reported request metrics: V1 RequestStateStats
        # exposes first_token_latency (seconds); V0 RequestMetrics exposed
        # first_token_time/arrival_time. None if the engine reports neither.
        ttft_ms = None
        metrics = getattr(outputs[0], "metrics", None)
        first_token_latency = getattr(metrics, "first_token_latency", None)
        first_token_time = getattr(metrics, "first_token_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)
        if first_token_latency:
            ttft_ms = first_token_latency * 1000.0
        elif first_token_time is not None and arrival_time is not None:
            ttft_ms = (first_token_time - arrival_time) * 1000.0

        return {
            "response": response,
            "elapsed_sec": elapsed_sec,
            "elapsed_ms": elapsed_sec * 1000.0,
            "ttft_ms": ttft_ms,
            "tokens": num_tokens,
            "throughput_tps": num_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
            "ttft_source": "engine-metrics" if ttft_ms is not None else None,
            "average_power_watts": None,
        }


def run_sweep_vllm(
    videos_root: Path,
    config: SweepConfig,
    video_limit: int | None = None,
) -> Path:
    """Run the real-time sweep on the vLLM backend.

    Identical to :func:`realtime_eval.pipeline.sweep.run_sweep` — same run-dir
    layout, ``results.jsonl`` and ``summary.json`` formats — except models are
    served by vLLM and the run's ``config.json`` records ``"backend": "vllm"``.

    Args:
        videos_root: Directory of labeled videos (or a single video file).
        config: Sweep grid and timing parameters.
        video_limit: Optional cap on number of videos used.

    Returns:
        Path to the created run directory.
    """
    max_images = max(config.num_frames_grid)

    def loader(model_id: str, hf_token: str | None) -> VLLMModel:
        return VLLMModel(model_id, hf_token=hf_token, max_images=max_images)

    return run_sweep(
        videos_root,
        config,
        video_limit=video_limit,
        backend="vllm",
        model_loader=loader,
    )
