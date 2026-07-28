from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vlm_eval.hardware import get_peak_vram_gb, reset_peak_memory_stats
from vlm_eval.inference.gemma import HuggingFaceVLM
from vlm_eval.video import sample_frames

from realtime_eval.core.metrics import RealtimeResult, naive_correct
from realtime_eval.core.power import PowerSampler

logger = logging.getLogger(__name__)


def load_model(model_id: str, hf_token: str | None = None) -> HuggingFaceVLM:
    """Load a VLM once for reuse across an entire config's repeats.

    Args:
        model_id: HuggingFace model ID.
        hf_token: Optional HuggingFace access token.

    Returns:
        A ready :class:`vlm_eval.inference.gemma.HuggingFaceVLM`.
    """
    return HuggingFaceVLM(model_id, hf_token=hf_token)


def _sample_cache(
    videos: list[tuple[Path, str]],
    num_frames: int,
) -> dict[Path, tuple[list[Any], float | None]]:
    """Decode and cache sampled frames once per video for this frame count.

    Sampling is kept out of the timed region so latency reflects model
    inference, not video I/O.

    Args:
        videos: ``(path, label)`` pairs to sample.
        num_frames: Frames to sample per video.

    Returns:
        Mapping of video path to ``(pil_frames, video_duration_sec)``. Videos
        that fail to decode are omitted.
    """
    cache: dict[Path, tuple[list[Any], float | None]] = {}
    for path, _label in videos:
        frames, duration_sec, _total, _fps = sample_frames(path, num_frames=num_frames)
        if frames is None:
            logger.warning("Skipping unreadable video: %s", path)
            continue
        cache[path] = (frames, duration_sec)
    return cache


def _timed_inference(
    model: HuggingFaceVLM,
    frames: list[Any],
    prompt: str,
    max_new_tokens: int,
    power_interval_sec: float,
) -> tuple[dict[str, Any], float | None, float | None, float | None]:
    """Run one inference, measuring power and peak VRAM around it.

    Args:
        model: Loaded VLM.
        frames: Sampled PIL frames.
        prompt: Instruction text.
        max_new_tokens: Generation cap.
        power_interval_sec: Background power sampling period.

    Returns:
        A 4-tuple ``(generated, mean_watts, peak_watts, peak_vram_gb)`` where
        ``generated`` is the raw dict from ``generate_from_frames``.
    """
    reset_peak_memory_stats()
    with PowerSampler(interval_sec=power_interval_sec) as sampler:
        generated = model.generate_from_frames(
            frames=frames,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
        )
    return generated, sampler.mean_watts, sampler.peak_watts, get_peak_vram_gb()


def run_config(
    model: HuggingFaceVLM,
    model_id: str,
    videos: list[tuple[Path, str]],
    num_frames: int,
    max_new_tokens: int,
    prompt: str,
    repeats: int,
    warmup: int,
    power_interval_sec: float = 0.1,
) -> list[RealtimeResult]:
    """Benchmark one ``(num_frames, max_new_tokens)`` config over a video set.

    Frames are sampled once per video, ``warmup`` discarded iterations run to
    absorb cold-start effects, then ``repeats`` timed iterations run per video.

    Args:
        model: A preloaded VLM (loaded once by the caller for all configs of
            the same model).
        model_id: Model ID recorded on each result.
        videos: ``(path, label)`` pairs to evaluate.
        num_frames: Frames sampled per inference (the primary latency lever).
        max_new_tokens: Generation cap matching the deployment target.
        prompt: Instruction text sent with the frames.
        repeats: Timed iterations per video for percentile estimation.
        warmup: Discarded iterations before timing.
        power_interval_sec: Background power sampling period.

    Returns:
        One :class:`RealtimeResult` per ``(video, repeat)``.
    """
    cache = _sample_cache(videos, num_frames)
    label_by_path = {path: label for path, label in videos}

    # Warmup on the first available video; results discarded.
    if warmup > 0 and cache:
        warm_path = next(iter(cache))
        warm_frames, _ = cache[warm_path]
        for _ in range(warmup):
            try:
                _timed_inference(
                    model, warm_frames, prompt, max_new_tokens, power_interval_sec
                )
            except Exception as exc:  # warmup failures are non-fatal
                logger.warning("Warmup iteration failed: %s", exc)

    results: list[RealtimeResult] = []
    for path, (frames, duration_sec) in cache.items():
        label = label_by_path.get(path, "unknown")
        for repeat_index in range(repeats):
            try:
                generated, mean_w, peak_w, peak_vram = _timed_inference(
                    model, frames, prompt, max_new_tokens, power_interval_sec
                )
            except Exception as exc:
                logger.error("Inference failed (%s, %d frames): %s", path.name, num_frames, exc)
                results.append(
                    RealtimeResult(
                        video=str(path),
                        label=label,
                        model_id=model_id,
                        num_frames=num_frames,
                        num_frames_actual=len(frames),
                        max_new_tokens=max_new_tokens,
                        repeat_index=repeat_index,
                        status="error",
                        error=str(exc),
                    )
                )
                continue

            latency_ms = generated["elapsed_ms"]
            ttft_ms = generated["ttft_ms"]
            # Backends that report no phase split (vLLM) still give the total.
            decode_ms = generated.get("decode_ms")
            if decode_ms is None and latency_ms is not None and ttft_ms is not None:
                decode_ms = latency_ms - ttft_ms
            rtf_inv = (
                (generated["elapsed_sec"] / duration_sec)
                if duration_sec and duration_sec > 0
                else None
            )
            response = generated["response"]
            results.append(
                RealtimeResult(
                    video=str(path),
                    label=label,
                    model_id=model_id,
                    num_frames=num_frames,
                    num_frames_actual=len(frames),
                    max_new_tokens=max_new_tokens,
                    repeat_index=repeat_index,
                    e2e_latency_ms=latency_ms,
                    preprocess_ms=generated.get("preprocess_ms"),
                    prefill_ms=generated.get("prefill_ms"),
                    ttft_ms=ttft_ms,
                    decode_ms=decode_ms,
                    tpot_ms=generated.get("tpot_ms"),
                    decode_tps=generated.get("decode_tps"),
                    throughput_tps=generated["throughput_tps"],
                    ttft_source=generated.get("ttft_source"),
                    rtf_inv=rtf_inv,
                    meets_realtime=(rtf_inv <= 1.0 if rtf_inv is not None else None),
                    video_duration_sec=duration_sec,
                    tokens=generated["tokens"],
                    mean_power_watts=mean_w,
                    peak_power_watts=peak_w,
                    peak_vram_gb=peak_vram,
                    response=response,
                    correct=naive_correct(response, label),
                )
            )
    return results
