from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_eval.hardware import get_peak_vram_reserved_gb, reset_peak_memory_stats
from vlm_eval.inference.gemma import HuggingFaceVLM
from vlm_eval.video import probe_video, sample_frames

from realtime_eval.core.metrics import RealtimeResult
from realtime_eval.core.power import DeviceSampler

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


@dataclass
class VideoWindow:
    """One benchmark sample: frames drawn from a single window of one clip.

    Attributes:
        path: Source clip.
        label: Clip label, carried through as provenance.
        index: Zero-based window position within the clip.
        start_sec: Window start offset in the clip.
        frames: Decoded PIL frames for this window.
        clip_duration_sec: Duration of the whole clip, not of the window.
    """

    path: Path
    label: str
    index: int
    start_sec: float
    frames: list[Any]
    clip_duration_sec: float | None


def build_window_cache(
    videos: list[tuple[Path, str]],
    num_frames: int,
    window_sec: float,
) -> list[VideoWindow]:
    """Tile each clip into windows and decode ``num_frames`` from each.

    The window is the unit of evaluation: a deployment runs one inference per
    ``window_sec`` of incoming video, so the frames fed to the model must span
    exactly that. Sampling across a whole clip instead would show the model a
    different temporal density than the one being judged.

    Decoding stays outside the timed region, so latency reflects inference and
    not video I/O.

    Args:
        videos: ``(path, label)`` pairs to sample.
        num_frames: Frames to sample per window.
        window_sec: Window length in seconds; windows do not overlap.

    Returns:
        One :class:`VideoWindow` per (clip, window), in clip then window order.

    Raises:
        ValueError: If any clip's frame rate cannot supply ``num_frames``
            distinct frames within one window. Raised before any inference
            runs, and lists every offending clip rather than only the first.
    """
    windows: list[VideoWindow] = []
    too_few: list[tuple[str, float, int]] = []

    for path, label in videos:
        total_frames, fps, duration_sec = probe_video(path)
        if not total_frames or not fps or not duration_sec:
            logger.warning("Skipping unreadable video: %s", path)
            continue

        # A window spans window_sec * fps frames; the floor is the fewest any
        # window can supply, since a non-integer count alternates (7.5 -> 8,7).
        capacity = math.floor(window_sec * fps)
        if num_frames > capacity:
            too_few.append((path.name, fps, capacity))
            continue

        n_windows = int(duration_sec // window_sec)
        if n_windows == 0:
            logger.warning(
                "Skipping %s: %.2fs is shorter than one %gs window",
                path.name,
                duration_sec,
                window_sec,
            )
            continue

        remainder = duration_sec - n_windows * window_sec
        if remainder > 1e-6:
            # Keeping a short tail window would mix a different frame density
            # into the same aggregate, so drop it.
            logger.info(
                "Dropping %.2fs partial tail window of %s", remainder, path.name
            )

        for index in range(n_windows):
            start_sec = index * window_sec
            frames, clip_duration_sec, _total, _fps = sample_frames(
                path,
                num_frames=num_frames,
                start_sec=start_sec,
                end_sec=start_sec + window_sec,
            )
            if frames is None:
                logger.warning("Skipping window %d of %s", index, path.name)
                continue
            windows.append(
                VideoWindow(
                    path=path,
                    label=label,
                    index=index,
                    start_sec=start_sec,
                    frames=frames,
                    clip_duration_sec=clip_duration_sec,
                )
            )

    if too_few:
        detail = "; ".join(
            f"{name} ({fps:.1f} fps supplies at most {cap} frames)"
            for name, fps, cap in too_few
        )
        needed = max(num_frames / fps for _name, fps, _cap in too_few)
        raise ValueError(
            f"num_frames={num_frames} exceeds what a {window_sec:g}s window holds "
            f"for {len(too_few)} clip(s): {detail}. Either reduce num_frames, or "
            f"raise window_sec to >= {needed:.4f}s."
        )
    return windows


def _timed_inference(
    model: HuggingFaceVLM,
    frames: list[Any],
    prompt: str,
    max_new_tokens: int,
    power_interval_sec: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one inference, measuring power and VRAM around it.

    Args:
        model: Loaded VLM.
        frames: Sampled PIL frames.
        prompt: Instruction text.
        max_new_tokens: Generation cap.
        power_interval_sec: Background power sampling period.

    Returns:
        ``(generated, device)`` where ``generated`` is the raw dict from
        ``generate_from_frames`` and ``device`` holds the power, energy and VRAM
        readings for this run.
    """
    reset_peak_memory_stats()
    with DeviceSampler(interval_sec=power_interval_sec) as sampler:
        generated = model.generate_from_frames(
            frames=frames,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
        )
    device = {
        "mean_power_watts": sampler.mean_watts,
        "peak_power_watts": sampler.peak_watts,
        "energy_j": sampler.energy_j,
        "power_sampled_sec": sampler.sampled_sec,
        "n_power_samples": sampler.n_power_samples,
        "peak_vram_reserved_gb": get_peak_vram_reserved_gb(),
        "peak_vram_device_gb": sampler.peak_device_vram_gb,
    }
    return generated, device


def run_config(
    model: HuggingFaceVLM,
    model_id: str,
    videos: list[tuple[Path, str]],
    num_frames: int,
    max_new_tokens: int,
    prompt: str,
    repeats: int,
    warmup: int,
    window_sec: float = 1.0,
    power_interval_sec: float = 0.1,
    cache: list[VideoWindow] | None = None,
) -> list[RealtimeResult]:
    """Benchmark one ``(num_frames, max_new_tokens)`` config over a video set.

    Each clip is tiled into ``window_sec`` windows and every window is one
    sample, so ``num_frames`` always span exactly one window. ``warmup``
    discarded iterations absorb cold-start effects, then ``repeats`` timed
    iterations run per window.

    Repeats and windows measure different things: repeats re-run identical
    content, so with greedy decoding they capture machine jitter only, while
    distinct windows vary the content and so the generated token count. The
    latency spread across windows is the one a deployment actually sees.

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
        window_sec: Window length, which is both the span the frames are drawn
            from and the stride the real-time verdict is judged against.
        power_interval_sec: Background power sampling period.
        cache: Pre-decoded windows from :func:`build_window_cache` for this
            frame count and window length. Pass it to avoid re-decoding the
            video set once per ``max_new_tokens`` value; decoded when omitted.

    Returns:
        One :class:`RealtimeResult` per ``(window, repeat)``.
    """
    if cache is None:
        cache = build_window_cache(videos, num_frames, window_sec)

    # Warmup on the first available window; results discarded.
    if warmup > 0 and cache:
        for _ in range(warmup):
            try:
                _timed_inference(
                    model, cache[0].frames, prompt, max_new_tokens, power_interval_sec
                )
            except Exception as exc:  # warmup failures are non-fatal
                logger.warning("Warmup iteration failed: %s", exc)

    results: list[RealtimeResult] = []
    for window in cache:
        path, label, frames = window.path, window.label, window.frames
        duration_sec = window.clip_duration_sec
        for repeat_index in range(repeats):
            try:
                generated, device = _timed_inference(
                    model, frames, prompt, max_new_tokens, power_interval_sec
                )
            except Exception as exc:
                logger.error(
                    "Inference failed (%s window %d, %d frames): %s",
                    path.name,
                    window.index,
                    num_frames,
                    exc,
                )
                results.append(
                    RealtimeResult(
                        video=str(path),
                        label=label,
                        model_id=model_id,
                        num_frames=num_frames,
                        num_frames_actual=len(frames),
                        max_new_tokens=max_new_tokens,
                        repeat_index=repeat_index,
                        window_index=window.index,
                        window_start_sec=window.start_sec,
                        window_sec=window_sec,
                        status="error",
                        error=str(exc),
                    )
                )
                continue

            latency_ms = generated["elapsed_ms"]
            ttft_ms = generated["ttft_ms"]
            tokens = generated["tokens"]
            # Backends that report no phase split (vLLM) still give the total,
            # so derive the decode metrics here rather than in each backend.
            decode_ms = generated.get("decode_ms")
            if decode_ms is None and latency_ms is not None and ttft_ms is not None:
                decode_ms = latency_ms - ttft_ms
            gaps = (tokens - 1) if tokens else 0
            tpot_ms = generated.get("tpot_ms")
            if tpot_ms is None and decode_ms is not None and gaps > 0:
                tpot_ms = decode_ms / gaps
            decode_tps = generated.get("decode_tps")
            if decode_tps is None and decode_ms is not None and decode_ms > 0 and gaps > 0:
                decode_tps = gaps / (decode_ms / 1000.0)
            elapsed_sec = generated["elapsed_sec"]
            rtf = (
                (elapsed_sec / duration_sec) if duration_sec and duration_sec > 0 else None
            )
            window_rtf = (elapsed_sec / window_sec) if window_sec > 0 else None
            sustainable_fps = (len(frames) / elapsed_sec) if elapsed_sec > 0 else None
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
                    tpot_ms=tpot_ms,
                    decode_tps=decode_tps,
                    throughput_tps=generated["throughput_tps"],
                    ttft_source=generated.get("ttft_source"),
                    window_index=window.index,
                    window_start_sec=window.start_sec,
                    window_sec=window_sec,
                    window_rtf=window_rtf,
                    max_sustainable_fps=sustainable_fps,
                    rtf=rtf,
                    video_duration_sec=duration_sec,
                    tokens=tokens,
                    response=response,
                    **device,
                )
            )
    return results
