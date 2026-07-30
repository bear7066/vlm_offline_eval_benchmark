from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from vlm_eval.hardware import get_hardware_name
from vlm_eval.paths import model_name_from_id, slugify

from realtime_eval.core.config import SweepConfig
from realtime_eval.core.dataset import discover_videos
from realtime_eval.core.metrics import SCHEMA_VERSION, RealtimeResult, aggregate
from realtime_eval.pipeline.runner import build_window_cache, load_model, run_config

logger = logging.getLogger(__name__)


def run_single(
    video: Path,
    model_id: str,
    num_frames: int,
    max_new_tokens: int,
    prompt: str,
    repeats: int = 1,
    warmup: int = 1,
    label: str | None = None,
    window_sec: float = 1.0,
    power_interval_sec: float = 0.1,
) -> list[RealtimeResult]:
    """Run one model on one video for quick verification.

    Loads the model, tiles the clip into ``window_sec`` windows, samples
    ``num_frames`` from each, runs ``warmup`` discarded iterations, then
    ``repeats`` timed iterations per window. Nothing is written to disk;
    results are returned for the caller to display.

    Args:
        video: Path to a single video file.
        model_id: HuggingFace model ID to run.
        num_frames: Frames sampled and fed to the model.
        max_new_tokens: Generation cap.
        prompt: Instruction text sent with the frames.
        repeats: Timed iterations (1 is enough for a smoke test).
        warmup: Discarded iterations before timing.
        label: Ground-truth label override; defaults to the file's parent
            directory name (the project convention).
        window_sec: Window length; frames span exactly this much video, and it
            is the denominator of ``window_rtf``.
        power_interval_sec: Background power sampling period.

    Returns:
        One :class:`RealtimeResult` per ``(window, repeat)``.
    """
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_token = os.environ.get("HF_TOKEN")

    video = Path(video)
    resolved_label = label if label is not None else (video.parent.name or "unknown")

    logger.info("Hardware: %s", get_hardware_name())
    logger.info("Loading model: %s", model_id)
    model = load_model(model_id, hf_token=hf_token)

    logger.info(
        "Running %s on %s | frames=%d | max_new_tokens=%d",
        model_name_from_id(model_id),
        video.name,
        num_frames,
        max_new_tokens,
    )
    return run_config(
        model=model,
        model_id=model_id,
        videos=[(video, resolved_label)],
        num_frames=num_frames,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
        repeats=repeats,
        warmup=warmup,
        window_sec=window_sec,
        power_interval_sec=power_interval_sec,
    )


def _append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
        f.write("\n")


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def run_sweep(
    videos_root: Path,
    config: SweepConfig,
    video_limit: int | None = None,
    backend: str = "python-transformers",
    model_loader: Callable[[str, str | None], Any] | None = None,
) -> Path:
    """Run the full real-time sweep and write results to a timestamped run dir.

    Loads each model once, then iterates its ``(num_frames, max_new_tokens)``
    configs over the discovered video set, streaming every per-run result to
    ``results.jsonl`` and writing an aggregated ``summary.json`` at the end.

    Args:
        videos_root: Directory of labeled videos (or a single video file).
        config: Sweep grid and timing parameters.
        video_limit: Optional cap on number of videos used.
        backend: Backend name recorded in the run's ``config.json``.
        model_loader: ``(model_id, hf_token) -> model`` factory returning any
            object with a ``generate_from_frames`` method compatible with
            :class:`vlm_eval.inference.gemma.HuggingFaceVLM`. Defaults to the
            transformers-based :func:`realtime_eval.pipeline.runner.load_model`.

    Returns:
        Path to the created run directory.
    """
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_token = os.environ.get("HF_TOKEN")
    if model_loader is None:
        model_loader = load_model

    videos = discover_videos(Path(videos_root), limit=video_limit)
    hardware_name = get_hardware_name()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_root) / slugify(f"sweep_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "videos_root": str(videos_root),
            "num_videos": len(videos),
            "hardware_name": hardware_name,
            "backend": backend,
            "model_ids": list(config.model_ids),
            "num_frames_grid": list(config.num_frames_grid),
            "max_new_tokens_grid": list(config.max_new_tokens_grid),
            "repeats": config.repeats,
            "warmup": config.warmup,
            "window_sec": config.window_sec,
            "realtime_threshold": config.realtime_threshold,
            "min_success_rate": config.min_success_rate,
            "prompt": config.prompt,
        },
    )

    results_path = run_dir / "results.jsonl"
    # Create it up front so the run dir is well-formed even when every model
    # fails to load; otherwise analyze() reports a missing file rather than the
    # real problem.
    results_path.touch()
    logger.info("Sweep run dir: %s", run_dir)
    logger.info("Hardware: %s | videos: %d", hardware_name, len(videos))

    all_results: list[RealtimeResult] = []
    # Group configs by model so each model loads exactly once.
    for model_id in config.model_ids:
        logger.info("Loading model: %s", model_id)
        try:
            model = model_loader(model_id, hf_token)
        except Exception as exc:
            logger.error("Failed to load %s, skipping: %s", model_id, exc)
            continue

        for num_frames in config.num_frames_grid:
            # Decode once per frame count, not once per (frames, tokens) pair.
            frame_cache = build_window_cache(videos, num_frames, config.window_sec)
            logger.info(
                "Sampled %d window(s) of %gs at %d frames (%.1f fps density)",
                len(frame_cache),
                config.window_sec,
                num_frames,
                num_frames / config.window_sec,
            )
            for max_new_tokens in config.max_new_tokens_grid:
                logger.info(
                    "Config: %s | frames=%d | max_new_tokens=%d",
                    model_name_from_id(model_id),
                    num_frames,
                    max_new_tokens,
                )
                results = run_config(
                    model=model,
                    model_id=model_id,
                    videos=videos,
                    num_frames=num_frames,
                    max_new_tokens=max_new_tokens,
                    prompt=config.prompt,
                    repeats=config.repeats,
                    warmup=config.warmup,
                    window_sec=config.window_sec,
                    power_interval_sec=config.power_sample_interval_sec,
                    cache=frame_cache,
                )
                for result in results:
                    _append_jsonl(results_path, result.to_dict())
                all_results.extend(results)

        # Free GPU memory before the next model.
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summaries = aggregate(
        all_results,
        threshold=config.realtime_threshold,
        min_success_rate=config.min_success_rate,
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hardware_name": hardware_name,
            "window_sec": config.window_sec,
            "realtime_threshold": config.realtime_threshold,
            "min_success_rate": config.min_success_rate,
            "configs": [s.to_dict() for s in summaries],
        },
    )
    if not all_results:
        logger.error(
            "Sweep produced no results: every model failed to load, or no video "
            "yielded a usable %gs window.",
            config.window_sec,
        )
    logger.info("Wrote %d results to %s", len(all_results), results_path)
    return run_dir
