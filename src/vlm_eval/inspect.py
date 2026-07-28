"""
Run `uv run python -m src.vlm_eval.inspect --help` to get information.

This script runs one inference for given video.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from vlm_eval.config import JudgeConfig
from vlm_eval.hardware import (
    get_hardware_name,
    get_peak_vram_reserved_gb,
    reset_peak_memory_stats,
)
from vlm_eval.inference.gemma import HuggingFaceVLM
from vlm_eval.judge.runner import run_judge
from vlm_eval.logging_utils import configure_logging, quiet_third_party_loggers
from vlm_eval.metrics import VideoResult
from vlm_eval.paths import build_run_id, display_label_from_video_path, ensure_run_dir, slugify
from vlm_eval.video import sample_frames


DEFAULT_PROMPT="Detect if any accident happens (e.g. faceplanting, falling, explosion), if yes, describe in one concise sentence, otherwise, say everything is normal."

def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def _append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
        f.write("\n")


def run_inference(
    video_path: Path,
    model_id: str,
    num_frames: int,
    output_root: Path,
    prompt: str,
    label: str,
) -> Path | None:
    load_dotenv()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    quiet_third_party_loggers()

    run_id = build_run_id(model_id, video_path.stem, num_frames)
    try:
        run_dir = ensure_run_dir(Path(output_root), run_id)
    except FileExistsError:
        run_dir = Path(output_root) / slugify(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir / "benchmark.log", mode="a")

    _write_json(
        run_dir / "config.json",
        {
            "video": video_path,
            "model_id": model_id,
            "num_frames": num_frames,
            "output_root": output_root,
            "prompt": prompt,
            "label": label,
            "run_id": run_id,
        },
    )

    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    predictions_path.touch()

    hf_token = os.environ.get("HF_TOKEN")
    hardware_name = get_hardware_name()

    logging.info("Run directory: %s", run_dir)
    logging.info("Loading model and processor: %s", model_id)

    try:
        model = HuggingFaceVLM(model_id, hf_token=hf_token)
    except Exception as exc:
        logging.error("Failed to load model: %s", exc)
        return None

    reset_peak_memory_stats()

    logging.info("Hardware Name: %s", hardware_name)
    logging.info("Fixed Frames : %s", num_frames)
    logging.info("Processing video: %s", video_path)

    frames, video_duration_sec, total_video_frames, original_fps = sample_frames(
        video_path,
        num_frames=num_frames,
    )
    if frames is None:
        result = VideoResult(
            video=str(video_path),
            label=label,
            status="error",
            error="Could not sample frames",
        )
        _append_jsonl(predictions_path, result.to_dict())
        return run_dir

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")
    logging.info("Sampled frames written to: %s", frames_dir)

    try:
        generated = model.generate_from_frames(frames=frames, prompt_text=prompt)
        elapsed_sec = generated["elapsed_sec"]
        sampled_fps = len(frames) / elapsed_sec if elapsed_sec > 0 else 0.0

        result = VideoResult(
            video=str(video_path),
            label=label,
            status="success",
            response=generated["response"],
            query_latency_sec=elapsed_sec,
            query_latency_ms=generated["elapsed_ms"],
            ttft_ms=generated["ttft_ms"],
            video_duration_sec=video_duration_sec,
            total_video_frames=total_video_frames,
            original_fps=original_fps,
            sampled_frames=len(frames),
            tokens=generated["tokens"],
            throughput_tps=generated["throughput_tps"],
            frames_per_second=sampled_fps,
            average_power_watts=generated["average_power_watts"],
        )

        logging.info("Query Latency: %.2f ms", result.query_latency_ms)
        logging.info("Model answer: %s", result.response)

    except Exception as exc:
        logging.error("Inference failed: %s", exc)
        result = VideoResult(
            video=str(video_path),
            label=label,
            status="error",
            error=str(exc),
            video_duration_sec=video_duration_sec,
            total_video_frames=total_video_frames,
            original_fps=original_fps,
            sampled_frames=len(frames),
        )

    _append_jsonl(predictions_path, result.to_dict())

    logging.info("Peak VRAM: %s GB", get_peak_vram_reserved_gb())
    logging.info("Predictions written to: %s", predictions_path)

    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single inference + judge pass on one video."
    )
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument("--model_id", "-m", type=str, default="google/gemma-3-4b-it", help="default to google/gemma-3-4b-it")
    parser.add_argument("--num_frames", "-n", type=int, default=8, help="num frames, default to 8")
    parser.add_argument("--output_root", "-o", type=Path, default=Path("runs"))
    parser.add_argument("--prompt", "-p", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--label", "-l", type=str, required=True)
    args = parser.parse_args(argv)

    label = args.label or display_label_from_video_path(args.video)

    run_dir = run_inference(
        video_path=args.video,
        model_id=args.model_id,
        num_frames=args.num_frames,
        output_root=args.output_root,
        prompt=args.prompt,
        label=label,
    )
    if run_dir is None:
        return 1

    judge_config = JudgeConfig(
        judge_model="gpt-4o",
        predictions_path=run_dir / "predictions.jsonl",
        skip_llm_judge=True,
    )
    judged_dir = run_judge(judge_config)
    return 0 if judged_dir is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
