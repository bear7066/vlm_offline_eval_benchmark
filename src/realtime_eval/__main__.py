"""CLI for the real-time VLM sweep.

Run ``uv run python -m realtime_eval --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from realtime_eval.core.config import DEFAULT_REALTIME_PROMPT, SweepConfig
from realtime_eval.core.metrics import RealtimeResult
from realtime_eval.pipeline.analyze import analyze
from realtime_eval.pipeline.sweep import run_single, run_sweep


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realtime_eval",
        description="Sweep VLM configs to find the largest model that stays real time.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser(
        "sweep",
        help="Run the benchmark sweep from a JSON config file.",
        description=(
            "Run the sweep described by a JSON config. The file fully specifies "
            "the run: a required 'videos' path, an optional 'limit', and any "
            "SweepConfig field (model_ids, num_frames_grid, max_new_tokens_grid, "
            "repeats, warmup, output_root, prompt, realtime_threshold, "
            "power_sample_interval_sec). Omitted fields keep their defaults."
        ),
    )
    sweep.add_argument("config", type=Path, help="Path to a JSON sweep config file.")

    sweep_vllm = sub.add_parser(
        "sweep-vllm",
        help="Run the benchmark sweep on the vLLM backend (requires the 'vllm' extra).",
        description=(
            "Same as 'sweep' but models are served by vLLM instead of "
            "transformers. Takes the same JSON config and writes the same "
            "output format. Requires the optional vLLM dependency: "
            "uv sync --extra vllm."
        ),
    )
    sweep_vllm.add_argument("config", type=Path, help="Path to a JSON sweep config file.")

    single = sub.add_parser(
        "single", help="Run one model on one video to verify the pipeline."
    )
    single.add_argument("video", type=Path, help="Path to a single video file.")
    single.add_argument(
        "--model_id",
        "-m",
        type=str,
        default="bear7011/gemma4-e2b-webvid4K_FT",
        help="Model to run (default: bear7011/gemma4-e2b-webvid4K_FT).",
    )
    single.add_argument("--num_frames", "-n", type=int, default=8, help="Frames to sample.")
    single.add_argument(
        "--max_new_tokens", type=int, default=40, help="Generation cap (default 40)."
    )
    single.add_argument("--repeats", type=int, default=1, help="Timed iterations (default 1).")
    single.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default 1).")
    single.add_argument("--label", "-l", type=str, default=None, help="Ground-truth label override.")
    single.add_argument(
        "--window_sec",
        type=float,
        default=1.0,
        help="Deployment stride in seconds; denominator of window_rtf (default 1.0).",
    )
    single.add_argument(
        "--prompt", "-p", type=str, default=DEFAULT_REALTIME_PROMPT, help="Prompt text."
    )

    an = sub.add_parser("analyze", help="Summarize a completed sweep run.")
    an.add_argument("run_dir", type=Path, help="Sweep run directory containing results.jsonl.")
    an.add_argument(
        "--threshold", type=float, default=0.8, help="p95 window_rtf cutoff (default 0.8)."
    )
    an.add_argument(
        "--min_success_rate",
        type=float,
        default=1.0,
        help="Run success fraction required to qualify as real time (default 1.0).",
    )

    config = sub.add_parser("config", help="Manage sweep config files.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    init = config_sub.add_parser(
        "init", help="Write a default sweep config to a target JSON file."
    )
    init.add_argument("target", type=Path, nargs="?", default="sweep-conf.json", help="Path to write the JSON config to.")
    init.add_argument(
        "--force", action="store_true", help="Overwrite the target if it already exists."
    )

    return parser


def _init_config(target: Path, force: bool = False) -> int:
    """Write a default sweep config to ``target`` as JSON.

    The file mirrors :meth:`SweepConfig.from_dict`'s accepted keys: the two
    input-selection keys (``videos``, ``limit``) followed by every
    :class:`SweepConfig` default, ready to edit.

    Args:
        target: Path to write the JSON config to.
        force: Overwrite an existing file instead of refusing.

    Returns:
        Process exit code (0 on success, 1 if the file exists without ``force``).
    """
    target = Path(target)
    if target.exists() and not force:
        print(
            f"Refusing to overwrite existing file: {target} (pass --force).",
            file=sys.stderr,
        )
        return 1

    payload = {"videos": "video.mp4", "limit": None, **SweepConfig().to_dict()}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote default sweep config to {target}")
    return 0


def _load_sweep_config(path: Path) -> tuple[Path, SweepConfig, int | None]:
    """Load a JSON sweep config into its components.

    The JSON object holds a required ``videos`` path, an optional ``limit``,
    and any :class:`SweepConfig` field. The two input-selection keys are pulled
    out and the rest is handed to :meth:`SweepConfig.from_dict`.

    Args:
        path: Path to the JSON config file.

    Returns:
        A ``(videos, config, limit)`` triple ready for :func:`run_sweep`.

    Raises:
        ValueError: If the JSON is not an object or is missing ``videos``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sweep config must be a JSON object, got {type(data).__name__}.")
    if "videos" not in data:
        raise ValueError("Sweep config must include a 'videos' path.")

    data = dict(data)
    videos = Path(data.pop("videos"))
    limit = data.pop("limit", None)
    return videos, SweepConfig.from_dict(data), limit


def _print_single(results: list[RealtimeResult]) -> None:
    """Print each timed run from a ``single`` invocation as JSON.

    Emits a JSON object with every metric for each run, so the output can be
    piped to ``jq`` or parsed downstream.

    Args:
        results: Results returned by :func:`realtime_eval.pipeline.sweep.run_single`.
    """
    print(json.dumps([r.to_dict() for r in results], indent=2))


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m realtime_eval``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "config":
        if args.config_command == "init":
            return _init_config(args.target, force=args.force)
        return 1

    if args.command == "sweep":
        videos, config, limit = _load_sweep_config(args.config)
        run_dir = run_sweep(videos, config, video_limit=limit)
        print(
            analyze(
                run_dir,
                threshold=config.realtime_threshold,
                min_success_rate=config.min_success_rate,
            )
        )
        return 0

    if args.command == "sweep-vllm":
        from realtime_eval.pipeline.sweep_vllm import run_sweep_vllm

        videos, config, limit = _load_sweep_config(args.config)
        run_dir = run_sweep_vllm(videos, config, video_limit=limit)
        print(
            analyze(
                run_dir,
                threshold=config.realtime_threshold,
                min_success_rate=config.min_success_rate,
            )
        )
        return 0

    if args.command == "single":
        results = run_single(
            video=args.video,
            model_id=args.model_id,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
            repeats=args.repeats,
            warmup=args.warmup,
            label=args.label,
            window_sec=args.window_sec,
        )
        _print_single(results)
        return 0

    if args.command == "analyze":
        print(
            analyze(
                args.run_dir,
                threshold=args.threshold,
                min_success_rate=args.min_success_rate,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
