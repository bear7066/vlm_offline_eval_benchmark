from __future__ import annotations

import json
from pathlib import Path

from vlm_eval.paths import model_name_from_id

from realtime_eval.core.metrics import (
    ConfigSummary,
    FrameScaling,
    RealtimeResult,
    aggregate,
    fit_frame_scaling,
    result_from_dict,
)


def load_results(run_dir: Path) -> list[RealtimeResult]:
    """Load per-run results from a sweep's ``results.jsonl``.

    Args:
        run_dir: Sweep run directory containing ``results.jsonl``.

    Returns:
        The list of :class:`RealtimeResult` records.

    Raises:
        FileNotFoundError: If ``results.jsonl`` is missing.
    """
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.jsonl in {run_dir}")

    results: list[RealtimeResult] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(result_from_dict(json.loads(line)))
    return results


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def format_table(summaries: list[ConfigSummary]) -> str:
    """Render per-config summaries as a fixed-width text table.

    Args:
        summaries: Aggregated config summaries.

    Returns:
        A multi-line string sorted by model, then frame count.
    """
    header = (
        f"{'model':<18}{'frames':>7}{'tok':>5}{'p50_e2e':>9}{'p95_e2e':>9}"
        f"{'max_e2e':>9}{'p95_ttft':>10}{'tpot':>7}{'p95_rtf':>9}"
        f"{'toks':>7}{'ovlp':>7}{'n':>4}{'RT?':>5}"
    )
    lines = [header, "-" * len(header)]
    for s in sorted(summaries, key=lambda x: (x.model_id, x.num_frames, x.max_new_tokens)):
        rt = "yes" if s.meets_realtime_p95 else "no"
        lines.append(
            f"{model_name_from_id(s.model_id):<18}"
            f"{s.num_frames:>7}{s.max_new_tokens:>5}"
            f"{_fmt(s.p50_e2e_latency_ms, '.0f'):>9}"
            f"{_fmt(s.p95_e2e_latency_ms, '.0f'):>9}"
            f"{_fmt(s.max_e2e_latency_ms, '.0f'):>9}"
            f"{_fmt(s.p95_ttft_ms, '.0f'):>10}"
            f"{_fmt(s.mean_tpot_ms, '.1f'):>7}"
            f"{_fmt(s.p95_rtf_inv):>9}"
            f"{_fmt(s.mean_tokens, '.0f'):>7}"
            f"{_fmt(s.naive_word_overlap):>7}"
            f"{s.n_videos_scored:>4}"
            f"{rt:>5}"
        )
    return "\n".join(lines)


def format_scaling_table(fits: list[FrameScaling]) -> str:
    """Render frame-scaling fits as a fixed-width text table.

    Args:
        fits: Per-``(model, tokens)`` frame-scaling fits.

    Returns:
        A multi-line string. ``R2`` is only meaningful at ``pts >= 3``; with two
        points the fit is exact by construction.
    """
    header = (
        f"{'model':<18}{'tok':>5}{'pts':>5}{'ttft_ms/frame':>15}{'ttft_fix_ms':>13}"
        f"{'R2':>7}{'pref_ms/frame':>15}{'pref_fix_ms':>13}{'R2':>7}"
    )
    lines = [header, "-" * len(header)]
    for f in fits:
        lines.append(
            f"{model_name_from_id(f.model_id):<18}"
            f"{f.max_new_tokens:>5}{f.n_points:>5}"
            f"{_fmt(f.ttft_ms_per_frame, '.1f'):>15}"
            f"{_fmt(f.ttft_fixed_ms, '.0f'):>13}"
            f"{_fmt(f.ttft_fit_r2, '.3f'):>7}"
            f"{_fmt(f.prefill_ms_per_frame, '.1f'):>15}"
            f"{_fmt(f.prefill_fixed_ms, '.0f'):>13}"
            f"{_fmt(f.prefill_fit_r2, '.3f'):>7}"
        )
    return "\n".join(lines)


def best_config(summaries: list[ConfigSummary]) -> ConfigSummary | None:
    """Pick the most capable config that still meets the real-time threshold.

    Ranks on frame count -- the most temporal evidence the model gets per
    inference -- breaking ties toward lower p95 latency.

    Deliberately does **not** rank on ``naive_word_overlap``. That proxy rewards
    verbosity, so it rises with ``max_new_tokens``, which is itself a swept axis;
    ranking on it would recommend whichever config was allowed to talk longest.
    Restore quality-based ranking only once ``vlm_eval.judge`` supplies real
    scores.

    Args:
        summaries: Aggregated config summaries.

    Returns:
        The recommended :class:`ConfigSummary`, or ``None`` if none are real time.
    """
    realtime = [s for s in summaries if s.meets_realtime_p95]
    if not realtime:
        return None
    return max(
        realtime,
        key=lambda s: (s.num_frames, -(s.p95_e2e_latency_ms or 0.0)),
    )


def analyze(run_dir: Path, threshold: float = 0.8) -> str:
    """Build a human-readable analysis report for a sweep run.

    Args:
        run_dir: Sweep run directory.
        threshold: p95 ``rtf_inv`` cutoff for the real-time decision.

    Returns:
        A report string with the per-config table and the recommended pick.
    """
    results = load_results(run_dir)
    summaries = aggregate(results, threshold=threshold)
    table = format_table(summaries)
    scaling = format_scaling_table(fit_frame_scaling(results))

    pick = best_config(summaries)
    if pick is None:
        verdict = (
            f"\nNo config meets p95 rtf_inv <= {threshold}. "
            "Reduce frames/resolution or consider a faster runtime (vLLM/quantization)."
        )
    else:
        verdict = (
            f"\nRecommended: {model_name_from_id(pick.model_id)} | "
            f"{pick.num_frames} frames | max_new_tokens={pick.max_new_tokens}\n"
            f"  Most frames among configs meeting p95 rtf_inv <= {threshold} "
            f"(p95={_fmt(pick.p95_rtf_inv)}, p95 latency="
            f"{_fmt(pick.p95_e2e_latency_ms, '.0f')} ms).\n"
            "  Quality is NOT ranked: ovlp is a verbosity-confounded proxy, not "
            "accuracy. Wire up the LLM judge before choosing on quality."
        )
    return f"{table}\n\nFrame scaling (marginal cost per added frame):\n{scaling}\n{verdict}"
