from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from statistics import StatisticsError, fmean, linear_regression
from typing import Any


@dataclass
class RealtimeResult:
    """One timed inference plus derived real-time metrics.

    The latency phases are an exact partition of the timed region:
    ``e2e_latency_ms == preprocess_ms + prefill_ms + decode_ms``, and
    ``ttft_ms == preprocess_ms + prefill_ms``.

    Attributes:
        video: Path to the source clip.
        label: Ground-truth action label.
        model_id: HuggingFace model ID used.
        num_frames: Frames *requested* for this config.
        num_frames_actual: Frames actually fed to the model. Lower than
            ``num_frames`` for clips with fewer frames than requested, since
            duplicate sample indices are dropped. Use this, not ``num_frames``,
            as the denominator of any per-frame quantity.
        max_new_tokens: Generation cap for this run.
        repeat_index: Zero-based index of the timed repeat.
        e2e_latency_ms: Whole timed region: preprocess + prefill + decode.
        preprocess_ms: Chat templating, image preprocessing, host-to-device copy.
        prefill_ms: Preprocessed inputs to first generated token.
        ttft_ms: End-to-end time to first token.
        decode_ms: First generated token to last.
        tpot_ms: Mean inter-token latency, ``decode_ms / (tokens - 1)``.
        decode_tps: Decode-only rate, ``(tokens - 1) / decode_sec``.
        throughput_tps: End-to-end rate, ``tokens / e2e_latency_sec``.
        ttft_source: How ``ttft_ms`` was obtained, so results from different
            backends are never compared blindly.
        rtf_inv: ``e2e_latency_sec / video_duration_sec``; <= 1.0 is real time.
        meets_realtime: Whether ``rtf_inv <= 1.0`` for this run.
        video_duration_sec: Source clip duration.
        tokens: Number of generated tokens.
        mean_power_watts: Mean GPU power over the inference.
        peak_power_watts: Peak GPU power over the inference.
        peak_vram_gb: Peak VRAM allocated during the inference.
        response: Model output text.
        correct: Whether the response matched the label (heuristic; may be None).
        status: ``"success"`` or ``"error"``.
        error: Error message when ``status == "error"``.
    """

    video: str
    label: str
    model_id: str
    num_frames: int
    max_new_tokens: int
    num_frames_actual: int | None = None
    repeat_index: int = 0
    e2e_latency_ms: float | None = None
    preprocess_ms: float | None = None
    prefill_ms: float | None = None
    ttft_ms: float | None = None
    decode_ms: float | None = None
    tpot_ms: float | None = None
    decode_tps: float | None = None
    throughput_tps: float | None = None
    ttft_source: str | None = None
    rtf_inv: float | None = None
    meets_realtime: bool | None = None
    video_duration_sec: float | None = None
    tokens: int | None = None
    mean_power_watts: float | None = None
    peak_power_watts: float | None = None
    peak_vram_gb: float | None = None
    response: str = ""
    correct: bool | None = None
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dict for JSON serialization."""
        return asdict(self)


def percentile(values: list[float], q: float) -> float | None:
    """Compute the ``q``-th percentile via linear interpolation.

    Args:
        values: Sample values; need not be sorted. Empty returns ``None``.
        q: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile, or ``None`` for an empty input.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


_WORD_RE = re.compile(r"[a-z0-9]+")


def naive_correct(response: str, label: str) -> bool:
    """Heuristic accuracy: does the response share label content words?

    This is a placeholder so the sweep is runnable without an LLM judge.
    It lowercases both strings, drops short stopword-like tokens, and checks
    whether any label content word appears in the response. Replace with the
    ``vlm_eval.judge`` pipeline before trusting the accuracy axis.

    Args:
        response: Model output text.
        label: Ground-truth label (underscores treated as spaces).

    Returns:
        True if any label content word is present in the response.
    """
    resp_words = set(_WORD_RE.findall(response.lower()))
    label_words = {
        word
        for word in _WORD_RE.findall(label.replace("_", " ").lower())
        if len(word) > 2
    }
    if not label_words:
        return False
    return bool(resp_words & label_words)


def _ols_fit(
    xs: list[float], ys: list[float]
) -> tuple[float | None, float | None, float | None]:
    """Least-squares fit of ``y = slope * x + intercept``, with its R^2.

    Args:
        xs: Independent values (frame counts).
        ys: Dependent values (milliseconds).

    Returns:
        ``(slope, intercept, r2)``, or ``(None, None, None)`` when fewer than
        two points or a constant ``x`` makes the slope unidentifiable. With
        exactly two points the fit is exact, so ``r2`` is 1.0 by construction
        and carries no information: check ``n_points`` before trusting it.
    """
    try:
        slope, intercept = linear_regression(xs, ys)
    except StatisticsError:
        return None, None, None
    mean_y = fmean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2


@dataclass
class FrameScaling:
    """How latency scales with frame count, for one ``(model, tokens)`` pair.

    Dividing a single TTFT by its frame count does not give a per-frame cost:
    TTFT also contains the text prefill, the first decode step and framework
    overhead, none of which depend on the frame count. That fixed term makes
    ``ttft_ms / num_frames`` drift downward as frames increase, so it cannot be
    compared across configs or used to extrapolate.

    Fitting ``latency(f) = slope * f + intercept`` across the frame grid
    separates the two: the slope is the marginal cost of one more frame, the
    intercept is the frame-independent floor.

    Attributes:
        model_id: HuggingFace model ID.
        max_new_tokens: Generation cap the fit was taken at.
        n_points: Distinct frame counts in the fit. Needs >= 2; >= 3 to make
            ``fit_r2`` meaningful.
        frame_counts: The distinct actual frame counts fitted, ascending.
        ttft_ms_per_frame: Marginal end-to-end cost per added frame
            (preprocessing + prefill).
        ttft_fixed_ms: End-to-end fixed cost at zero frames.
        ttft_fit_r2: R^2 of the TTFT fit; low values mean the linear model is
            a poor summary and the slope should not be extrapolated.
        prefill_ms_per_frame: Marginal GPU prefill cost per added frame.
        prefill_fixed_ms: Prefill fixed cost at zero frames.
        prefill_fit_r2: R^2 of the prefill fit.
    """

    model_id: str
    max_new_tokens: int
    n_points: int
    frame_counts: list[int] = field(default_factory=list)
    ttft_ms_per_frame: float | None = None
    ttft_fixed_ms: float | None = None
    ttft_fit_r2: float | None = None
    prefill_ms_per_frame: float | None = None
    prefill_fixed_ms: float | None = None
    prefill_fit_r2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the fit as a plain dict for JSON serialization."""
        return asdict(self)


def _frames_of(result: RealtimeResult) -> int:
    """Frames actually fed, falling back to the requested count."""
    return result.num_frames_actual if result.num_frames_actual is not None else result.num_frames


def fit_frame_scaling(results: list[RealtimeResult]) -> list[FrameScaling]:
    """Fit latency-vs-frames across the frame grid, per ``(model, tokens)``.

    Averages each distinct actual frame count first, so every frame count gets
    equal weight in the fit regardless of how many videos or repeats ran at it.

    Args:
        results: All per-run results from a sweep.

    Returns:
        One :class:`FrameScaling` per ``(model_id, max_new_tokens)``, ordered by
        model then token cap. Groups with a single frame count are still
        returned, with ``None`` slopes, so the gap is visible rather than silent.
    """
    groups: dict[tuple[str, int], dict[int, list[RealtimeResult]]] = {}
    for result in results:
        if result.status != "success":
            continue
        by_frames = groups.setdefault((result.model_id, result.max_new_tokens), {})
        by_frames.setdefault(_frames_of(result), []).append(result)

    fits: list[FrameScaling] = []
    for (model_id, max_new_tokens), by_frames in sorted(groups.items()):
        counts = sorted(by_frames)
        xs: list[float] = []
        ttfts: list[float] = []
        prefills: list[float] = []
        for count in counts:
            items = by_frames[count]
            ttft_values = [r.ttft_ms for r in items if r.ttft_ms is not None]
            prefill_values = [r.prefill_ms for r in items if r.prefill_ms is not None]
            if not ttft_values or not prefill_values:
                continue
            xs.append(float(count))
            ttfts.append(fmean(ttft_values))
            prefills.append(fmean(prefill_values))

        ttft_slope, ttft_fixed, ttft_r2 = _ols_fit(xs, ttfts)
        prefill_slope, prefill_fixed, prefill_r2 = _ols_fit(xs, prefills)
        fits.append(
            FrameScaling(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                n_points=len(xs),
                frame_counts=[int(x) for x in xs],
                ttft_ms_per_frame=ttft_slope,
                ttft_fixed_ms=ttft_fixed,
                ttft_fit_r2=ttft_r2,
                prefill_ms_per_frame=prefill_slope,
                prefill_fixed_ms=prefill_fixed,
                prefill_fit_r2=prefill_r2,
            )
        )
    return fits


@dataclass
class ConfigSummary:
    """Aggregated metrics for one ``(model, frames, tokens)`` config."""

    model_id: str
    num_frames: int
    max_new_tokens: int
    n_runs: int
    num_frames_actual: int | None
    p50_e2e_latency_ms: float | None
    p95_e2e_latency_ms: float | None
    max_e2e_latency_ms: float | None
    p50_ttft_ms: float | None
    p95_ttft_ms: float | None
    p50_rtf_inv: float | None
    p95_rtf_inv: float | None
    mean_preprocess_ms: float | None
    mean_prefill_ms: float | None
    mean_decode_ms: float | None
    mean_tpot_ms: float | None
    mean_decode_tps: float | None
    mean_tokens: float | None
    mean_peak_vram_gb: float | None
    mean_power_watts: float | None
    accuracy: float | None
    meets_realtime_p95: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain dict for JSON serialization."""
        return asdict(self)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _single(values: list[int]) -> int | None:
    """The one distinct value, or ``None`` if the runs disagree."""
    unique = set(values)
    return unique.pop() if len(unique) == 1 else None


def aggregate(results: list[RealtimeResult], threshold: float = 0.8) -> list[ConfigSummary]:
    """Collapse per-run results into one summary per config.

    Groups successful results by ``(model_id, num_frames, max_new_tokens)`` and
    computes latency percentiles, the p95 real-time factor, and accuracy.

    Args:
        results: All per-run results from a sweep.
        threshold: p95 ``rtf_inv`` cutoff used to set ``meets_realtime_p95``.

    Returns:
        One :class:`ConfigSummary` per config, ordered by model then frames.
    """
    groups: dict[tuple[str, int, int], list[RealtimeResult]] = {}
    for result in results:
        if result.status != "success":
            continue
        key = (result.model_id, result.num_frames, result.max_new_tokens)
        groups.setdefault(key, []).append(result)

    def column(items: list[RealtimeResult], name: str) -> list[float]:
        return [v for v in (getattr(r, name) for r in items) if v is not None]

    summaries: list[ConfigSummary] = []
    for (model_id, num_frames, max_new_tokens), items in sorted(groups.items()):
        latencies = column(items, "e2e_latency_ms")
        ttfts = column(items, "ttft_ms")
        rtfs = column(items, "rtf_inv")
        correct = [r.correct for r in items if r.correct is not None]
        p95_rtf = percentile(rtfs, 95)
        summaries.append(
            ConfigSummary(
                model_id=model_id,
                num_frames=num_frames,
                max_new_tokens=max_new_tokens,
                n_runs=len(items),
                num_frames_actual=_single([_frames_of(r) for r in items]),
                p50_e2e_latency_ms=percentile(latencies, 50),
                p95_e2e_latency_ms=percentile(latencies, 95),
                max_e2e_latency_ms=max(latencies) if latencies else None,
                p50_ttft_ms=percentile(ttfts, 50),
                p95_ttft_ms=percentile(ttfts, 95),
                p50_rtf_inv=percentile(rtfs, 50),
                p95_rtf_inv=p95_rtf,
                mean_preprocess_ms=_mean(column(items, "preprocess_ms")),
                mean_prefill_ms=_mean(column(items, "prefill_ms")),
                mean_decode_ms=_mean(column(items, "decode_ms")),
                mean_tpot_ms=_mean(column(items, "tpot_ms")),
                mean_decode_tps=_mean(column(items, "decode_tps")),
                mean_tokens=_mean(column(items, "tokens")),
                mean_peak_vram_gb=_mean(column(items, "peak_vram_gb")),
                mean_power_watts=_mean(column(items, "mean_power_watts")),
                accuracy=(sum(correct) / len(correct)) if correct else None,
                meets_realtime_p95=(p95_rtf <= threshold) if p95_rtf is not None else None,
            )
        )
    return summaries


def result_from_dict(data: dict[str, Any]) -> RealtimeResult:
    """Build a :class:`RealtimeResult` from a ``results.jsonl`` record.

    Args:
        data: One decoded JSONL record.

    Returns:
        The parsed result.

    Raises:
        ValueError: If the record carries fields this schema does not define,
            which means it came from a sweep whose metric definitions differ.
    """
    known = {f.name for f in fields(RealtimeResult)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Result record has unknown fields {sorted(unknown)}; it predates the "
            "current metric definitions and is not comparable. Re-run the sweep."
        )
    return RealtimeResult(**data)
