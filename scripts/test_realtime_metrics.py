"""Self-check for realtime_eval metric definitions.

Run: uv run scripts/test_realtime_metrics.py

Covers the timing primitives without needing a GPU or model weights, by driving
the streamer and a stub model directly.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from realtime_eval.core.metrics import (
    RealtimeResult,
    aggregate,
    fit_frame_scaling,
    result_from_dict,
)
from realtime_eval.core.power import DeviceSampler
from realtime_eval.pipeline.analyze import best_config
from vlm_eval.hardware import get_gpu_power_watts
from vlm_eval.inference.gemma import HuggingFaceVLM, _first_token_streamer_cls
from vlm_eval.video import frame_span, sample_frames

# Deliberately word-shaped: the first token has no trailing space, which is
# exactly the case the old TTFT measurement got wrong.
_VOCAB = {5: "everything", 6: " is", 7: " normal"}


class _FakeTokenizer:
    """Minimal stand-in; ``TextStreamer`` only ever calls ``decode``."""

    def decode(self, ids, **kwargs) -> str:
        return "".join(_VOCAB.get(int(i), "?") for i in ids)

    def encode(self, text, **kwargs) -> list[int]:
        return list(range(len(text.split())))


def check_first_token_timestamp() -> None:
    """The streamer must time the first *token*, not the first non-empty chunk."""
    streamer = _first_token_streamer_cls()(_FakeTokenizer(), skip_prompt=True)

    streamer.put(torch.tensor([[1, 2, 3, 4]]))  # prompt
    assert streamer.first_token_time is None, "prompt must not be timed"

    before = time.perf_counter()
    streamer.put(torch.tensor([5]))  # first generated token
    after = time.perf_counter()

    t1 = streamer.first_token_time
    assert t1 is not None, "first generated token was not timed"
    assert before <= t1 <= after, (before, t1, after)

    # The regression this guards: the first queued chunk is EMPTY, because
    # TextStreamer withholds text until a space arrives. Timing the first
    # non-empty chunk charged token 2's arrival to TTFT.
    assert streamer.text_queue.get_nowait() == "", "expected an empty first chunk"

    streamer.put(torch.tensor([6]))  # second token completes a word
    assert streamer.text_queue.get_nowait() == "everything ", "expected a word flush"
    assert streamer.first_token_time == t1, "timestamp must not move after the first token"


class _Inputs(dict):
    """Mapping that mimics a ``BatchFeature``: ``.to(device)`` returns itself."""

    def to(self, *_args, **_kwargs) -> "_Inputs":
        return self


class _FakeProcessor:
    PREPROCESS_SEC = 0.02

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs) -> str:
        return "<prompt>"

    def __call__(self, text=None, images=None, return_tensors=None) -> _Inputs:
        time.sleep(self.PREPROCESS_SEC)  # stand in for resize/normalize cost
        return _Inputs(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            pixel_values=torch.zeros(1, 3, 4, 4),
        )


class _FakeModel:
    PREFILL_SEC = 0.05
    STEP_SEC = 0.01
    NEW_IDS = [5, 6, 7]

    device = "cpu"

    def generate(self, **kwargs):
        streamer = kwargs["streamer"]
        input_ids = kwargs["input_ids"]
        streamer.put(input_ids)  # prompt
        time.sleep(self.PREFILL_SEC)
        for tid in self.NEW_IDS:
            streamer.put(torch.tensor([tid]))
            time.sleep(self.STEP_SEC)
        streamer.end()
        return torch.cat([input_ids, torch.tensor([self.NEW_IDS])], dim=-1)


def _stub_vlm() -> HuggingFaceVLM:
    """Build a HuggingFaceVLM without loading weights."""
    vlm = HuggingFaceVLM.__new__(HuggingFaceVLM)
    vlm.torch = torch
    vlm.model_id = "stub/model"
    vlm.processor = _FakeProcessor()
    vlm.model = _FakeModel()
    return vlm


def check_phase_split() -> None:
    """Phase timings must be exact, additive, and correctly normalized."""
    out = _stub_vlm().generate_from_frames(
        frames=[object(), object()], prompt_text="describe", max_new_tokens=8
    )

    assert out["response"] == "everything is normal", out["response"]
    assert out["tokens"] == 3, out["tokens"]
    assert out["ttft_source"] == "first-token-hook"

    pre, prefill, decode = out["preprocess_ms"], out["prefill_ms"], out["decode_ms"]

    # The advertised identities must hold to float precision.
    assert abs(out["ttft_ms"] - (pre + prefill)) < 1e-6, out
    assert abs(out["elapsed_ms"] - (pre + prefill + decode)) < 1e-6, out
    assert abs(out["elapsed_ms"] - (out["ttft_ms"] + decode)) < 1e-6, out

    # Each phase must be attributed to the right place, not lumped together.
    assert pre >= _FakeProcessor.PREPROCESS_SEC * 1000, pre
    assert prefill >= _FakeModel.PREFILL_SEC * 1000, prefill
    # TTFT must NOT have absorbed a decode step: the old code would land near
    # prefill + one STEP_SEC. Allow generous slack for scheduling jitter.
    assert prefill < (_FakeModel.PREFILL_SEC + _FakeModel.STEP_SEC) * 1000, prefill

    # decode_ms spans the 2 gaps between 3 tokens.
    assert abs(out["tpot_ms"] - decode / 2) < 1e-6, out
    assert abs(out["decode_tps"] - 2 / (decode / 1000.0)) < 1e-6, out
    assert out["decode_tps"] > out["throughput_tps"], "decode rate must exceed end-to-end rate"


def check_token_count() -> None:
    """Exact count from generated ids; re-encode only as a fallback."""
    vlm = _stub_vlm()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    assert vlm._new_token_count(ids, 4, "ignored") == 3
    assert vlm._new_token_count(ids, 7, "ignored") == 0, "must not go negative"
    # No ids and no prompt length -> fallback to re-encoding the text.
    assert vlm._new_token_count(None, None, "one two three") == 3


def check_generate_error_propagates() -> None:
    """A failure inside the generation thread must raise, not hang."""
    vlm = _stub_vlm()

    class _Boom:
        device = "cpu"

        def generate(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    vlm.model = _Boom()
    try:
        vlm.generate_from_frames(frames=[object()], prompt_text="x", max_new_tokens=4)
    except RuntimeError as exc:
        assert "out of memory" in str(exc), exc
    else:
        raise AssertionError("expected the generate failure to propagate")


# --- frame scaling (C2) ---------------------------------------------------

PREFILL_SLOPE = 12.0  # ms per frame
PREFILL_FIXED = 40.0  # ms of frame-independent cost
PREPROCESS_SLOPE = 3.0  # ms per frame of CPU preprocessing


def _synth(frames: int, max_tokens: int = 20, **over) -> RealtimeResult:
    """A successful result whose latency follows the known linear model."""
    prefill = PREFILL_FIXED + PREFILL_SLOPE * frames
    preprocess = PREPROCESS_SLOPE * frames
    decode = 100.0
    fields = dict(
        video=f"v{frames}.mp4",
        label="fall_general",
        model_id="m/one",
        num_frames=frames,
        num_frames_actual=frames,
        max_new_tokens=max_tokens,
        preprocess_ms=preprocess,
        prefill_ms=prefill,
        ttft_ms=preprocess + prefill,
        decode_ms=decode,
        e2e_latency_ms=preprocess + prefill + decode,
        window_sec=1.0,
        window_rtf=0.5,
        max_sustainable_fps=float(frames) / 0.5,
        rtf=0.5,
    )
    fields.update(over)
    return RealtimeResult(**fields)


def check_frame_scaling_recovers_slope() -> None:
    """The fit must recover the true marginal cost and separate fixed overhead."""
    results = [_synth(f) for f in (8, 12, 16)]
    (fit,) = fit_frame_scaling(results)

    assert fit.n_points == 3, fit
    assert fit.frame_counts == [8, 12, 16], fit
    assert abs(fit.prefill_ms_per_frame - PREFILL_SLOPE) < 1e-9, fit
    assert abs(fit.prefill_fixed_ms - PREFILL_FIXED) < 1e-9, fit
    assert abs(fit.prefill_fit_r2 - 1.0) < 1e-9, fit
    # TTFT scaling picks up preprocessing on top of prefill.
    assert abs(fit.ttft_ms_per_frame - (PREFILL_SLOPE + PREPROCESS_SLOPE)) < 1e-9, fit
    assert abs(fit.ttft_fixed_ms - PREFILL_FIXED) < 1e-9, fit


def check_slope_beats_naive_ratio() -> None:
    """The old ttft/num_frames metric drifts across the grid; the slope does not."""
    naive = [(PREFILL_FIXED + PREFILL_SLOPE * f) / f for f in (8, 12, 16)]
    assert naive[0] > naive[1] > naive[2], naive  # monotone drift, not a constant
    assert max(naive) - min(naive) > 2.0, naive  # and not a rounding-level drift

    # Same data, fitted: one number, equal to the true per-frame cost.
    (fit,) = fit_frame_scaling([_synth(f) for f in (8, 12, 16)])
    assert abs(fit.prefill_ms_per_frame - PREFILL_SLOPE) < 1e-9, fit
    assert all(abs(fit.prefill_ms_per_frame - n) > 2.0 for n in naive), (fit, naive)


def check_scaling_needs_two_points() -> None:
    """A single frame count yields no slope, and says so rather than guessing."""
    (fit,) = fit_frame_scaling([_synth(8), _synth(8, video="other.mp4")])
    assert fit.n_points == 1, fit
    assert fit.prefill_ms_per_frame is None, fit
    assert fit.ttft_fit_r2 is None, fit


def check_scaling_uses_actual_frames() -> None:
    """Short clips feed fewer frames; the fit must use what was actually fed."""
    # Requested 16 but only 10 frames existed: grouped and fitted at 10.
    results = [_synth(8), _synth(16, num_frames_actual=10, num_frames=16)]
    (fit,) = fit_frame_scaling(results)
    assert fit.frame_counts == [8, 10], fit


def check_aggregate_and_schema() -> None:
    """Aggregation reads the new fields; stale records are rejected loudly."""
    (summary,) = aggregate([_synth(8), _synth(8, video="b.mp4")], threshold=0.8)
    assert summary.n_success == 2, summary
    assert summary.num_frames_actual == 8, summary
    assert abs(summary.mean_prefill_ms - (PREFILL_FIXED + PREFILL_SLOPE * 8)) < 1e-9, summary
    assert summary.p95_e2e_latency_ms is not None and summary.max_e2e_latency_ms is not None
    assert summary.meets_realtime_p95 is True, summary

    # Records from the pre-fix schema must not be silently reinterpreted.
    good = _synth(8).to_dict()
    assert result_from_dict(good).num_frames == 8
    try:
        result_from_dict({**good, "prefill_ms_per_frame": 17.0})
    except ValueError as exc:
        assert "prefill_ms_per_frame" in str(exc), exc
    else:
        raise AssertionError("expected stale schema fields to be rejected")


# --- real-time criterion (C4) and reliability (C5) ------------------------


def check_window_rtf_is_duration_independent() -> None:
    """Identical work on clips of different lengths must get the same verdict."""
    # Same latency (400 ms), same frame count, different clip durations.
    short = _synth(8, e2e_latency_ms=400.0, window_rtf=0.4, rtf=0.4 / 1.0, video_duration_sec=1.0)
    long = _synth(8, e2e_latency_ms=400.0, window_rtf=0.4, rtf=0.4 / 10.0, video_duration_sec=10.0)

    # rtf disagrees by 10x purely because of clip length -- the old confound.
    assert abs(short.rtf / long.rtf - 10.0) < 1e-9, (short.rtf, long.rtf)
    # window_rtf is identical, because the deployment stride is what matters.
    assert short.window_rtf == long.window_rtf

    (summary,) = aggregate([short, long], threshold=0.8)
    assert summary.p95_window_rtf == 0.4, summary
    assert summary.meets_realtime_p95 is True, summary
    assert summary.window_sec == 1.0, summary


def check_window_rtf_drives_the_verdict() -> None:
    """A config slower than its stride must fail, however short the clips are."""
    # 2 s of work per 1 s window: not real time, even though rtf looks fine
    # against a 30 s clip.
    slow = _synth(8, e2e_latency_ms=2000.0, window_rtf=2.0, rtf=2.0 / 30.0, video_duration_sec=30.0)
    (summary,) = aggregate([slow], threshold=0.8)
    assert summary.p95_rtf < 0.1, summary  # would have passed on the old metric
    assert summary.meets_realtime_p95 is False, summary


def check_failures_are_counted_and_gate_the_verdict() -> None:
    """Crashed runs must not vanish, and must disqualify a real-time claim."""
    items = [_synth(8, repeat_index=0), _synth(8, repeat_index=1)]
    items += [
        _synth(8, repeat_index=i, status="error", error="CUDA out of memory")
        for i in range(2, 5)
    ]
    (summary,) = aggregate(items, threshold=0.8, min_success_rate=1.0)

    assert summary.n_attempted == 5, summary
    assert summary.n_success == 2, summary
    assert summary.n_error == 3, summary
    assert abs(summary.success_rate - 0.4) < 1e-9, summary
    # The two surviving runs are fast, but the config is not dependable.
    assert summary.p95_window_rtf == 0.5, summary
    assert summary.meets_realtime_p95 is False, summary

    # Relaxing the floor lets it qualify again -- an explicit, visible choice.
    (relaxed,) = aggregate(items, threshold=0.8, min_success_rate=0.4)
    assert relaxed.meets_realtime_p95 is True, relaxed


def check_all_error_config_still_reported() -> None:
    """A config that never succeeded must appear, not silently disappear."""
    items = [_synth(8, repeat_index=i, status="error", error="boom") for i in range(3)]
    (summary,) = aggregate(items)
    assert summary.n_attempted == 3 and summary.n_success == 0, summary
    assert summary.success_rate == 0.0, summary
    assert summary.p95_e2e_latency_ms is None, summary
    assert summary.meets_realtime_p95 is None, summary


# --- power and energy (S8) ------------------------------------------------


def check_energy_integration() -> None:
    """Energy must be a proper time integral, and mean power time-weighted."""
    sampler = DeviceSampler(interval_sec=0.01)
    # Hand-place samples: 100 W for 1 s, then ramp to 300 W over the next 1 s.
    # Trapezoid: 100*1 + (100+300)/2*1 = 300 J over a 2 s span -> 150 W mean.
    sampler._samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 300.0)]

    assert abs(sampler.energy_j - 300.0) < 1e-9, sampler.energy_j
    assert abs(sampler.sampled_sec - 2.0) < 1e-9
    assert abs(sampler.mean_watts - 150.0) < 1e-9, sampler.mean_watts
    assert sampler.peak_watts == 300.0
    assert sampler.n_power_samples == 3

    # An unweighted mean of the readings would give 166.7 W -- biased by the
    # uneven spacing. That is the bug the time weighting removes.
    naive = sum(w for _t, w in sampler._samples) / 3
    assert abs(naive - 166.667) < 0.01 and abs(naive - sampler.mean_watts) > 15, naive


def check_single_sample_is_not_integrated() -> None:
    """One reading spans no time, so energy is unavailable rather than wrong."""
    sampler = DeviceSampler()
    sampler._samples = [(0.0, 120.0)]
    assert sampler.energy_j is None
    assert sampler.sampled_sec is None
    assert sampler.mean_watts == 120.0, "still report the lone reading"
    assert sampler.n_power_samples == 1, "caller can see it is untrustworthy"

    empty = DeviceSampler()
    assert empty.energy_j is None and empty.mean_watts is None and empty.peak_watts is None


def check_power_aggregates_by_total_energy() -> None:
    """Mean power over runs must be total energy / total time, not mean of means."""
    # Run A: 10 J over 1 s (10 W). Run B: 300 J over 10 s (30 W).
    # Correct: 310 J / 11 s = 28.2 W. Mean of means would say 20 W.
    a = _synth(8, energy_j=10.0, power_sampled_sec=1.0, mean_power_watts=10.0)
    b = _synth(8, video="b.mp4", energy_j=300.0, power_sampled_sec=10.0, mean_power_watts=30.0)
    (summary,) = aggregate([a, b])

    assert abs(summary.mean_power_watts - 310.0 / 11.0) < 1e-9, summary.mean_power_watts
    assert abs(summary.mean_power_watts - 20.0) > 5.0, "must not be the mean of means"
    assert abs(summary.mean_energy_j - 155.0) < 1e-9, summary.mean_energy_j


def check_vram_reports_both_views() -> None:
    """Allocator reserved and device-wide used are distinct, both reported."""
    a = _synth(8, peak_vram_reserved_gb=4.0, peak_vram_device_gb=4.7)
    b = _synth(8, video="b.mp4", peak_vram_reserved_gb=6.0, peak_vram_device_gb=6.8)
    (summary,) = aggregate([a, b])

    assert abs(summary.mean_peak_vram_reserved_gb - 5.0) < 1e-9, summary
    # Device-wide is a max, not a mean: sizing hardware needs the worst case.
    assert abs(summary.max_peak_vram_device_gb - 6.8) < 1e-9, summary
    assert summary.max_peak_vram_device_gb > summary.mean_peak_vram_reserved_gb


def check_power_reading_is_cheap() -> None:
    """Sampling must not cost enough to perturb the region being measured."""
    get_gpu_power_watts()  # warm the NVML handle
    start = time.perf_counter()
    for _ in range(20):
        get_gpu_power_watts()
    per_call_ms = (time.perf_counter() - start) / 20 * 1000

    # The nvidia-smi subprocess this replaced measured ~32 ms/call on this box.
    # Only assert when a reading is actually available (NVML present).
    if get_gpu_power_watts() is not None:
        assert per_call_ms < 5.0, f"{per_call_ms:.1f} ms/call is too slow to sample"
    print(f"  power reading: {per_call_ms:.3f} ms/call")


def check_best_config_ranks_on_frames() -> None:
    """Ranking follows frames then latency; nothing else is consulted."""
    summaries = aggregate([_synth(8), _synth(16)])
    pick = best_config(summaries)
    assert pick is not None and pick.num_frames == 16, pick

    # Nothing is recommended when no config is fast enough.
    infeasible = aggregate([_synth(8, window_rtf=5.0)], threshold=0.8)
    assert best_config(infeasible) is None


# --- windowed sampling -----------------------------------------------------


def check_frame_span_is_half_open() -> None:
    """Consecutive windows must tile the clip without sharing a frame."""
    fps, total = 30.0, 120  # 4.0 s

    # 1 s windows divide evenly: 30 frames each, contiguous, no overlap.
    spans = [frame_span(k * 1.0, (k + 1) * 1.0, fps, total) for k in range(4)]
    assert spans == [(0, 29), (30, 59), (60, 89), (90, 119)], spans
    for (_, prev_last), (next_first, _) in zip(spans, spans[1:]):
        assert next_first == prev_last + 1, (prev_last, next_first)

    # The whole-clip span must match the pre-window behaviour exactly.
    assert frame_span(0.0, total / fps, fps, total) == (0, total - 1)

    # 0.25 s at 30 fps is 7.5 frames, so availability alternates 8/7. The
    # validation rule uses floor() = 7, the minimum any window can supply.
    counts = [
        last - first + 1
        for first, last in (
            frame_span(k * 0.25, (k + 1) * 0.25, fps, total) for k in range(4)
        )
    ]
    assert counts == [8, 7, 8, 7], counts
    assert min(counts) == math.floor(0.25 * fps) == 7, counts

    # An interval falling between two frame times yields an empty span.
    first, last = frame_span(0.001, 0.002, fps, total)
    assert last < first, (first, last)


def check_windowed_sampling_on_real_video() -> None:
    """Sampled timestamps must land inside the requested window."""
    video = Path(__file__).resolve().parents[1] / "video.mp4"
    if not video.exists():
        print("  (skipped windowed video check: video.mp4 absent)")
        return

    whole, duration, total, fps = sample_frames(video, num_frames=8)
    assert whole is not None and len(whole) == 8
    assert abs(duration - 4.0) < 0.05, duration

    # window_sec == clip duration must reproduce whole-clip sampling exactly.
    same, _, _, _ = sample_frames(video, num_frames=8, start_sec=0.0, end_sec=duration)
    assert same is not None and len(same) == len(whole)
    assert [f.tobytes() for f in same] == [f.tobytes() for f in whole], (
        "window == whole clip must be byte-identical to unwindowed sampling"
    )

    # Four 1 s windows: 8 frames each, all timestamps inside their window.
    for k in range(4):
        start, end = k * 1.0, (k + 1) * 1.0
        frames, _, _, _ = sample_frames(video, num_frames=8, start_sec=start, end_sec=end)
        assert frames is not None and len(frames) == 8, (k, frames)
        first, last = frame_span(start, end, fps, total)
        assert start <= first / fps and (last + 1) / fps <= end + 1e-9, (k, first, last)

    # Windows carrying different content must differ from each other.
    w0, _, _, _ = sample_frames(video, num_frames=8, start_sec=0.0, end_sec=1.0)
    w3, _, _, _ = sample_frames(video, num_frames=8, start_sec=3.0, end_sec=4.0)
    assert w0[0].tobytes() != w3[0].tobytes(), "windows should show different frames"


def check_empty_window_returns_none() -> None:
    """A degenerate or out-of-range window must fail cleanly, not crash."""
    video = Path(__file__).resolve().parents[1] / "video.mp4"
    if not video.exists():
        print("  (skipped empty-window check: video.mp4 absent)")
        return
    for start, end in ((1.0, 1.0), (2.0, 1.0), (99.0, 100.0)):
        frames, _, _, _ = sample_frames(video, num_frames=8, start_sec=start, end_sec=end)
        assert frames is None, (start, end, frames)


def demo() -> None:
    check_first_token_timestamp()
    check_phase_split()
    check_token_count()
    check_generate_error_propagates()
    check_frame_scaling_recovers_slope()
    check_slope_beats_naive_ratio()
    check_scaling_needs_two_points()
    check_scaling_uses_actual_frames()
    check_aggregate_and_schema()
    check_window_rtf_is_duration_independent()
    check_window_rtf_drives_the_verdict()
    check_failures_are_counted_and_gate_the_verdict()
    check_all_error_config_still_reported()
    check_energy_integration()
    check_single_sample_is_not_integrated()
    check_power_aggregates_by_total_energy()
    check_vram_reports_both_views()
    check_power_reading_is_cheap()
    check_best_config_ranks_on_frames()
    check_frame_span_is_half_open()
    check_windowed_sampling_on_real_video()
    check_empty_window_returns_none()
    print("ok")


if __name__ == "__main__":
    demo()
