"""Self-check for realtime_eval metric definitions.

Run: uv run scripts/test_realtime_metrics.py

Covers the timing primitives without needing a GPU or model weights, by driving
the streamer and a stub model directly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from realtime_eval.core.metrics import (
    RealtimeResult,
    aggregate,
    fit_frame_scaling,
    label_word_overlap,
    result_from_dict,
)
from realtime_eval.pipeline.analyze import best_config
from vlm_eval.inference.gemma import HuggingFaceVLM, _first_token_streamer_cls

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


def _synth(frames: int, tokens: int = 20, **over) -> RealtimeResult:
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
        max_new_tokens=tokens,
        preprocess_ms=preprocess,
        prefill_ms=prefill,
        ttft_ms=preprocess + prefill,
        decode_ms=decode,
        e2e_latency_ms=preprocess + prefill + decode,
        rtf_inv=0.5,
        label_word_overlap=True,
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
    assert summary.n_runs == 2, summary
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


# --- quality axis (C3) ----------------------------------------------------


def check_overlap_scored_once_per_video() -> None:
    """Greedy repeats are identical, so they must not inflate the denominator."""
    items = [_synth(8, repeat_index=i) for i in range(5)]
    items += [_synth(8, video="b.mp4", repeat_index=i, label_word_overlap=False) for i in range(5)]
    (summary,) = aggregate(items)

    assert summary.n_runs == 10, summary  # latency still uses every run
    assert summary.n_videos_scored == 2, summary  # quality uses each video once
    assert summary.naive_word_overlap == 0.5, summary


def check_overlap_is_verbosity_confounded() -> None:
    """The proxy rises with response length, so it tracks the swept token cap."""
    label = "fall_general"
    terse = "everything is normal"
    verbose = "everything is normal, no fall or general incident is visible here"
    assert label_word_overlap(terse, label) is False
    assert label_word_overlap(verbose, label) is True, "longer text hits by chance"


def check_best_config_ignores_overlap() -> None:
    """Ranking must follow frames, not the placeholder quality proxy."""
    low_frames_high_overlap = [_synth(8)]
    high_frames_low_overlap = [_synth(16, label_word_overlap=False)]
    summaries = aggregate(low_frames_high_overlap + high_frames_low_overlap)

    by_frames = {s.num_frames: s for s in summaries}
    assert by_frames[8].naive_word_overlap == 1.0
    assert by_frames[16].naive_word_overlap == 0.0

    pick = best_config(summaries)
    assert pick is not None and pick.num_frames == 16, pick

    # Nothing is recommended when no config is fast enough.
    infeasible = aggregate([_synth(8, rtf_inv=5.0)], threshold=0.8)
    assert best_config(infeasible) is None


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
    check_overlap_scored_once_per_video()
    check_overlap_is_verbosity_confounded()
    check_best_config_ignores_overlap()
    print("ok")


if __name__ == "__main__":
    demo()
