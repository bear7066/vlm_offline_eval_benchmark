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


def demo() -> None:
    check_first_token_timestamp()
    check_phase_split()
    check_token_count()
    check_generate_error_propagates()
    print("ok")


if __name__ == "__main__":
    demo()
