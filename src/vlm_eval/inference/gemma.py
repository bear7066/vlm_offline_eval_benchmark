from __future__ import annotations

import time
from functools import lru_cache
from threading import Thread
from typing import Any

from vlm_eval.hardware import get_gpu_power_watts


@lru_cache(maxsize=1)
def _first_token_streamer_cls() -> type:
    """Build the timestamping streamer subclass on first use.

    Kept behind a cached factory so importing this module does not drag in
    ``transformers``, matching the lazy-import style used elsewhere here.
    """
    from transformers import TextIteratorStreamer

    class _FirstTokenStreamer(TextIteratorStreamer):
        """Streamer that records when the first *generated* token appeared.

        ``TextStreamer.put`` only forwards text up to the last space, so the
        first non-empty chunk a consumer pulls off the queue arrives one or more
        decode steps after the first token actually existed (a single-token
        response like ``"everything"`` yields ``""`` until a space shows up).
        Stamping the clock inside ``put`` — which runs synchronously in the
        generation thread — times the token itself, with no detokenization or
        queue handoff folded into the number.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.first_token_time: float | None = None

        def put(self, value: Any) -> None:
            # The first call carries the prompt; ``super().put`` clears the flag.
            is_prompt = self.skip_prompt and self.next_tokens_are_prompt
            if not is_prompt and self.first_token_time is None:
                self.first_token_time = time.perf_counter()
            super().put(value)

    return _FirstTokenStreamer


class HuggingFaceVLM:
    def __init__(self, model_id: str, hf_token: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _new_token_count(self, generated_ids: Any, prompt_len: int | None, response: str) -> int:
        """Count generated tokens exactly, falling back to re-encoding the text.

        The exact count comes from the sequence ``generate`` returned minus the
        prompt length. Re-encoding ``response`` is only a fallback: detokenizing
        and re-tokenizing is not the identity, and the text has already lost
        special tokens and surrounding whitespace, so it undercounts.

        Args:
            generated_ids: Tensor returned by ``model.generate``, or ``None``.
            prompt_len: Prompt length in tokens, or ``None`` if unavailable.
            response: Decoded response text, used only by the fallback.

        Returns:
            Number of tokens generated, including any EOS/end-of-turn token.
        """
        if generated_ids is not None and prompt_len is not None:
            try:
                return max(int(generated_ids.shape[-1]) - prompt_len, 0)
            except (AttributeError, IndexError, TypeError):
                pass
        return len(self.processor.tokenizer.encode(response, add_special_tokens=False))

    def generate_from_frames(
        self,
        frames: list[Any],
        prompt_text: str,
        max_new_tokens: int = 150,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Run one generation over ``frames`` and return the text plus timings.

        The timed region starts at prompt formatting and ends when the last
        token has been produced, so it covers the per-request work a deployment
        actually pays for: chat templating, image preprocessing, the host-to-
        device copy, prefill, and decode. Video decoding and frame sampling are
        the caller's job and are deliberately outside it.

        Args:
            frames: Sampled PIL frames, one image placeholder each.
            prompt_text: Instruction text appended after the images.
            max_new_tokens: Generation cap.
            enable_thinking: Passed through to the chat template.

        Returns:
            A dict of the response and its timings. The phase split is exact:
            ``elapsed_ms == preprocess_ms + prefill_ms + decode_ms`` and
            ``ttft_ms == preprocess_ms + prefill_ms``.

            - ``response`` (str): generated text, stripped.
            - ``preprocess_ms``: chat template + processor + H2D copy.
            - ``prefill_ms``: preprocessed inputs to first generated token
              (includes thread dispatch and the first decode step).
            - ``ttft_ms``: end-to-end time to first token.
            - ``decode_ms``: first generated token to last.
            - ``tpot_ms``: mean inter-token latency, ``decode_ms / (tokens - 1)``.
            - ``decode_tps``: decode-only rate, ``(tokens - 1) / decode_sec``.
            - ``throughput_tps``: end-to-end rate, ``tokens / elapsed_sec``.
            - ``elapsed_sec`` / ``elapsed_ms``: whole timed region.
            - ``tokens`` (int): exact generated token count.
            - ``ttft_source`` (str): how ``ttft_ms`` was obtained, so results
              from different backends are never compared blindly.
            - ``average_power_watts``: mean of a start and end reading.

            Timing fields are ``None`` when no token was generated.

        Raises:
            Exception: Whatever ``model.generate`` raised, re-raised here.
        """
        start_power = get_gpu_power_watts()

        # --- preprocess: real per-request work, so it is inside the timed region
        t_start = time.perf_counter()

        content_items: list[dict[str, Any]] = [{"type": "image"} for _ in range(len(frames))]
        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]

        formatted_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.processor(
            text=formatted_prompt,
            images=frames,
            return_tensors="pt",
        ).to(self.model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.bfloat16)

        t_preprocessed = time.perf_counter()

        # --- generate
        streamer = _first_token_streamer_cls()(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )

        outcome: dict[str, Any] = {}

        def _generate() -> None:
            try:
                outcome["ids"] = self.model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                outcome["error"] = exc
                # ponytail: generate() only calls end() on its success path, so
                # without this a failure leaves the consumer blocked forever.
                streamer.end()

        thread = Thread(target=_generate)
        thread.start()

        response_chunks = list(streamer)
        thread.join()

        t_end = time.perf_counter()
        end_power = get_gpu_power_watts()

        if "error" in outcome:
            raise outcome["error"]

        response = "".join(response_chunks).strip()

        prompt_len = None
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            prompt_len = int(input_ids.shape[-1])
        tokens = self._new_token_count(outcome.get("ids"), prompt_len, response)

        first_token_time = streamer.first_token_time
        elapsed_sec = t_end - t_start
        preprocess_ms = (t_preprocessed - t_start) * 1000.0
        prefill_ms = (
            (first_token_time - t_preprocessed) * 1000.0 if first_token_time is not None else None
        )
        decode_ms = (t_end - first_token_time) * 1000.0 if first_token_time is not None else None
        ttft_ms = preprocess_ms + prefill_ms if prefill_ms is not None else None

        # decode_ms spans the (tokens - 1) gaps between the first and last token.
        gaps = tokens - 1
        tpot_ms = decode_ms / gaps if decode_ms is not None and gaps > 0 else None
        decode_tps = (
            gaps / (decode_ms / 1000.0) if decode_ms is not None and decode_ms > 0 and gaps > 0
            else None
        )

        power_reading = None
        if start_power is not None and end_power is not None:
            power_reading = (start_power + end_power) / 2.0
        elif start_power is not None:
            power_reading = start_power
        elif end_power is not None:
            power_reading = end_power

        return {
            "response": response,
            "elapsed_sec": elapsed_sec,
            "elapsed_ms": elapsed_sec * 1000.0,
            "preprocess_ms": preprocess_ms,
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "tpot_ms": tpot_ms,
            "decode_tps": decode_tps,
            "tokens": tokens,
            "throughput_tps": tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
            "ttft_source": "first-token-hook",
            "average_power_watts": power_reading,
        }
