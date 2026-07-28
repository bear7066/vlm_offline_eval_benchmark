from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


DEFAULT_REALTIME_PROMPT = (
    "Detect if any accident happens, if yes, describe the main accident in one concise sentence with 3 to 5 words, otherwise, say everything is normal."
)

# The two fine-tuned candidates. gemma-3-4b is intentionally excluded: project
# memory notes it hangs on V100 and has a double-BOS issue.
DEFAULT_MODEL_IDS: tuple[str, ...] = (
    "google/gemma-4-E2B-it",
    "THChou1220/gemma-4-e2b-kinetics54K_FFT",
    "google/gemma-4-E4B-it",
    "THChou1220/gemma-4-e4b-kinetics9K_FFT",
    "THChou1220/gemma-4-e4b-kinetics54K_FFT",
)


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for a real-time benchmark sweep.

    The sweep runs the cartesian product of ``model_ids`` x
    ``num_frames_grid`` x ``max_new_tokens_grid``, timing each config with
    ``warmup`` discarded iterations followed by ``repeats`` timed iterations
    per video.

    Attributes:
        model_ids: HuggingFace model IDs to evaluate.
        num_frames_grid: Frame counts to sweep (the dominant latency lever).
        max_new_tokens_grid: Generation caps to sweep; should bracket the
            deployment target rather than the library default of 150.
        repeats: Timed iterations per video, used for percentile estimation.
        warmup: Discarded iterations per (model, num_frames) before timing,
            to absorb CUDA autotuning and lazy initialization.
        output_root: Parent directory for run outputs.
        prompt: Instruction sent with the sampled frames.
        window_sec: The deployment contract -- one inference per ``window_sec``
            of incoming video, sampling ``num_frames`` from that window. This is
            the denominator of ``window_rtf``, so it must match the stride the
            system will actually run at. Judging against clip duration instead
            would make the verdict depend on how long the benchmark clips
            happen to be, since frame count (and so compute) is fixed.
        realtime_threshold: Target p95 ``window_rtf`` a config must satisfy to
            be considered real time (0.8 leaves headroom under the 1.0 limit).
        min_success_rate: Fraction of runs that must succeed before a config can
            qualify as real time.
        power_sample_interval_sec: Sampling period for the background GPU power
            sampler.
    """

    model_ids: tuple[str, ...] = DEFAULT_MODEL_IDS
    num_frames_grid: tuple[int, ...] = (8, 12, 16)
    max_new_tokens_grid: tuple[int, ...] = (20, 40)
    repeats: int = 5
    warmup: int = 2
    output_root: Path = Path("realtime_runs")
    prompt: str = DEFAULT_REALTIME_PROMPT
    window_sec: float = 1.0
    realtime_threshold: float = 0.8
    min_success_rate: float = 1.0
    power_sample_interval_sec: float = 0.1

    def configs(self) -> list[tuple[str, int, int]]:
        """Enumerate every ``(model_id, num_frames, max_new_tokens)`` combination.

        Returns:
            A list of grid points, ordered model-major then by frame count,
            so all configs for one model run before the next model is loaded.
        """
        points: list[tuple[str, int, int]] = []
        for model_id in self.model_ids:
            for num_frames in self.num_frames_grid:
                for max_new_tokens in self.max_new_tokens_grid:
                    points.append((model_id, num_frames, max_new_tokens))
        return points

    @staticmethod
    def quick() -> "SweepConfig":
        """Return a fast smoke-test sweep (one small model, minimal grid).

        Returns:
            A :class:`SweepConfig` suited to verifying the pipeline end to end
            without running the full grid.
        """
        return SweepConfig(
            model_ids=("bear7011/gemma4-e2b-webvid4K_FT",),
            num_frames_grid=(4, 8),
            max_new_tokens_grid=(20,),
            repeats=2,
            warmup=1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SweepConfig":
        """Build a :class:`SweepConfig` from a plain dict (e.g. parsed JSON).

        Any field omitted keeps its default. The grid fields accept JSON
        arrays (converted to tuples) and ``output_root`` accepts a string
        path. Unknown keys raise :class:`ValueError` so typos in a hand-edited
        config fail loudly instead of being silently ignored.

        Args:
            data: Mapping of :class:`SweepConfig` field names to values.

        Returns:
            A :class:`SweepConfig` with the given overrides applied.

        Raises:
            ValueError: If ``data`` contains keys that are not config fields.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown SweepConfig keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}."
            )
        kwargs = dict(data)
        for key in ("model_ids", "num_frames_grid", "max_new_tokens_grid"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        if "output_root" in kwargs:
            kwargs["output_root"] = Path(kwargs["output_root"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict that roundtrips through :meth:`from_dict`.

        Tuples become lists and ``output_root`` becomes a string so the result
        can be written straight to a JSON config file.

        Returns:
            A mapping of field names to JSON-friendly values.
        """
        return {
            "model_ids": list(self.model_ids),
            "num_frames_grid": list(self.num_frames_grid),
            "max_new_tokens_grid": list(self.max_new_tokens_grid),
            "repeats": self.repeats,
            "warmup": self.warmup,
            "output_root": str(self.output_root),
            "prompt": self.prompt,
            "window_sec": self.window_sec,
            "realtime_threshold": self.realtime_threshold,
            "min_success_rate": self.min_success_rate,
            "power_sample_interval_sec": self.power_sample_interval_sec,
        }
