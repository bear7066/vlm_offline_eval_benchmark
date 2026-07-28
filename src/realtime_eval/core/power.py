from __future__ import annotations

import threading
import time

from vlm_eval.hardware import get_device_vram_used_gb, get_gpu_power_watts


class DeviceSampler:
    """Background sampler for GPU power draw and device VRAM use.

    A daemon thread reads NVML power and ``mem_get_info`` at a fixed interval
    while the sampler is active, timestamping every reading so energy can be
    integrated properly rather than averaged over an unknown span.

    Both readings come from one loop because they are wanted over the same
    window and each costs microseconds; a second thread would add jitter to the
    region being measured for no benefit.

    Use it as a context manager around the work to be measured::

        with DeviceSampler(interval_sec=0.1) as sampler:
            model.generate(...)
        print(sampler.mean_watts, sampler.energy_j, sampler.peak_device_vram_gb)

    Power is whole-board, so it only means anything on an otherwise idle GPU.

    Args:
        interval_sec: Seconds between successive readings.
    """

    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval_sec = interval_sec
        self._samples: list[tuple[float, float]] = []
        self._vram_gb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            watts = get_gpu_power_watts()
            if watts is not None:
                self._samples.append((time.perf_counter(), watts))
            vram = get_device_vram_used_gb()
            if vram is not None:
                self._vram_gb.append(vram)
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "DeviceSampler":
        self._samples = []
        self._vram_gb = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Deliberately no post-hoc reading: a sample taken after the work has
        # finished measures idle decay, not the workload, and reporting it as
        # the run's power would understate a short inference. Callers should
        # read `n_power_samples` and treat a low count as untrustworthy.

    @property
    def n_power_samples(self) -> int:
        """Number of power readings collected; 0 or 1 means untrustworthy."""
        return len(self._samples)

    @property
    def energy_j(self) -> float | None:
        """Energy over the sampled window in joules, by trapezoidal integration.

        ``None`` with fewer than two samples, since a single reading spans no
        time and cannot be integrated.
        """
        if len(self._samples) < 2:
            return None
        total = 0.0
        for (t0, w0), (t1, w1) in zip(self._samples, self._samples[1:]):
            total += (w0 + w1) / 2.0 * (t1 - t0)
        return total

    @property
    def sampled_sec(self) -> float | None:
        """Span between the first and last power reading, in seconds."""
        if len(self._samples) < 2:
            return None
        return self._samples[-1][0] - self._samples[0][0]

    @property
    def mean_watts(self) -> float | None:
        """Time-weighted mean power, ``energy_j / sampled_sec``.

        Time-weighted rather than a plain mean of readings, so an irregular
        sampling interval cannot bias the result. Falls back to the single
        reading when only one was collected.
        """
        energy = self.energy_j
        span = self.sampled_sec
        if energy is not None and span is not None and span > 0:
            return energy / span
        if self._samples:
            return self._samples[0][1]
        return None

    @property
    def peak_watts(self) -> float | None:
        """Maximum power reading, or ``None`` if none were collected."""
        return max((w for _t, w in self._samples), default=None)

    @property
    def peak_device_vram_gb(self) -> float | None:
        """Highest device-wide VRAM use observed during the window."""
        return max(self._vram_gb, default=None)
