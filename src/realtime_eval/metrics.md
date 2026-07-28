# Real-time benchmark metric definitions

Every metric emitted by `realtime_eval`, with its exact formula. Defined in
`core/metrics.py`; produced by `pipeline/runner.py`; rendered by
`pipeline/analyze.py`.

**Schema version 2.** Recorded in each run's `config.json` and `summary.json`.
Results from different schema versions are not comparable, and
`result_from_dict` rejects records carrying fields from another version rather
than reinterpreting them.

---

## 1. Per-run metrics — `RealtimeResult`

One record per `(video, repeat)`. 33 fields.

### 1.1 Identity and configuration

| Field | Definition | Unit |
|---|---|---|
| `video` | Path to the source clip | — |
| `label` | Ground-truth action label | — |
| `model_id` | HuggingFace model ID | — |
| `num_frames` | Frames **requested** for this config | frames |
| `num_frames_actual` | Frames **actually fed** to the model. Lower than `num_frames` for clips with fewer frames than requested, because duplicate sample indices are dropped. Denominator of every per-frame quantity | frames |
| `max_new_tokens` | Generation cap | tokens |
| `repeat_index` | Zero-based index of the timed repeat | — |

### 1.2 Latency phases

The phases are an **exact partition** of the timed region:

```
e2e_latency_ms = preprocess_ms + prefill_ms + decode_ms
ttft_ms        = preprocess_ms + prefill_ms
```

All clocks use `time.perf_counter()` (monotonic). The timed region starts at
chat templating and ends when the last token is produced, so it covers the
per-request work a deployment actually pays for. Video decoding and frame
sampling are the caller's job and sit outside it.

| Field | Definition | Unit |
|---|---|---|
| `preprocess_ms` | Chat templating + image preprocessing + host-to-device copy | ms |
| `prefill_ms` | Preprocessed inputs → first generated token. Includes thread dispatch and the first decode step (standard TTFT convention) | ms |
| `ttft_ms` | End-to-end time to first token, `preprocess_ms + prefill_ms` | ms |
| `decode_ms` | First generated token → last token | ms |
| `e2e_latency_ms` | The whole timed region | ms |

**How the first token is timed.** `TextStreamer.put` only forwards text up to
the last space, so the first chunk a consumer pulls off the queue is empty
until a space arrives — a response beginning `"everything"` yields `""` on
token 1. Timing the first *non-empty* chunk therefore charged token 2's
arrival to TTFT. The clock is now stamped inside `put`, which runs
synchronously in the generation thread, so no detokenization or queue handoff
is folded into the number.

### 1.3 Token rates

| Field | Definition | Unit |
|---|---|---|
| `tokens` | Exact generated token count, `generated_ids.shape[-1] − prompt_len`. Includes EOS / end-of-turn. Re-encoding the decoded response is only a fallback: detokenize→retokenize is not the identity and the text has already lost special tokens and surrounding whitespace, so it undercounts | tokens |
| `tpot_ms` | Mean inter-token latency, `decode_ms / (tokens − 1)`. `decode_ms` spans exactly the `tokens − 1` gaps between the first and last token, so this is exact | ms/token |
| `decode_tps` | Decode-only rate, `(tokens − 1) / decode_sec` | tokens/s |
| `throughput_tps` | End-to-end rate, `tokens / e2e_latency_sec`. Includes prefill, so it is **not** decode speed; use `decode_tps` for that | tokens/s |

### 1.4 Provenance

| Field | Definition |
|---|---|
| `ttft_source` | How `ttft_ms` was obtained: `"first-token-hook"` (transformers backend) or `"engine-metrics"` (vLLM, measured from request arrival). **Never compare TTFT across sources without checking this** — the vLLM path does not provide the phase split above |

### 1.5 Real-time criterion

| Field | Definition | Unit |
|---|---|---|
| `window_sec` | The deployment contract: one inference per `window_sec` of incoming video, sampling `num_frames` from that window. Set it to the stride the system will really run at | s |
| **`window_rtf`** | **`e2e_latency_sec / window_sec`.** The real-time test: ≤ 1.0 means the pipeline consumes video at least as fast as it arrives. Independent of clip length | ratio |
| `max_sustainable_fps` | `num_frames_actual / e2e_latency_sec` — the input frame rate this pipeline can keep up with | frames/s |
| `rtf` | `e2e_latency_sec / video_duration_sec`, the conventional real-time factor (≤ 1.0 is faster than playback). **Reference only.** Frame count is fixed regardless of clip length, so compute per inference is constant while the denominator varies: the ratio scales as `1 / duration` and longer clips pass trivially. It describes the video set as much as the model — decide on `window_rtf` | ratio |
| `video_duration_sec` | Clip duration, `total_frames / fps` | s |

### 1.6 Power and energy

Power is **whole-board** (includes any other process on the GPU), so it is only
meaningful on an otherwise idle device. Readings come from NVML at ~0.25 ms per
call; the previous `nvidia-smi` subprocess cost ~32 ms per call, which both
capped the sampling rate and stole CPU from the region being timed.

| Field | Definition | Unit |
|---|---|---|
| `energy_j` | `∫P dt` over the inference, trapezoidal over timestamped samples. The figure that matters at the edge: a fast-but-thirsty config and a slow-but-frugal one can draw the same average watts | J |
| `power_sampled_sec` | Span the power samples cover, slightly shorter than the inference. Recorded so energy aggregates as total-energy-over-total-time | s |
| `mean_power_watts` | `energy_j / power_sampled_sec` — time-weighted, so an irregular sampling interval cannot bias it. Falls back to the lone reading when only one was collected | W |
| `peak_power_watts` | Highest power reading | W |
| `n_power_samples` | Readings collected. **Treat 0 or 1 as untrustworthy**: a short inference may not span a full sampling interval. No reading is taken after the work finishes, since that would measure idle decay rather than load | — |

### 1.7 VRAM

Both views are reported because they answer different questions.

| Field | Definition | Unit |
|---|---|---|
| `peak_vram_reserved_gb` | Peak VRAM **reserved** by the PyTorch caching allocator, summed over all visible devices. Reserved rather than allocated, so it includes fragmentation; summed because `max_memory_*` defaults to the current device and would report a single shard of a `device_map="auto"` model. Excludes the CUDA context and non-PyTorch allocations | GiB |
| `peak_vram_device_gb` | Highest device-wide use observed, `total − free` from `mem_get_info`, summed over devices. Includes the CUDA context (a few hundred MB), driver overhead and any other process. **Size deployment hardware against this** | GiB |

### 1.8 Output and status

| Field | Definition |
|---|---|
| `response` | Model output text, stripped |
| `label_word_overlap` | Whether the response contained any label content word (longer than two characters). **Not correctness** — see §2.7 |
| `status` | `"success"` or `"error"` |
| `error` | Failure message when `status == "error"` |

---

## 2. Per-config metrics — `ConfigSummary`

One record per `(model_id, num_frames, max_new_tokens)`. 32 fields. Metric
columns describe the **successful** runs; the counts in §2.1 describe every
attempt.

### 2.1 Reliability

Failed runs are excluded from the metric columns but still counted. Previously
they were dropped before grouping, so a config where four of five runs crashed
reported one clean fast run and could be recommended.

| Field | Definition |
|---|---|
| `n_attempted` | Runs started for this config, successful or not |
| `n_success` | Runs that completed |
| `n_error` | `n_attempted − n_success` |
| `success_rate` | `n_success / n_attempted` |

A config with no successful run is still emitted, with `None` metrics, rather
than disappearing from the report.

### 2.2 Configuration echo

| Field | Definition |
|---|---|
| `model_id`, `num_frames`, `max_new_tokens` | The grouping key |
| `num_frames_actual` | The one distinct actual frame count across runs, or `None` if runs disagree (which means some clips were shorter than the request) |
| `window_sec` | The stride the verdict is judged against, or `None` if runs disagree |

### 2.3 Latency

Percentiles use linear interpolation over all successful runs of the config.

| Field | Definition |
|---|---|
| `p50_e2e_latency_ms` / `p95_e2e_latency_ms` / `max_e2e_latency_ms` | Percentiles and max of `e2e_latency_ms` |
| `p50_ttft_ms` / `p95_ttft_ms` | Percentiles of `ttft_ms` — the responsiveness number |

With fewer than ~20 successful runs the interpolated p95 lands in the top
interval and is effectively the max; the report flags this and prints
`max_e2e_latency_ms` alongside so the two can be compared.

### 2.4 Real-time

| Field | Definition |
|---|---|
| `p50_window_rtf` / `p95_window_rtf` | Percentiles of `window_rtf` — the criterion |
| `p50_rtf` / `p95_rtf` | Percentiles of `rtf` — reference only, duration-confounded (§1.5) |
| `mean_max_sustainable_fps` | Mean of `max_sustainable_fps` |

### 2.5 Phase and token means

| Field | Definition |
|---|---|
| `mean_preprocess_ms` | Mean `preprocess_ms` |
| `mean_prefill_ms` | Mean `prefill_ms` |
| `mean_decode_ms` | Mean `decode_ms`. Varies with how many tokens were generated as well as how fast; use `mean_tpot_ms` for speed alone |
| `mean_tpot_ms` | Mean `tpot_ms` — per-token latency, independent of response length |
| `mean_decode_tps` | Mean `decode_tps` |
| `mean_tokens` | Mean generated tokens. Read this next to `naive_word_overlap`: it makes the verbosity confound visible |

### 2.6 Resources

| Field | Definition |
|---|---|
| `mean_energy_j` | Mean joules per inference |
| `mean_power_watts` | `Σ energy_j / Σ power_sampled_sec` across runs — total energy over total sampled time, the correctly time-weighted mean. A mean of per-run means would weight a short run the same as a long one |
| `mean_peak_vram_reserved_gb` | Mean of `peak_vram_reserved_gb` |
| `max_peak_vram_device_gb` | **Max** of `peak_vram_device_gb`. A max, not a mean: sizing hardware needs the worst case |

### 2.7 Quality

| Field | Definition |
|---|---|
| `n_videos_scored` | Videos contributing to the quality score |
| `naive_word_overlap` | Fraction of scored videos whose response contained ≥ 1 label content word |

`naive_word_overlap` is a crude proxy so the sweep is runnable without an LLM
judge. It is **not accuracy** and must not be used to rank configs:

- **It rewards verbosity.** A longer response has strictly more chances to hit
  a label word, so raising `max_new_tokens` raises the score for free — and
  `max_new_tokens` is a swept axis, so the metric is confounded with the very
  thing the sweep varies.
- **It ignores meaning.** A response naming the label while describing the
  opposite event still counts as a hit.
- **It assumes a shared vocabulary.** It is meaningless when the prompt and the
  labels use different ones, e.g. an accident-detection prompt against action
  labels.

Scoring counts each video **once** (`repeat_index == 0`). Decoding is greedy, so
repeats of a video are byte-identical and carry no extra information; counting
them inflated the denominator by a factor of `repeats` while the effective
sample size stayed at the number of videos.

Replace with the `vlm_eval.judge` pipeline before drawing any quality
conclusion.

### 2.8 The verdict

| Field | Definition |
|---|---|
| `meets_realtime_p95` | `p95_window_rtf ≤ realtime_threshold` **and** `success_rate ≥ min_success_rate` |

`realtime_threshold` defaults to 0.8, leaving headroom under the 1.0 limit.
`min_success_rate` defaults to 1.0: a config that intermittently crashes does
not meet a real-time guarantee. `None` when no run produced a `window_rtf`.

There is exactly one cutoff, applied once at p95. Per-run records carry raw
ratios only and no pass/fail flag.

**Config ranking** (`best_config`) selects, among configs with
`meets_realtime_p95`, the one with the most frames — the most temporal evidence
per inference — tie-broken by lower p95 latency. It deliberately does **not**
rank on `naive_word_overlap`, which would recommend whichever config was
allowed to talk longest. Restore quality-based ranking only once a real judge
supplies scores.

---

## 3. Frame-scaling metrics — `FrameScaling`

One record per `(model_id, max_new_tokens)`, fitted across the frame grid. 10
fields.

**Why a fit rather than a ratio.** Dividing a single TTFT by its frame count
does not give a per-frame cost: TTFT also contains the text prefill, the first
decode step and framework overhead, none of which depend on the frame count.
That fixed term makes `ttft_ms / num_frames` drift downward as frames increase,
so it can neither be compared across configs nor used to extrapolate. Fitting

```
latency(f) = slope · f + intercept
```

separates the two: the slope is the marginal cost of one more frame, the
intercept is the frame-independent floor.

Ordinary least squares, over `num_frames_actual`. Each distinct frame count is
**averaged first**, so every frame count carries equal weight regardless of how
many videos or repeats ran at it.

| Field | Definition | Unit |
|---|---|---|
| `model_id`, `max_new_tokens` | The grouping key | — |
| `n_points` | Distinct frame counts in the fit. Needs ≥ 2 for a slope to exist; ≥ 3 before `R²` carries information | — |
| `frame_counts` | The distinct actual frame counts fitted, ascending | frames |
| `ttft_ms_per_frame` | Slope of TTFT vs frames — marginal **end-to-end** cost per added frame (preprocessing + prefill) | ms/frame |
| `ttft_fixed_ms` | Intercept of the TTFT fit — end-to-end cost at zero frames | ms |
| `ttft_fit_r2` | R² of the TTFT fit. Low values mean the linear model is a poor summary and the slope must not be extrapolated | — |
| `prefill_ms_per_frame` | Slope of prefill vs frames — marginal **GPU** cost per added frame | ms/frame |
| `prefill_fixed_ms` | Intercept of the prefill fit — text prefill + first decode step + framework overhead | ms |
| `prefill_fit_r2` | R² of the prefill fit | — |

Slopes and R² are `None` when fewer than two distinct frame counts are present,
or when the frame count is constant, so a missing fit is visible rather than
silently guessed.

Comparing the two slopes answers a real deployment question: if
`ttft_ms_per_frame` markedly exceeds `prefill_ms_per_frame`, the per-frame cost
is CPU image preprocessing rather than GPU prefill, and the fix is a faster
preprocessing path, not a smaller model.

**Caution on R².** With exactly two points the fit is exact by construction and
`R² = 1.0` regardless of how well a line actually describes the relationship.
Always read `n_points` alongside it.

---

## 4. Verification

`uv run scripts/test_realtime_metrics.py` checks these definitions without a
GPU or model weights: the phase-split identities, the first-token timestamp,
slope recovery against a known linear model, duration-independence of
`window_rtf`, failure gating, energy integration, and that config ranking
ignores the quality proxy.
