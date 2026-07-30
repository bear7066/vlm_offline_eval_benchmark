from __future__ import annotations

import logging
import math
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".mkv")


def get_video_duration(total_frames: int, fps: float | None) -> float | None:
    """Compute video duration in seconds from a frame count and FPS.

    Args:
        total_frames: Number of frames in the video.
        fps: Average frames per second, or ``None``/0 if unavailable.

    Returns:
        Duration in seconds, or ``None`` if FPS is unavailable or zero.
    """
    if fps and fps > 0:
        return total_frames / fps
    return None


def frame_span(
    start_sec: float, end_sec: float, fps: float, total_frames: int
) -> tuple[int, int]:
    """Frame indices covered by the half-open time interval ``[start, end)``.

    Frame ``k`` sits at ``k / fps``, so it falls in the interval when
    ``start * fps <= k < end * fps``. Half-open so that consecutive windows
    tile a clip without sharing a frame.

    Args:
        start_sec: Interval start, inclusive.
        end_sec: Interval end, exclusive.
        fps: Frames per second of the source.
        total_frames: Frame count, used to clamp the upper bound.

    Returns:
        ``(first, last)`` inclusive frame indices. ``last < first`` when the
        interval contains no frame.
    """
    first = max(0, math.ceil(start_sec * fps))
    last = min(total_frames - 1, math.ceil(end_sec * fps) - 1)
    return first, last


def probe_video(video_path: Path) -> tuple[int | None, float | None, float | None]:
    """Read a clip's frame count, FPS and duration without decoding any frame.

    Cheap enough to call for every clip before a sweep starts: it opens the
    container and reads header properties only.

    Args:
        video_path: Path to the video file.

    Returns:
        ``(total_frames, fps, duration_sec)``, each ``None`` when unavailable.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, None, None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        duration = get_video_duration(total_frames, fps) if total_frames else None
        return total_frames, fps, duration
    finally:
        cap.release()


def sample_frames(
    video_path: Path,
    num_frames: int = 8,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    """Sample frames evenly from a video file, or from a time window of one.

    Uses OpenCV for I/O and numpy to compute uniformly spaced indices across
    the requested span. With no ``start_sec``/``end_sec`` the span is the whole
    clip, which is the historical behaviour.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample. Must be > 0.
        start_sec: Window start, inclusive. ``None`` means the clip start.
        end_sec: Window end, exclusive. ``None`` means the clip end.

    Returns:
        A 4-tuple of ``(pil_frames, video_duration_sec, total_frames,
        original_fps)``.  All elements are ``None`` when the video
        cannot be opened, contains no frames, *num_frames* is invalid, or the
        requested window contains no frame.

        - **pil_frames** (list[PIL.Image.Image] | None): Sampled frames
          as PIL images.
        - **video_duration_sec** (float | None): Duration of the **whole clip**
          in seconds, not of the requested window.
        - **total_frames** (int | None): Frame count reported by OpenCV.
        - **original_fps** (float | None): Average FPS reported by OpenCV.
    """
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Could not read video %s", video_path)
        cap.release()
        return None, None, None, None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or None
        video_duration_sec = get_video_duration(total_frames, original_fps)

        if total_frames == 0:
            return None, None, None, None

        if num_frames <= 0:
            logging.error("num_frames must be > 0, got %s", num_frames)
            return None, None, None, None

        first, last = 0, total_frames - 1
        if start_sec is not None or end_sec is not None:
            if not original_fps:
                logging.error("Windowed sampling needs FPS, unavailable for %s", video_path)
                return None, None, None, None
            window_start = 0.0 if start_sec is None else start_sec
            window_end = (
                total_frames / original_fps if end_sec is None else end_sec
            )
            if window_end <= window_start:
                logging.error(
                    "Empty window [%s, %s) for %s", window_start, window_end, video_path
                )
                return None, None, None, None
            first, last = frame_span(window_start, window_end, original_fps, total_frames)
            if last < first:
                logging.warning(
                    "Window [%.3f, %.3f) of %s contains no frame",
                    window_start,
                    window_end,
                    video_path,
                )
                return None, None, None, None

        # Pick num_frames positions spread evenly across the span, endpoints
        # included. e.g. frames 0..99 with num_frames=4 → [0, 33, 66, 99]
        indices = np.linspace(first, last, num_frames, dtype=int)
        # Guard against floating-point rounding pushing an index out of bounds.
        indices = np.clip(indices, first, last)
        # When num_frames > total_frames, linspace produces duplicates; drop them
        # so we never decode the same frame twice.
        indices = np.unique(indices)

        # Shouldn't happen with valid inputs, but fall back to the first frame
        # of the span rather than crashing.
        if len(indices) == 0:
            indices = np.array([first], dtype=int)

        # Seek to each index and decode. OpenCV reads BGR, so convert to RGB
        # before handing PIL images to the model processors.
        pil_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                logging.warning("Failed to read frame %s from %s", idx, video_path)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            return None, None, None, None
    finally:
        cap.release()

    return pil_frames, video_duration_sec, total_frames, original_fps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample frames from a video and save them as images for inspection."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video file.")
    parser.add_argument("--num_frames", "-n", type=int, default=8, help="Number of frames to sample.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to write sampled frames to (default: <video_stem>_frames next to the video).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.video_path.with_name(f"{args.video_path.stem}_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_frames, duration_sec, total_frames, fps = sample_frames(args.video_path, args.num_frames)
    if pil_frames is None:
        raise SystemExit(f"Failed to sample frames from {args.video_path}")

    print(f"video: {args.video_path}")
    print(f"total_frames: {total_frames}, fps: {fps}, duration_sec: {duration_sec}")
    print(f"sampled {len(pil_frames)} frame(s) -> {output_dir}")

    for i, frame in enumerate(pil_frames):
        frame_path = output_dir / f"frame_{i:03d}.png"
        frame.save(frame_path)
        print(f"  saved {frame_path}")
