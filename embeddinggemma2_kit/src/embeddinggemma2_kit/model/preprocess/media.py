"""Media decoding into the dict form the multimodal processor expects.

This generalises the decoding logic that the verified CPU benchmark scripts
used (``media_io.py`` in the earlier audio/video evaluation experiments). Those
scripts only ever ran one media item at a time from a manifest; here the same
rules are exposed as reusable helpers with validation.

Ground rules carried over from the verified experiments:

* Audio is always mono 16 kHz float32. ``soxr`` does the resampling at HQ.
* Video frames are chosen by the caller and passed with an explicit
  ``frames_indices`` list; the model must not resample internally.
* When a video carries audio, the audio is added AND an explicit
  ``<|video|><|audio|>`` text marker is supplied, because the processor needs
  to be told the modality order rather than inferring it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...constants import AUDIO_SAMPLE_RATE
from ...errors import MediaDecodeError

# Decoding video is CPU-heavy and not thread-safe in a useful way per stream;
# the verified experiments pinned this to 1 and used process-level parallelism.
_STREAM_THREAD_COUNT = 1


def _require_av():
    try:
        import av  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        from ...errors import MissingDependencyError

        raise MissingDependencyError("av", "media", "audio/video decoding") from exc
    import av

    return av


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        from ...errors import MissingDependencyError

        raise MissingDependencyError("soundfile", "media", "audio file decoding") from exc
    return sf


def _require_soxr():
    try:
        import soxr
    except ImportError as exc:  # pragma: no cover
        from ...errors import MissingDependencyError

        raise MissingDependencyError("soxr", "media", "audio resampling") from exc
    return soxr


def _check_media_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise MediaDecodeError(f"media file not found: {p}")
    if not p.is_file():
        raise MediaDecodeError(f"media path is not a regular file: {p}")
    return p


def decode_audio(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode an audio file to ``{'audio': {...}}`` plus an audit dict.

    Multi-channel input is downmixed by averaging -- the same rule the verified
    encoder used, so embeddings stay comparable with the recorded baselines.
    """
    p = _check_media_path(path)
    sf = _require_soundfile()
    soxr = _require_soxr()

    try:
        waveform, sample_rate = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception as exc:
        raise MediaDecodeError(f"failed to read audio {p}: {exc}") from exc

    if waveform.size == 0:
        raise MediaDecodeError(f"audio file contains no samples: {p}")

    channels = waveform.shape[1]
    waveform = waveform.mean(axis=1)
    source_rate = int(sample_rate)
    duration = float(len(waveform) / source_rate)

    if source_rate != AUDIO_SAMPLE_RATE:
        waveform = soxr.resample(waveform, source_rate, AUDIO_SAMPLE_RATE, quality="HQ")

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise MediaDecodeError(f"audio contains non-finite samples: {p}")

    data = {"audio": {"array": waveform, "sampling_rate": AUDIO_SAMPLE_RATE}}
    audit = {
        "duration": duration,
        "source_sample_rate": source_rate,
        "channels": channels,
        "samples_16k": int(len(waveform)),
    }
    return data, audit


def _select_frame_indices(total_frames: int, max_frames: int) -> list[int]:
    """Evenly spaced frame indices, inclusive of first and last frame."""
    if max_frames <= 0:
        raise MediaDecodeError(f"max_frames must be positive, got {max_frames}")
    count = min(max_frames, total_frames)
    return np.linspace(0, total_frames - 1, count, dtype=int).tolist()


def decode_video(
    path: str | Path,
    max_frames: int = 8,
    include_audio: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a video to model input, optionally with its audio track.

    Frame selection is done here, not by the processor. The decode loop asserts
    that the frames it collected are exactly the frames it asked for; a mismatch
    would silently misalign a video embedding, which is very hard to notice
    downstream, so it is treated as a hard error.
    """
    p = _check_media_path(path)
    av = _require_av()

    try:
        with av.open(str(p)) as container:
            if not container.streams.video:
                raise MediaDecodeError(f"no video stream in {p}")
            stream = container.streams.video[0]
            stream.thread_count = _STREAM_THREAD_COUNT

            fps = float(stream.average_rate)
            declared_frames = int(stream.frames)
            if declared_frames <= 0 or fps <= 0:
                raise MediaDecodeError(
                    f"video reports unusable frame count/fps ({declared_frames}, {fps}): {p}"
                )

            indices = _select_frame_indices(declared_frames, max_frames)
            wanted = set(indices)
            frames: list[np.ndarray] = []
            collected: list[int] = []

            decoded = 0
            for i, frame in enumerate(container.decode(stream)):
                decoded += 1
                if i in wanted:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    collected.append(i)

            if decoded != declared_frames:
                raise MediaDecodeError(
                    f"decoded {decoded} frames but header declared {declared_frames}: {p}"
                )
            if collected != indices:
                raise MediaDecodeError(
                    f"sampled frames {collected} do not match requested {indices}: {p}"
                )

            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else decoded / fps
            )
            audio_streams = len(container.streams.audio)
    except MediaDecodeError:
        raise
    except Exception as exc:
        raise MediaDecodeError(f"failed to decode video {p}: {exc}") from exc

    metadata = {
        "fps": fps,
        "total_num_frames": declared_frames,
        "duration": duration,
        "frames_indices": indices,
    }
    data: dict[str, Any] = {
        "video": {"array": np.stack(frames), "video_metadata": metadata}
    }
    audit: dict[str, Any] = {
        **metadata,
        "sampled_frames": len(frames),
        "audio_streams": audio_streams,
        "frame_shape": list(frames[0].shape),
    }

    if include_audio:
        if audio_streams == 0:
            raise MediaDecodeError(f"requested audio but video has no audio track: {p}")
        waveform = _decode_audio_track(p, av)
        data["audio"] = {"array": waveform, "sampling_rate": AUDIO_SAMPLE_RATE}
        # Explicit modality order. Without this marker the processor cannot know
        # whether the audio belongs to the video or is a separate input. No human
        # caption is supplied for either media modality.
        data["text"] = "<|video|><|audio|>"
        audit["audio_samples_16k"] = int(len(waveform))
        audit["audio_duration"] = float(len(waveform) / AUDIO_SAMPLE_RATE)

    return data, audit


def _decode_audio_track(path: Path, av) -> np.ndarray:
    """Decode the first audio stream of a video, resampled to mono 16 kHz."""
    chunks: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.thread_count = _STREAM_THREAD_COUNT
            resampler = av.AudioResampler(
                format="fltp", layout="mono", rate=AUDIO_SAMPLE_RATE
            )
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # Flush the resampler so the tail of the track is not dropped.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except Exception as exc:
        raise MediaDecodeError(f"failed to decode audio track of {path}: {exc}") from exc

    if not chunks:
        raise MediaDecodeError(f"video audio track produced no samples: {path}")

    waveform = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise MediaDecodeError(f"video audio track is empty or non-finite: {path}")
    return waveform