"""Encode audio and video.

Point DATA at a directory containing .wav/.mp4 files; the datasets used by the
project's benchmarks (Clotho, MSR-VTT) work well.
"""

from pathlib import Path

from embeddinggemma2_kit import Embedder

DATA = Path("/home/zhuangye/embeddinggemma2-av-cpu-20261015/data")

audio_files = sorted(DATA.rglob("*.wav"))[:2]
video_files = sorted(DATA.rglob("*.mp4"))[:1]

if not audio_files and not video_files:
    print("no media found under", DATA)
    raise SystemExit(0)

emb = Embedder(dtype="bfloat16", num_threads=16)

if audio_files:
    vectors = emb.embed_audio(audio_files)
    print("audio:", [p.name for p in audio_files], "->", vectors.shape)

if video_files:
    # Video with its audio track: the kit adds the <|video|><|audio|> marker.
    vectors = emb.embed_video(video_files, frames=8, with_audio=True)
    print("video(+audio):", [p.name for p in video_files], "->", vectors.shape)
