"""Encode audio and video.

Point DATA at a directory containing .wav/.mp4 files; the datasets used by the
project's benchmarks (Clotho, MSR-VTT) work well.
"""

from _common import make_embedder, media_dir

DATA = media_dir()    # EG2_MEDIA overrides

audio_files = sorted(DATA.rglob("*.wav"))[:2]
video_files = sorted(DATA.rglob("*.mp4"))[:1]

if not audio_files and not video_files:
    print("no media found under", DATA)
    raise SystemExit(0)

emb = make_embedder(num_threads=16)

if audio_files:
    vectors = emb.embed_audio(audio_files)
    print("audio:", [p.name for p in audio_files], "->", vectors.shape)

if video_files:
    # Video with its audio track: the kit adds the <|video|><|audio|> marker.
    vectors = emb.embed_video(video_files, frames=8, with_audio=True)
    print("video(+audio):", [p.name for p in video_files], "->", vectors.shape)
