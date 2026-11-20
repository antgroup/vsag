"""Shared helpers for the examples: device choice and corpus discovery.

Kept in one place so every example behaves the same way on every host.

Why this exists: ``Embedder`` defaults to ``device="cpu", dtype="bfloat16"``,
which is right for x86 hosts with AVX512-BF16 but wrong on Apple Silicon, where
bf16 is emulated on the CPU (~12 s per image instead of ~0.13 s on the GPU).
Hard-coding either choice makes the examples unrunnable on the other platform,
so the choice is made at runtime instead.

Override with the environment when you know better:

    EG2_DEVICE=cpu EG2_DTYPE=float32 python examples/02_embed_image.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Corpus locations are host-specific and are never redistributed with the kit.
DEFAULT_DATA = "/home/zhuangye/embeddinggemma2-cpu-20261014/data"
DEFAULT_MEDIA = "/home/zhuangye/embeddinggemma2-av-cpu-20261015/data"


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:                                 # noqa: BLE001
        return False


def pick_device() -> str:
    """Resolve the device: ``EG2_DEVICE`` > MPS if present > CPU."""
    override = os.environ.get("EG2_DEVICE")
    if override:
        return override
    return "mps" if _mps_available() else "cpu"


def pick_dtype(device: str) -> str:
    """Resolve the dtype: ``EG2_DTYPE`` > float32 on an arm64 CPU > bfloat16.

    bfloat16 needs an AVX512-BF16 style hardware path to be worth using. When
    MPS exists, the arm64 CPU is the slow fallback, so prefer float32 there.
    """
    override = os.environ.get("EG2_DTYPE")
    if override:
        return override
    if device == "cpu" and _mps_available():
        return "float32"
    return "bfloat16"


def make_embedder(**kwargs):
    """Build an ``Embedder`` with the settings this host should use."""
    from embeddinggemma2_kit import Embedder

    device = kwargs.pop("device", None) or pick_device()
    dtype = kwargs.pop("dtype", None) or pick_dtype(device)
    print(f"[device={device} dtype={dtype}]")
    return Embedder(device=device, dtype=dtype, **kwargs)


def corpus_dir() -> Path:
    """Directory holding ``manifest.json`` + ``images/``."""
    return Path(os.environ.get("EG2_DATA", DEFAULT_DATA))


def media_dir() -> Path:
    """Directory holding ``.wav`` / ``.mp4`` files for the audio/video example."""
    return Path(os.environ.get("EG2_MEDIA", DEFAULT_MEDIA))