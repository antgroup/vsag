"""The unified multimodal embedder.

One entry point per modality plus a mixed-modality ``embed``, all returning the
same thing: a contiguous ``float32`` array of shape ``(N, 768)`` with L2-normalised
rows. Normalising here (rather than leaving it to callers) means dot product is
cosine similarity everywhere downstream, which is what the HGraph index and the
Flickr30k baselines both assume.

Two rules are enforced because violating them fails silently rather than loudly:

1. Queries and documents get different task prefixes. Callers pass ``role``, never
   a raw prompt name -- see :mod:`embeddinggemma2_kit.model.prompts`.
2. Audio/video items are encoded one at a time. The verified encoder ran media
   items with ``batch_size=1``; batching them changes padding and the resulting
   vectors, so it is not offered.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..constants import (
    AUDIO_SAMPLE_RATE,
    EMBEDDING_DIM,
    SUPPORTED_MODALITIES,
    VIDEO_PROCESSING_KWARGS,
)
from ..errors import ConfigError, KitError
from . import loader
from .preprocess import media as media_pre
from .preprocess import text_image as text_image_pre
from .prompts import ROLE_DOCUMENT, ROLE_QUERY, prompt_for_role

logger = logging.getLogger(__name__)

DEFAULT_TEXT_BATCH_SIZE = 32
DEFAULT_VIDEO_FRAMES = 8

# Bounded default for torch intra-op threads. This host is shared and often
# heavily loaded; torch would otherwise claim every visible core.
DEFAULT_CPU_THREADS = 8


class Embedder:
    """Encodes text, images, audio and video into a shared 768-d space.

    Example:
        >>> emb = Embedder(dtype="bfloat16")
        >>> emb.embed_text(["a dog on grass"], role="query").shape
        (1, 768)
        >>> emb.embed_images(["dog.jpg"]).shape
        (1, 768)

    The model is loaded lazily on first use, so constructing an ``Embedder`` is
    cheap and does not touch the checkpoint until something is actually encoded.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        text_batch_size: int = DEFAULT_TEXT_BATCH_SIZE,
        video_frames: int = DEFAULT_VIDEO_FRAMES,
        video_include_audio: bool = False,
        verify: bool = True,
        trust_remote_code: bool = False,
        num_threads: int | None = None,
    ) -> None:
        if text_batch_size <= 0:
            raise ConfigError(f"text_batch_size must be positive, got {text_batch_size}")
        if video_frames <= 0:
            raise ConfigError(f"video_frames must be positive, got {video_frames}")
        if num_threads is not None and num_threads <= 0:
            raise ConfigError(f"num_threads must be positive, got {num_threads}")

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.text_batch_size = text_batch_size
        self.video_frames = video_frames
        self.video_include_audio = video_include_audio
        self._verify = verify
        self._trust_remote_code = trust_remote_code
        self._num_threads = num_threads
        self._threads_applied = False
        self._model = None
        self._facts: dict[str, Any] | None = None

    # ------------------------------------------------------------------ model

    def _apply_thread_settings(self) -> None:
        """Bound the thread pools before the first forward pass.

        torch defaults to one thread per visible core. On a shared host that
        means a single ``encode`` call can spawn hundreds of threads and thrash
        against every other job -- the earlier CPU benchmarks on this host
        pinned 4 cores per worker for exactly this reason.

        Setting ``torch.set_num_threads`` alone is NOT sufficient: the image
        path runs through torchvision, whose native kernels honour OpenMP rather
        than torch's pool. Measured on this 192-core host, encoding one image
        spawned 335 threads and took 4.97s with only torch configured, versus
        209 threads and 1.84s once the OMP/MKL variables below were set too.
        They are therefore applied here, before any heavy work, and only when
        the caller has not already chosen values.
        """
        if self._threads_applied or self.device != "cpu":
            return
        import torch

        if self._num_threads is None:
            try:
                available = len(os.sched_getaffinity(0))
            except AttributeError:  # pragma: no cover - non-Linux
                available = os.cpu_count() or 1
            # Leave headroom on busy shared hosts rather than claiming everything.
            target = max(1, min(available, DEFAULT_CPU_THREADS))
        else:
            target = self._num_threads

        # Must be set before the native libraries initialise their pools.
        # setdefault preserves anything the caller already exported.
        os.environ.setdefault("OMP_NUM_THREADS", str(target))
        os.environ.setdefault("MKL_NUM_THREADS", str(target))
        # OpenBLAS nested inside OMP is a common oversubscription source.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        torch.set_num_threads(target)
        # Interop threads only matter for cross-op parallelism; 1 avoids
        # nested oversubscription.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can only be set before parallel work starts; harmless if already set.
            pass
        logger.info("cpu threads bounded to %d (torch/OMP/MKL), interop=1", target)
        self._threads_applied = True

    @property
    def model(self):
        """The underlying ``SentenceTransformer``, loaded on first access."""
        if self._model is None:
            self._apply_thread_settings()
            logger.info("loading EmbeddingGemma 2 checkpoint (device=%s, dtype=%s)",
                        self.device, self.dtype)
            self._model = loader.load_sentence_transformer(
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
                verify=self._verify,
                trust_remote_code=self._trust_remote_code,
            )
            self._facts = loader.verify_model(self._model, dtype=self.dtype)
        return self._model

    @property
    def facts(self) -> dict[str, Any]:
        """Verification facts about the loaded checkpoint."""
        _ = self.model
        assert self._facts is not None
        return dict(self._facts)

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    # ------------------------------------------------------------- text

    def embed_text(
        self,
        texts: Sequence[str],
        role: str = ROLE_DOCUMENT,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode text with the task prefix for ``role``.

        Args:
            texts: non-empty strings.
            role: ``"query"`` for the thing you search with, ``"document"`` for
                corpus items. Getting this wrong degrades retrieval without
                raising, which is why it is an explicit parameter.
            batch_size: overrides the instance default.

        Returns:
            ``(len(texts), 768)`` float32, L2-normalised.
        """
        validated = text_image_pre.validate_texts(list(texts))
        prompt_name = prompt_for_role(role)
        return self._encode(
            validated,
            prompt_name=prompt_name,
            batch_size=batch_size or self.text_batch_size,
        )

    # ------------------------------------------------------------ images

    def embed_images(self, images: Sequence[Any], batch_size: int | None = None) -> np.ndarray:
        """Encode images (file paths or PIL images).

        Images have no query/document asymmetry in this checkpoint, so no role
        is taken.
        """
        if not images:
            raise KitError("image batch is empty")
        prepared = [text_image_pre.to_model_image(item) for item in images]
        # Images go through the same processor as text; a modest batch is safe
        # because all items are already fixed-size pixel tensors.
        return self._encode(prepared, batch_size=batch_size or 8)

    # ------------------------------------------------------------- audio

    def embed_audio(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Encode audio files (any format soundfile can read), one at a time."""
        if not paths:
            raise KitError("audio batch is empty")
        vectors = []
        for path in paths:
            data, _audit = media_pre.decode_audio(path)
            vectors.append(self._encode([data], batch_size=1))
        return np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)

    # ------------------------------------------------------------- video

    def embed_video(
        self,
        paths: Sequence[str | Path],
        frames: int | None = None,
        with_audio: bool | None = None,
    ) -> np.ndarray:
        """Encode video files by sampling evenly spaced frames.

        Args:
            paths: video files readable by PyAV.
            frames: number of frames to sample (defaults to the instance setting).
            with_audio: include the audio track and its ``<|video|><|audio|>``
                marker. Defaults to the instance setting.
        """
        if not paths:
            raise KitError("video batch is empty")
        n_frames = frames or self.video_frames
        include_audio = self.video_include_audio if with_audio is None else with_audio

        vectors = []
        for path in paths:
            data, _audit = media_pre.decode_video(
                path, max_frames=n_frames, include_audio=include_audio
            )
            # Video kwargs must be passed per call; they are not model defaults.
            vectors.append(
                self._encode([data], batch_size=1, processing_kwargs=VIDEO_PROCESSING_KWARGS)
            )
        return np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)

    # ------------------------------------------------------------- mixed

    def embed(self, items: Sequence[dict[str, Any]], text_role: str = ROLE_DOCUMENT) -> np.ndarray:
        """Encode a mixed batch of ``{"text"|"image"|"audio"|"video": ...}`` items.

        Text items in a mixed batch use ``text_role``; a per-item ``"role"`` key
        overrides it. Media items that carry their own ``"text"`` marker (such as
        video+audio) keep it.
        """
        if not items:
            raise KitError("item batch is empty")

        vectors: list[np.ndarray] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise KitError(f"item at index {index} is {type(item).__name__}, expected dict")
            keys = [k for k in item if k in SUPPORTED_MODALITIES]
            if not keys:
                raise KitError(
                    f"item at index {index} has none of {SUPPORTED_MODALITIES}: {sorted(item)}"
                )

            # Text-only item.
            if keys == ["text"]:
                role = item.get("role", text_role)
                vectors.append(self.embed_text([item["text"]], role=role))
                continue

            # Media item: feed the dict through as-is so combinations such as
            # {"image": ..., "text": ...} keep their explicit ordering.
            payload = {k: item[k] for k in keys}
            if "image" in payload and isinstance(payload["image"], (str, Path)):
                payload["image"] = text_image_pre.load_image(payload["image"])
            vectors.append(self._encode([payload], batch_size=1))

        return np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)

    # ------------------------------------------------------------ internals

    def _encode(
        self,
        inputs: Sequence[Any],
        prompt_name: str | None = None,
        batch_size: int = 1,
        processing_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Run the encoder and enforce the output contract."""
        import torch

        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        if processing_kwargs is not None:
            kwargs["processing_kwargs"] = processing_kwargs

        try:
            with torch.inference_mode():
                raw = self.model.encode(list(inputs), **kwargs)
        except Exception as exc:
            raise KitError(f"encoding failed: {exc}") from exc

        return _finalize(raw, expected_rows=len(inputs))


def _finalize(raw: Any, expected_rows: int) -> np.ndarray:
    """Coerce encoder output to contiguous, finite, unit-norm float32.

    The model already normalises, but re-normalising is cheap and makes the
    contract hold even if a caller disables normalisation somewhere upstream.
    """
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] != expected_rows:
        raise KitError(
            f"encoder returned {matrix.shape[0]} rows for {expected_rows} inputs"
        )
    if matrix.shape[1] != EMBEDDING_DIM:
        raise KitError(
            f"encoder returned {matrix.shape[1]}-d vectors, expected {EMBEDDING_DIM}"
        )
    if not np.isfinite(matrix).all():
        raise KitError("encoder produced non-finite values")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise KitError("encoder produced a zero-norm vector")
    matrix = matrix / norms
    return np.ascontiguousarray(matrix, dtype=np.float32)


def embed_to_role(kind: str) -> str:
    """Convenience mapping for retrieval pipelines: queries vs corpus items."""
    return ROLE_QUERY if kind == "query" else ROLE_DOCUMENT