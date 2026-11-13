"""Checkpoint loading and verification.

Loads the EmbeddingGemma 2 sentence-transformers checkpoint on CPU and asserts
that what came back is the model we expect. The verification is deliberately
strict: this repository's retrieval baselines (Flickr30k text->image hit@1 =
0.8632, image->text hit@1 = 0.9440) were produced by the 744M multimodal
checkpoint, and a partial or wrong checkpoint would still produce plausible
768-d vectors while quietly invalidating every comparison against them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..constants import EMBEDDING_DIM, EXPECTED_PARAMETER_COUNT
from ..errors import ConfigError, ModelLoadError

logger = logging.getLogger(__name__)

_VALID_DTYPES = ("bfloat16", "float32", "float16")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        from ..errors import MissingDependencyError

        raise MissingDependencyError("torch", "", "running the model") from exc
    return torch


def resolve_model_path(model_path: str | Path | None) -> Path:
    """Resolve a checkpoint directory, defaulting to ``weights/embeddinggemma-2``.

    The default is intentionally a symlink created by
    ``scripts/fetch_weights.py``; weights are never bundled with the package.
    """
    if model_path is None:
        # .../src/embeddinggemma2_kit/model/loader.py -> kit root
        kit_root = Path(__file__).resolve().parents[3]
        model_path = kit_root / "weights" / "embeddinggemma-2"

    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(
            f"checkpoint directory not found: {path}\n"
            "Provision it first, e.g.: python scripts/fetch_weights.py "
            "--src /path/to/embeddinggemma-2"
        )
    if not path.is_dir():
        raise ModelLoadError(f"checkpoint path is not a directory: {path}")
    for required in ("config.json", "model.safetensors"):
        if not (path / required).exists():
            raise ModelLoadError(f"checkpoint at {path} is missing {required}")
    return path


def load_sentence_transformer(
    model_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "bfloat16",
    verify: bool = True,
    trust_remote_code: bool = False,
):
    """Load and verify the EmbeddingGemma 2 encoder.

    Args:
        model_path: checkpoint directory; defaults to ``weights/embeddinggemma-2``.
        device: torch device string. The verified baselines are CPU.
        dtype: ``bfloat16`` (verified CPU default), ``float32``, or ``float16``.
            Note that the model does not support FP16 activations.
        verify: assert parameter count and multimodal towers are present.
        trust_remote_code: forwarded to transformers; off by default since the
            verified environment runs a pinned transformers source checkout.

    Returns:
        A ``SentenceTransformer`` in eval mode with inference-only settings.
    """
    if dtype not in _VALID_DTYPES:
        raise ConfigError(f"dtype must be one of {_VALID_DTYPES}, got {dtype!r}")
    if dtype == "float16":
        logger.warning(
            "float16 is not supported for EmbeddingGemma 2 activations; "
            "use bfloat16 (verified CPU default) or float32."
        )

    torch = _require_torch()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        from ..errors import MissingDependencyError

        raise MissingDependencyError(
            "sentence-transformers", "", "loading the model"
        ) from exc

    path = resolve_model_path(model_path)
    torch_dtype = getattr(torch, dtype)

    try:
        model = SentenceTransformer(
            str(path),
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ModelLoadError(f"failed to load checkpoint from {path}: {exc}") from exc

    model.eval()
    if verify:
        verify_model(model, dtype=dtype)
    return model


def verify_model(model: Any, dtype: str | None = None) -> dict[str, Any]:
    """Assert the loaded model matches the expected multimodal checkpoint.

    Returns a small dict of facts suitable for logging or writing into index
    metadata.

    Raises:
        ModelLoadError: if the parameter count is wrong (wrong or partial
            checkpoint) or a modality tower is missing (text-only checkpoint).
    """
    torch = _require_torch()

    total = sum(p.numel() for p in model.parameters())
    if total != EXPECTED_PARAMETER_COUNT:
        raise ModelLoadError(
            f"parameter count mismatch: expected {EXPECTED_PARAMETER_COUNT:,} "
            f"for the 744M multimodal checkpoint, got {total:,}. "
            "A text-only or partially downloaded checkpoint would still produce "
            "768-d vectors, so this is treated as fatal."
        )

    if dtype is not None:
        expected = getattr(torch, dtype)
        wrong = {str(p.dtype) for p in model.parameters() if p.dtype != expected}
        if wrong:
            raise ModelLoadError(
                f"model parameters dtype {sorted(wrong)} != requested {dtype}"
            )

    # The towers live on the core model inside the sentence-transformers wrapper.
    try:
        core = model[0].auto_model
    except Exception as exc:
        raise ModelLoadError(f"unexpected SentenceTransformer layout: {exc}") from exc

    missing = [
        name
        for name in ("audio_tower", "vision_tower")
        if getattr(core, name, None) is None
    ]
    if missing:
        raise ModelLoadError(
            f"checkpoint is missing modality tower(s): {', '.join(missing)}. "
            "Cross-modal search requires the full multimodal checkpoint."
        )

    return {
        "parameters": total,
        "embedding_dim": EMBEDDING_DIM,
        "device": str(model.device),
        "dtype": dtype,
        "max_seq_length": getattr(model, "max_seq_length", None),
    }