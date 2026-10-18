"""Text and image preprocessing.

Text needs no transformation here: the model's own tokenizer and chat template
handle it, and the task prefix is applied via sentence-transformers prompt
names (see :mod:`embeddinggemma2_kit.model.prompts`). This module therefore
only covers images, plus the small amount of text validation that catches
mistakes early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...errors import MediaDecodeError

# Sentence-transformers accepts several image input spellings. These are the
# ones this kit supports explicitly so callers get a clear error for anything
# else instead of a confusing failure deep inside the processor.
_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        from ...errors import MissingDependencyError

        raise MissingDependencyError("pillow", "", "image decoding") from exc
    return Image


def load_image(path: str | Path):
    """Load an image file into a PIL RGB image (no resizing).

    The model's processor performs its own resizing/normalisation, so this only
    guarantees the image is RGB. Grayscale and CMYK sources are converted rather
    than rejected, since the processor expects three channels.
    """
    p = Path(path)
    if not p.exists():
        raise MediaDecodeError(f"image file not found: {p}")
    if not p.is_file():
        raise MediaDecodeError(f"image path is not a regular file: {p}")
    if p.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise MediaDecodeError(
            f"unsupported image extension '{p.suffix}' for {p}; "
            f"expected one of {sorted(_IMAGE_EXTENSIONS)}"
        )

    Image = _require_pil()
    try:
        with Image.open(p) as handle:
            handle.load()
            image = handle.convert("RGB")
    except Exception as exc:
        raise MediaDecodeError(f"failed to read image {p}: {exc}") from exc
    return image


def to_model_image(value: Any) -> Any:
    """Normalise a user-supplied image into something the processor accepts.

    Accepts a path (str/Path) or an already-loaded PIL image, which keeps the
    API convenient for callers that decode images themselves.
    """
    if isinstance(value, (str, Path)):
        return load_image(value)

    # Imported lazily so the module stays importable without Pillow.
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None  # type: ignore[assignment]

    if Image is not None and isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict) and "image" in value:
        # Already in model-input form; pass through unchanged.
        return value

    raise MediaDecodeError(
        "unsupported image input; expected a file path, a PIL.Image, or a "
        f"pre-built model-input dict, got {type(value).__name__}"
    )


def validate_texts(texts: list[str]) -> list[str]:
    """Validate a batch of texts, failing on the mistakes that matter.

    Empty strings are rejected: they still produce a well-formed embedding, so
    an accidental empty caption would otherwise enter the corpus unnoticed and
    be impossible to spot later.
    """
    if not isinstance(texts, (list, tuple)):
        raise MediaDecodeError(f"expected a list of strings, got {type(texts).__name__}")
    if not texts:
        raise MediaDecodeError("text batch is empty")

    out: list[str] = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise MediaDecodeError(
                f"text at index {i} is {type(text).__name__}, expected str"
            )
        if not text.strip():
            raise MediaDecodeError(f"text at index {i} is empty or whitespace-only")
        out.append(text)
    return out