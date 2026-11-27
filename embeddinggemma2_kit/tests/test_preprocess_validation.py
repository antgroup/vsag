"""Input validation for text and image preprocessing."""

from __future__ import annotations

import pytest

from embeddinggemma2_kit.errors import MediaDecodeError
from embeddinggemma2_kit.model.preprocess.text_image import (
    load_image,
    to_model_image,
    validate_texts,
)


def test_validate_texts_rejects_empty_string():
    """An empty caption would silently enter the corpus, so it is an error."""
    with pytest.raises(MediaDecodeError, match="empty"):
        validate_texts(["ok", "   "])


def test_validate_texts_rejects_non_string():
    with pytest.raises(MediaDecodeError, match="expected str"):
        validate_texts([123])


def test_validate_texts_rejects_empty_batch():
    with pytest.raises(MediaDecodeError, match="empty"):
        validate_texts([])


def test_load_image_missing_file():
    with pytest.raises(MediaDecodeError, match="not found"):
        load_image("/nonexistent/pic.jpg")


def test_load_image_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_bytes(b"not an image")
    with pytest.raises(MediaDecodeError, match="unsupported image extension"):
        load_image(path)


def test_load_image_converts_to_rgb(tmp_path):
    """Grayscale sources must be converted, since the processor wants 3 channels."""
    PIL = pytest.importorskip("PIL.Image")
    path = tmp_path / "gray.png"
    PIL.new("L", (8, 8), color=128).save(path)
    image = load_image(path)
    assert image.mode == "RGB"


def test_to_model_image_rejects_bad_type():
    with pytest.raises(MediaDecodeError, match="unsupported image input"):
        to_model_image(3.14)
