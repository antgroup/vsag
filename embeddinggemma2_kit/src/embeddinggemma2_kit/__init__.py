"""EmbeddingGemma 2 multimodal embedding, indexing, and retrieval kit.

Quick start::

    from embeddinggemma2_kit import Embedder, AutoIndex

    emb = Embedder(dtype="bfloat16")
    vectors = emb.embed_text(["a dog on grass"], role="query")
    images = emb.embed_images(["dog.jpg"])

    index = AutoIndex(expected_size=len(images))
    index.build(images, ids)

The package is intentionally decoupled from pyvsag: importing it never requires
pyvsag, and the only module that imports pyvsag is
``embeddinggemma2_kit.index.backends.pyvsag_hgraph``.
"""

from __future__ import annotations

from .constants import EMBEDDING_DIM, SUPPORTED_MODALITIES
from .errors import (
    ConfigError,
    FinetuneError,
    IndexBackendUnavailable,
    IndexStateError,
    KitError,
    MediaDecodeError,
    MissingDependencyError,
    ModelLoadError,
)
from .index import AutoIndex, SearchHit, VectorIndex, create_index, recall_against_exact
from .model import ROLE_DOCUMENT, ROLE_QUERY, Embedder

__version__ = "0.1.0"

__all__ = [
    # core
    "Embedder",
    "AutoIndex",
    "VectorIndex",
    "SearchHit",
    "create_index",
    "recall_against_exact",
    "ROLE_QUERY",
    "ROLE_DOCUMENT",
    "EMBEDDING_DIM",
    "SUPPORTED_MODALITIES",
    # errors
    "KitError",
    "ConfigError",
    "ModelLoadError",
    "MediaDecodeError",
    "MissingDependencyError",
    "IndexBackendUnavailable",
    "IndexStateError",
    "FinetuneError",
    "__version__",
]