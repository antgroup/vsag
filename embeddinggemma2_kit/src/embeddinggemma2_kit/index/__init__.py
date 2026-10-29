"""Vector index backends and automatic construction."""

from .auto import AutoIndex, create_index, recall_against_exact
from .base import IndexStats, SearchHit, VectorIndex

__all__ = [
    "AutoIndex",
    "create_index",
    "recall_against_exact",
    "IndexStats",
    "SearchHit",
    "VectorIndex",
]
