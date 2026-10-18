"""Cross-modal retrieval pipelines."""

from .pipeline import (
    CrossModalSearch,
    ImageTextSearch,
    Result,
    TextImageSearch,
)

__all__ = [
    "CrossModalSearch",
    "TextImageSearch",
    "ImageTextSearch",
    "Result",
]
