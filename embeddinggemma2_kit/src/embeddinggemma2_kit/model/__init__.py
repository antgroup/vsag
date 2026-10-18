"""Model loading and multimodal encoding."""

from .embedder import Embedder
from .prompts import ROLE_DOCUMENT, ROLE_QUERY, VALID_ROLES

__all__ = ["Embedder", "ROLE_DOCUMENT", "ROLE_QUERY", "VALID_ROLES"]
