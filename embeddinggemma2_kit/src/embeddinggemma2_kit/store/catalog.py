"""Persistence of vectors plus their id/metadata mapping.

An index on its own only knows integer positions. To return anything useful --
a file path, a caption, a source id -- the vectors need to be stored alongside
the metadata that produced them, and it all has to be written with enough
provenance to know which model and preprocessing produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import BASELINE_MODEL_REVISION, EMBEDDING_DIM
from ..errors import IndexStateError
from .manifest import CorpusItem, Manifest

CATALOG_FILE = "catalog.json"
VECTORS_FILE = "vectors.npy"


@dataclass
class Catalog:
    """A corpus, its vectors, and the provenance needed to trust them."""

    ids: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    vectors: np.ndarray | None = None
    dim: int = EMBEDDING_DIM
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, item: CorpusItem, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise IndexStateError(
                f"vector dim {vec.shape[0]} does not match catalog dim {self.dim}"
            )
        self.ids.append(item.id)
        self.metadata.append(
            {"primary_modality": item.primary_modality, **item.metadata}
        )
        if self.vectors is None:
            self.vectors = vec[None, :]
        else:
            self.vectors = np.concatenate([self.vectors, vec[None, :]], axis=0)

    def subset(self, indices: np.ndarray | list[int]) -> "Catalog":
        """New catalog containing only the given row positions."""
        rows = np.asarray(indices, dtype=int)
        return Catalog(
            ids=[self.ids[int(i)] for i in rows],
            metadata=[self.metadata[int(i)] for i in rows],
            vectors=None if self.vectors is None else self.vectors[rows],
            dim=self.dim,
            provenance=dict(self.provenance),
        )

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if self.vectors is None:
            raise IndexStateError("cannot save a catalog with no vectors")
        np.save(out / VECTORS_FILE, self.vectors)
        (out / CATALOG_FILE).write_text(
            json.dumps(
                {
                    "ids": self.ids,
                    "metadata": self.metadata,
                    "dim": self.dim,
                    "count": len(self.ids),
                    "provenance": self.provenance,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Catalog":
        src = Path(directory)
        meta_path = src / CATALOG_FILE
        vec_path = src / VECTORS_FILE
        if not meta_path.exists() or not vec_path.exists():
            raise IndexStateError(f"not a saved catalog directory: {src}")
        meta = json.loads(meta_path.read_text())
        vectors = np.load(vec_path)
        if vectors.shape[0] != len(meta["ids"]):
            raise IndexStateError(
                f"{src}: {vectors.shape[0]} vectors but {len(meta['ids'])} ids"
            )
        return cls(
            ids=list(meta["ids"]),
            metadata=list(meta["metadata"]),
            vectors=vectors,
            dim=int(meta.get("dim", vectors.shape[1])),
            provenance=dict(meta.get("provenance", {})),
        )


def default_provenance(embedder_facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance block recorded alongside every saved catalog.

    Embeddings are only comparable when produced by the same model at the same
    precision; recording this makes an accidental cross-model comparison
    detectable after the fact.
    """
    import platform

    return {
        "model_revision": BASELINE_MODEL_REVISION,
        "model_facts": embedder_facts or {},
        "python": platform.python_version(),
    }


def manifest_to_catalog_inputs(manifest: Manifest):
    """Yield ``(item, model_input)`` pairs for encoding a whole manifest."""
    for item in manifest:
        yield item, item.to_model_input()