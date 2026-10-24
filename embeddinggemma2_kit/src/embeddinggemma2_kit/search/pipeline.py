"""Text-to-image and image-to-text retrieval.

The single most important detail here: the two sides of a cross-modal search
are encoded with *different* prompts. The query side gets the query prefix and
the corpus side gets the document prefix. This checkpoint does not raise if you
get it wrong -- it just returns worse neighbours -- so the role is set by this
module and never exposed as a raw prompt name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..constants import EMBEDDING_DIM
from ..errors import ConfigError, IndexStateError
from ..index.auto import AutoIndex
from ..index.base import SearchHit
from ..model.embedder import Embedder
from ..model.prompts import ROLE_DOCUMENT, ROLE_QUERY
from ..store.catalog import Catalog, default_provenance
from ..store.manifest import CorpusItem, Manifest

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """One retrieval hit, resolved back to its source item."""

    id: str
    score: float
    rank: int
    metadata: dict[str, Any]

    @property
    def image_path(self) -> str | None:
        return self.metadata.get("image")

    @property
    def text(self) -> str | None:
        return self.metadata.get("text")

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"Result(id={self.id!r}, score={self.score:.4f}, rank={self.rank})"


class CrossModalSearch:
    """Builds an index over a corpus and searches it from either modality.

    Both directions share one corpus encoding, so a collection only needs to be
    embedded once to support text->image and image->text.

    Example:
        >>> search = CrossModalSearch()
        >>> search.build_from_manifest(manifest)
        >>> search.search_text("a dog on grass", topk=5)
        >>> search.search_image("query.jpg", topk=5)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        dim: int = EMBEDDING_DIM,
        fallback: bool = False,
        backend: str = "auto",
        ef_search: int | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.dim = dim
        self._fallback = fallback
        self._backend = backend
        self._ef_search = ef_search
        self.catalog: Catalog | None = None
        self.index: AutoIndex | None = None

    # ------------------------------------------------------------- building

    def build_from_manifest(
        self,
        manifest: Manifest,
        batch_size: int = 32,
        progress: bool = True,
    ) -> "CrossModalSearch":
        """Embed every item in ``manifest`` and build the vector index.

        Media items are encoded one at a time (required by the verified
        encoder); text items are batched.
        """
        catalog = Catalog(dim=self.dim, provenance=default_provenance(self.embedder.facts))

        text_items = [i for i in manifest if i.primary_modality == "text"]
        media_items = [i for i in manifest if i.primary_modality != "text"]

        # Text side: these are corpus documents, so they take the document role.
        for start in range(0, len(text_items), batch_size):
            chunk = text_items[start : start + batch_size]
            vectors = self.embedder.embed_text(
                [i.text for i in chunk], role=ROLE_DOCUMENT
            )
            for item, vector in zip(chunk, vectors):
                catalog.add(item, vector)
            if progress:
                logger.info("embedded %d/%d text items", len(catalog), len(manifest))

        # Media side: one forward pass per item.
        for item in media_items:
            vector = self._embed_media_item(item)
            catalog.add(item, vector)
            if progress:
                logger.info("embedded %d/%d items", len(catalog), len(manifest))

        # Preserve manifest order so ids line up predictably.
        order = {item_id: pos for pos, item_id in enumerate(manifest.ids)}
        perm = np.argsort([order[i] for i in catalog.ids], kind="stable")
        catalog = catalog.subset(perm)

        assert catalog.vectors is not None
        self.catalog = catalog
        self.index = AutoIndex(
            dim=self.dim,
            backend=self._backend,
            fallback=self._fallback,
            expected_size=len(catalog),
            ef_search=self._ef_search,
        )
        self.index.build(catalog.vectors, catalog.ids)
        logger.info("built %s index over %d items", self.index.backend, len(catalog))
        return self

    def _embed_media_item(self, item: CorpusItem) -> np.ndarray:
        modality = item.primary_modality
        if modality == "image":
            return self.embedder.embed_images([item.image])[0]
        if modality == "audio":
            return self.embedder.embed_audio([item.audio])[0]
        if modality == "video":
            return self.embedder.embed_video([item.video])[0]
        raise ConfigError(f"unsupported media modality {modality!r} for item {item.id!r}")

    def add_items(self, items: Sequence[CorpusItem]) -> None:
        """Incrementally embed and add items to an existing index."""
        if self.catalog is None or self.index is None:
            raise IndexStateError("call build_from_manifest before add_items")

        vectors: list[np.ndarray] = []
        for item in items:
            if item.primary_modality == "text":
                vectors.append(self.embedder.embed_text([item.text], role=ROLE_DOCUMENT)[0])
            else:
                vectors.append(self._embed_media_item(item))
            self.catalog.add(item, vectors[-1])

        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        self.index.add(matrix, [i.id for i in items])

    # ------------------------------------------------------------ searching

    def _resolve(self, hits: Sequence[SearchHit]) -> list[Result]:
        assert self.catalog is not None
        positions = {item_id: pos for pos, item_id in enumerate(self.catalog.ids)}
        results = []
        for hit in hits:
            pos = positions.get(hit.id)
            meta = dict(self.catalog.metadata[pos]) if pos is not None else {}
            results.append(Result(id=hit.id, score=hit.score, rank=hit.rank, metadata=meta))
        return results

    def _require_index(self) -> AutoIndex:
        if self.index is None or self.catalog is None:
            raise IndexStateError("index is not built; call build_from_manifest first")
        return self.index

    def search_text(
        self,
        query: str | Sequence[str],
        topk: int = 10,
        modality: str | None = None,
    ) -> list[Result] | list[list[Result]]:
        """Text query -> corpus items ("text to image", and to audio/video too).

        Args:
            query: one string, or several.
            topk: number of neighbours.
            modality: restrict results to items whose primary modality matches.
                Filtering happens after search over the full index, so raise
                topk when filtering heavily.
        """
        index = self._require_index()
        single = isinstance(query, str)
        queries = [query] if single else list(query)

        # Query side uses the query prefix -- deliberately different from the
        # document prefix used when the corpus was encoded.
        vectors = self.embedder.embed_text(queries, role=ROLE_QUERY)
        raw = index.search(vectors, topk)
        out = [self._filter(self._resolve(hits), modality) for hits in raw]
        return out[0] if single else out

    def search_image(
        self,
        image: str | Path | Sequence[Any],
        topk: int = 10,
        modality: str | None = None,
    ) -> list[Result] | list[list[Result]]:
        """Image query -> corpus items ("image to text" when the corpus is text)."""
        index = self._require_index()
        single = isinstance(image, (str, Path))
        images = [image] if single else list(image)

        vectors = self.embedder.embed_images(images)
        raw = index.search(vectors, topk)
        out = [self._filter(self._resolve(hits), modality) for hits in raw]
        return out[0] if single else out

    def search_media(
        self, media: str | Path | Sequence[Any], modality: str, topk: int = 10
    ):
        """Generic media query, dispatching on the declared modality."""
        if modality == "image":
            return self.search_image(media, topk=topk)
        index = self._require_index()
        single = isinstance(media, (str, Path))
        items = [media] if single else list(media)
        if modality == "audio":
            vectors = self.embedder.embed_audio(items)
        elif modality == "video":
            vectors = self.embedder.embed_video(items)
        else:
            raise ConfigError(f"unsupported query modality {modality!r}")
        raw = index.search(vectors, topk)
        out = [self._resolve(hits) for hits in raw]
        return out[0] if single else out

    @staticmethod
    def _filter(results: list[Result], modality: str | None) -> list[Result]:
        if modality is None:
            return results
        return [r for r in results if r.metadata.get("primary_modality") == modality]

    # --------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> None:
        if self.catalog is None or self.index is None:
            raise IndexStateError("nothing to save; build the index first")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.catalog.save(out / "catalog")
        self.index.save(str(out / "index"))

    def load(self, directory: str | Path) -> "CrossModalSearch":
        """Restore a previously saved corpus and index.

        The backend is read from the saved index's own ``stats.json`` rather
        than assumed from this object's construction-time arguments. Previously
        a search object built with default settings could not reload an index
        that had been built with ``backend="numpy"`` (or vice versa), because it
        would try to re-select the backend and reject the mismatch.
        """
        src = Path(directory)
        self.catalog = Catalog.load(src / "catalog")
        self.dim = self.catalog.dim

        stats_path = src / "index" / "stats.json"
        saved_backend = None
        if stats_path.exists():
            saved_backend = json.loads(stats_path.read_text()).get("backend")

        # Map the saved backend name onto a selection this run can satisfy.
        if saved_backend == "numpy_flat":
            backend, fallback = "numpy", self._fallback
        elif saved_backend == "pyvsag_hgraph":
            # Reloading an HGraph index genuinely requires pyvsag.
            backend, fallback = "hgraph", False
        else:
            backend, fallback = self._backend, self._fallback

        self.index = AutoIndex(dim=self.dim, backend=backend, fallback=fallback)
        self.index.load(str(src / "index"))
        return self

    # ----------------------------------------------------- convenience API

    @classmethod
    def from_flickr30k(
        cls, manifest_path: str | Path, images_dir: str | Path, **kwargs: Any
    ) -> tuple["CrossModalSearch", Manifest]:
        """Load the baseline Flickr30k manifest and build the index."""
        manifest = Manifest.from_flickr30k(manifest_path, images_dir)
        search = cls(**kwargs)
        search.build_from_manifest(manifest)
        return search, manifest


class TextImageSearch(CrossModalSearch):
    """Backwards-compatible alias with a narrower name."""


class ImageTextSearch(CrossModalSearch):
    """Same object viewed from the image-query direction."""