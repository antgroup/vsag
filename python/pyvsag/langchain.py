# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LangChain ``VectorStore`` adapter backed by :class:`pyvsag.Index`.

String IDs are mapped to VSAG ``int64`` IDs; text and metadata stay in Python.
Only dense ``float32`` embeddings and synchronous operations are supported
(LangChain's async methods fall back to a thread pool).

Scores are VSAG's raw distances, sorted ascending: ``l2`` is squared L2, ``ip``
is ``1 - dot``, and ``cosine`` is ``1 - cosine_similarity``.

Example::

    store = VSAGVectorStore.from_texts(["alpha", "beta"], embedding)
    store.similarity_search("alpha", k=1)[0].page_content
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised via subprocess test
    raise ImportError(
        "pyvsag.langchain requires numpy. Install it with:\n"
        '    pip install "pyvsag[langchain]"'
    ) from exc

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - exercised via subprocess test
    if exc.name is not None and not exc.name.startswith("langchain_core"):
        raise
    raise ImportError(
        "pyvsag.langchain requires langchain-core. Install it with:\n"
        '    pip install "pyvsag[langchain]"'
    ) from exc

import pyvsag

__all__ = ["VSAGVectorStore"]

logger = logging.getLogger(__name__)

_HGRAPH = "hgraph"
_METRICS = frozenset({"l2", "ip", "cosine"})
_DEFAULT_EF_SEARCH = 100

_Json = Union[str, Mapping[str, Any]]


class VSAGVectorStore(VectorStore):
    """A LangChain ``VectorStore`` backed by ``pyvsag.Index``.

    Args:
        embedding: LangChain embeddings used for documents and queries.
        index_type: VSAG index name, for example ``"hgraph"``, ``"ivf"`` or
            ``"brute_force"``.
        metric_type: One of ``"l2"``, ``"ip"`` or ``"cosine"``.
        index_param: VSAG ``index_param`` object, as a mapping or JSON object
            string. Defaults to VSAG's own defaults for ``index_type``.
        search_params: Default VSAG search parameters used by
            ``Index.knn_search``, as a mapping or JSON object string. Required
            for index types whose search parameters cannot be inferred
            (``ivf``, ``pyramid``, ...). HGraph and BruteForce have defaults.

    Notes:
        The index is created on the first non-empty :meth:`add_texts`. VSAG
        removal is tombstone based and the adapter never compacts or rebuilds;
        HGraph removal is enabled with ``support_remove``. Mutations are
        serialized, while searches release the lock before calling VSAG.
        Document mappings are not persisted.

        Search methods accept ``search_params`` and ``ef_search``; unknown
        keyword arguments are ignored for LangChain compatibility.
    """

    def __init__(
            self,
            embedding: Embeddings,
            *,
            index_type: str = "hgraph",
            metric_type: str = "l2",
            index_param: Optional[_Json] = None,
            search_params: Optional[_Json] = None,
    ) -> None:
        if not callable(getattr(embedding, "embed_documents", None)) or not callable(
                getattr(embedding, "embed_query", None)
        ):
            raise TypeError(
                "embedding must implement embed_documents() and embed_query()"
            )
        if not isinstance(index_type, str) or not index_type:
            raise ValueError("index_type must be a non-empty string")
        if metric_type not in _METRICS:
            raise ValueError(
                f"unsupported metric_type {metric_type!r}; expected one of {sorted(_METRICS)}"
            )

        index_param = (
            _coerce_json_object(index_param, "index_param")
            if index_param is not None
            else {}
        )
        search_params = (
            _coerce_json_object(search_params, "search_params")
            if search_params is not None
            else None
        )
        if index_type == _HGRAPH:
            if "support_remove" in index_param and not index_param["support_remove"]:
                raise ValueError(
                    "hgraph indexes require support_remove=true; "
                    "delete() and upsert rely on it"
                )
            index_param["support_remove"] = True
        if search_params is None and index_type not in (_HGRAPH, "brute_force"):
            raise ValueError(
                f"index_type {index_type!r} requires explicit search_params, for example "
                'search_params={"ivf": {"scan_buckets_count": 16}}'
            )

        self._embedding = embedding
        self._index_type = index_type
        self._is_hgraph = index_type == _HGRAPH
        self._metric_type = metric_type
        self._index_param = index_param
        self._search_params = search_params
        self._index: Optional[pyvsag.Index] = None
        self._dim: Optional[int] = None
        self._documents: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: Dict[int, str] = {}
        self._next_int_id = 0
        self._lock = threading.RLock()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def index(self) -> Optional[pyvsag.Index]:
        """The underlying VSAG index, or ``None`` before the first insert.

        Exposed for advanced use such as ``index.save()``. Document text,
        metadata and ID mappings are not persisted with the index.
        """
        return self._index

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Return documents for ``ids`` in input order, skipping missing IDs.

        Args:
            ids: LangChain document IDs.

        Returns:
            Documents with their ``id`` field set to the stored string ID.
        """
        documents: List[Document] = []
        with self._lock:
            for raw_id in ids:
                doc_id = str(raw_id)
                stored = self._documents.get(doc_id)
                if stored is None:
                    continue
                text, metadata = stored
                documents.append(
                    Document(id=doc_id, page_content=text, metadata=dict(metadata))
                )
        return documents

    @classmethod
    def from_texts(
            cls,
            texts: Iterable[str],
            embedding: Embeddings,
            metadatas: Optional[List[Dict[str, Any]]] = None,
            *,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> "VSAGVectorStore":
        """Create a store from texts.

        Args:
            ids: Missing values are generated with ``uuid4()``.
            **kwargs: ``VSAGVectorStore`` constructor arguments.
        """
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            *,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Embed and add texts, returning their string IDs.

        Existing IDs are overwritten: new vectors are added before the old ones
        are removed.

        Args:
            texts: Texts to embed and add.
            metadatas: Optional metadata, one per text.
            ids: Optional string IDs, one per text. Missing values are generated.
            **kwargs: Ignored for LangChain compatibility.

        Returns:
            Document IDs in input order.

        Raises:
            ValueError: If lengths differ, IDs repeat within the batch, or the
                embeddings do not match the index.
        """
        if kwargs:
            logger.debug("ignoring unsupported add_texts options: %s", sorted(kwargs))
        text_list = [str(text) for text in texts]
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError("metadatas length must match texts length")
        if ids is not None and len(ids) != len(text_list):
            raise ValueError("ids length must match texts length")
        if not text_list:
            return []

        document_ids = _normalize_ids(ids, len(text_list))
        metadata_list = [
            dict(metadatas[i]) if metadatas is not None and metadatas[i] is not None else {}
            for i in range(len(text_list))
        ]
        vectors = self._embed_documents(text_list)

        with self._lock:
            if self._dim is not None and vectors.shape[1] != self._dim:
                raise ValueError(
                    f"embedding dimension {vectors.shape[1]} does not match "
                    f"index dimension {self._dim}"
                )

            replaced_int_ids = [
                self._id_to_int[doc_id]
                for doc_id in document_ids
                if doc_id in self._id_to_int
            ]
            new_int_ids = list(
                range(self._next_int_id, self._next_int_id + len(document_ids))
            )
            # Reserve IDs before touching the index so a failed call never
            # reuses an ID that VSAG may have partially inserted.
            self._next_int_id += len(document_ids)
            int_array = np.asarray(new_int_ids, dtype=np.int64)

            if self._index is None:
                index = pyvsag.Index(
                    self._index_type, self._build_params(int(vectors.shape[1]))
                )
                index.build(vectors, int_array, len(document_ids), vectors.shape[1])
                self._index = index
                self._dim = int(vectors.shape[1])
            else:
                self._index.add(
                    vectors, int_array, len(document_ids), int(vectors.shape[1])
                )

            if replaced_int_ids and self._index is not None:
                removed = self._index.remove(
                    np.asarray(replaced_int_ids, dtype=np.int64)
                )
                if removed != len(replaced_int_ids):
                    logger.warning(
                        "VSAG removed %d of %d replaced vectors; "
                        "stale vectors are filtered from search results",
                        removed,
                        len(replaced_int_ids),
                    )
                for old_int_id in replaced_int_ids:
                    self._int_to_id.pop(old_int_id, None)

            for doc_id, int_id, text, metadata in zip(
                    document_ids, new_int_ids, text_list, metadata_list
            ):
                self._documents[doc_id] = (text, metadata)
                self._id_to_int[doc_id] = int_id
                self._int_to_id[int_id] = doc_id

        return document_ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete documents by ID.

        Unknown IDs are ignored and ``ids=None`` deletes everything. The
        adapter always drops its mappings; vectors left tombstoned by a partial
        VSAG removal are filtered from search results.

        Args:
            ids: Document IDs to delete, or ``None`` to delete all.
            **kwargs: Ignored for LangChain compatibility.

        Returns:
            ``True`` on success, including no-op deletes of unknown IDs.
        """
        if kwargs:
            logger.debug("ignoring unsupported delete options: %s", sorted(kwargs))
        with self._lock:
            target_ids = (
                list(self._documents)
                if ids is None
                else list(dict.fromkeys(str(i) for i in ids))
            )
            known_ids = [doc_id for doc_id in target_ids if doc_id in self._id_to_int]
            if not known_ids:
                return True

            if self._index is not None:
                int_ids = [self._id_to_int[doc_id] for doc_id in known_ids]
                removed = self._index.remove(np.asarray(int_ids, dtype=np.int64))
                if removed != len(int_ids):
                    logger.warning(
                        "VSAG removed %d of %d vectors; "
                        "stale vectors are filtered from search results",
                        removed,
                        len(int_ids),
                    )

            for doc_id in known_ids:
                int_id = self._id_to_int.pop(doc_id)
                self._int_to_id.pop(int_id, None)
                self._documents.pop(doc_id, None)
        return True

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return the ``k`` documents most similar to ``query``."""
        return [
            document
            for document, _ in self.similarity_search_with_score(query, k, **kwargs)
        ]

    def similarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return ``(document, distance)`` pairs for ``query``.

        Distances are VSAG's raw values: squared L2 for ``l2``, ``1 - dot`` for
        ``ip``, and ``1 - cosine_similarity`` for ``cosine``. Smaller is better;
        pairs are sorted by ascending distance.
        """
        _validate_positive_int(k, "k")
        if not self._documents:
            return []
        return self._search(self._embed_query(query), k, **kwargs)

    def similarity_search_by_vector(
            self, embedding: Sequence[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return the ``k`` documents most similar to a dense query vector."""
        return [
            document
            for document, _ in self.similarity_search_with_score_by_vector(
                embedding, k, **kwargs
            )
        ]

    def similarity_search_with_score_by_vector(
            self, embedding: Sequence[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return ``(document, distance)`` pairs for a dense query vector.

        See :meth:`similarity_search_with_score` for distance semantics.
        """
        _validate_positive_int(k, "k")
        if not self._documents:
            return []
        return self._search(self._as_vector(embedding), k, **kwargs)

    def _search(
            self, vector: np.ndarray, k: int, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        parameters = self._resolve_search_params(k, kwargs)
        with self._lock:
            index = self._index
            limit = min(k, len(self._documents))
        if index is None:
            return []

        vsag_ids, distances = index.knn_search(vector, limit, parameters)

        results: List[Tuple[Document, float]] = []
        with self._lock:
            for vsag_id, distance in zip(vsag_ids.tolist(), distances.tolist()):
                doc_id = self._int_to_id.get(int(vsag_id))
                if doc_id is None:
                    continue
                stored = self._documents.get(doc_id)
                if stored is None:
                    continue
                text, metadata = stored
                results.append(
                    (
                        Document(
                            id=doc_id, page_content=text, metadata=dict(metadata)
                        ),
                        float(distance),
                    )
                )
        return results

    def _build_params(self, dim: int) -> str:
        parameters: Dict[str, Any] = {
            "dtype": "float32",
            "metric_type": self._metric_type,
            "dim": dim,
        }
        if self._index_param:
            parameters["index_param"] = dict(self._index_param)
        return json.dumps(parameters)

    def _embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(
            self._embedding.embed_documents(list(texts)), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("embed_documents() must return one vector per text")
        vectors = np.ascontiguousarray(vectors)
        _validate_finite(vectors, "embedding")
        return vectors

    def _embed_query(self, query: str) -> np.ndarray:
        return self._as_vector(self._embedding.embed_query(query))

    def _as_vector(self, embedding: Sequence[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embedding must be a 1-D dense vector")
        vector = np.ascontiguousarray(vector)
        if self._dim is not None and vector.shape[0] != self._dim:
            raise ValueError(
                f"embedding dimension {vector.shape[0]} does not match "
                f"index dimension {self._dim}"
            )
        _validate_finite(vector, "embedding")
        return vector

    def _resolve_search_params(self, k: int, kwargs: Dict[str, Any]) -> str:
        if kwargs.get("filter"):
            raise NotImplementedError(
                "metadata filtering is not supported by VSAGVectorStore"
            )
        if kwargs.get("where_document"):
            raise NotImplementedError(
                "document content filtering is not supported by VSAGVectorStore"
            )
        ignored = set(kwargs) - {"filter", "where_document", "search_params", "ef_search"}
        if ignored:
            logger.debug("ignoring unsupported search options: %s", sorted(ignored))

        if kwargs.get("search_params") is not None:
            parameters = _coerce_json_object(
                kwargs["search_params"], "search_params"
            )
        elif self._search_params is not None:
            parameters = copy.deepcopy(self._search_params)
        elif self._is_hgraph:
            parameters = {"hgraph": {"ef_search": max(k, _DEFAULT_EF_SEARCH)}}
        else:
            parameters = {}

        ef_search = kwargs.get("ef_search")
        if self._is_hgraph:
            node = parameters.setdefault("hgraph", {})
            if not isinstance(node, dict):
                raise ValueError('search_params["hgraph"] must be an object')
            if "ef_search" not in node:
                node["ef_search"] = max(k, _DEFAULT_EF_SEARCH)
            if ef_search is not None:
                node["ef_search"] = _validate_positive_int(ef_search, "ef_search")
        elif ef_search is not None:
            raise ValueError("ef_search is only supported for hgraph indexes")

        return json.dumps(parameters)


def _coerce_json_object(value: _Json, name: str) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or JSON object string")
    return copy.deepcopy(dict(value))


def _normalize_ids(ids: Optional[List[str]], count: int) -> List[str]:
    if ids is None:
        return [str(uuid.uuid4()) for _ in range(count)]
    document_ids = [
        str(raw_id) if raw_id is not None else str(uuid.uuid4()) for raw_id in ids
    ]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("duplicate document IDs are not allowed in one batch")
    return document_ids


def _validate_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
