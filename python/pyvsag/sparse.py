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

"""Helpers for working with sparse vectors.

The module provides two groups of helpers:

* :func:`build_term_vocabulary` and :func:`term_weights_to_csr` convert
  human-readable term-weight mappings (``{term: weight}``) into the
  ``uint32`` / ``float32`` CSR arrays accepted by pyvsag's sparse indexes
  such as SINDI.
* :func:`load_sparse_hdf5` and :func:`decode_sparse_bytes` read sparse
  datasets produced by the ``eval_performance`` C++ tool.

The eval tool stores sparse vectors in a custom layout that does **not**
match the typical ``(N, D)`` HDF5 dense matrix layout. This module decodes
that layout into Python-friendly structures.

On-disk layout (see ``tools/eval/eval_dataset.cpp`` for the reference
implementation):

* File-level attribute ``type``:
    - ``"dense"`` (or attribute missing, for compatibility with public
      datasets): ``/train`` and ``/test`` are ``(N, D)`` matrices of
      ``INT8`` or ``FLOAT32``.
    - ``"sparse"``: ``/train`` and ``/test`` are one-dimensional
      ``INT8`` (``H5T_INTEGER`` of size 1) datasets that hold a raw byte
      stream. The stored ``int8`` dtype is a transport detail; the bytes
      are actually a packed sequence of records:

        ``uint32 len | uint32 ids[len] | float32 vals[len]``

      concatenated back-to-back, one record per vector. ``len == 0`` is
      allowed and occupies only the 4-byte length field.

* File-level attribute ``distance``: for sparse datasets only ``"ip"`` is
  supported by the C++ eval tool; the value here is returned as-is.

The decoder verifies that the byte stream is fully consumed; a malformed
file raises :class:`ValueError`.
"""

from __future__ import annotations

import math
import operator
import struct
from collections.abc import Mapping as MappingABC
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

__all__ = [
    "build_term_vocabulary",
    "TermVocabulary",
    "term_weights_to_csr",
    "load_sparse_hdf5",
    "decode_sparse_bytes",
]

# Term IDs and CSR offsets are transported as uint32, so every identifier
# must fit in this range.
_UINT32_MAX = (1 << 32) - 1
_MAX_TERM_COUNT = 1 << 32
_FLOAT32_MAX = float(np.finfo(np.float32).max)

# Each record header is a single little-endian uint32 storing the number
# of non-zero entries in the vector.
_LEN_STRUCT = struct.Struct("<I")
_LEN_SIZE = _LEN_STRUCT.size  # 4


def build_term_vocabulary(
    documents: Sequence[Mapping[str, float]],
) -> Dict[str, int]:
    """Assign a stable ``uint32``-compatible ID to every term.

    Terms are sorted lexicographically and numbered from ``0``, so the
    result is deterministic and does not depend on dict insertion order or
    on the order of ``documents``. The returned mapping can be passed to
    :func:`term_weights_to_csr` for both documents and queries.

    Args:
        documents: Sequence (or iterable) of term-weight mappings, for
            example ``[{"vector": 1.0, "search": 0.8}]``. Only the keys
            are inspected; weights may be any numeric value.

    Returns:
        A ``dict[str, int]`` whose IDs are contiguous from ``0`` to
        ``len(vocabulary) - 1``. An empty input yields an empty dict.

        Wrap the result in :class:`TermVocabulary` when the same vocabulary
        is reused for many conversions (for example, converting queries
        against a large corpus vocabulary): the wrapper validates the IDs
        once, so each later :func:`term_weights_to_csr` call only pays for
        the terms it actually converts.

    Raises:
        TypeError: If a document is not a mapping or a term is not a
            string.
        ValueError: If the number of distinct terms exceeds the ``uint32``
            ID space.

    Example:
        >>> from pyvsag.sparse import build_term_vocabulary
        >>> build_term_vocabulary([{"b": 1.0, "a": 0.5}])
        {'a': 0, 'b': 1}
    """
    terms: set[str] = set()
    for position, document in enumerate(documents):
        if not isinstance(document, MappingABC):
            raise TypeError(
                f"documents[{position}] must be a mapping of str to numeric "
                f"weight, got {type(document).__name__}"
            )
        for term in document:
            if not isinstance(term, str):
                raise TypeError(
                    f"documents[{position}] contains a non-string term "
                    f"{term!r} ({type(term).__name__})"
                )
            terms.add(term)

    if len(terms) > _MAX_TERM_COUNT:
        raise ValueError(
            f"vocabulary has {len(terms)} terms, exceeding the uint32 term "
            f"ID space (maximum {_MAX_TERM_COUNT})"
        )

    return {term: index for index, term in enumerate(sorted(terms))}


def _validated_term_ids(vocabulary: Mapping[str, int]) -> Dict[str, int]:
    """Validate ``vocabulary`` once and return its ``term -> ID`` mapping."""
    if not isinstance(vocabulary, MappingABC):
        raise TypeError(
            "vocabulary must be a mapping of str to uint32-compatible integer, "
            f"got {type(vocabulary).__name__}"
        )

    term_to_id: Dict[str, int] = {}
    id_to_term: Dict[int, str] = {}
    for term, raw_id in vocabulary.items():
        if not isinstance(term, str):
            raise TypeError(
                f"vocabulary contains a non-string term {term!r} "
                f"({type(term).__name__})"
            )
        # Detect duplicates before normalizing, so equal-but-differently
        # typed IDs (``True == 1``, ``1.0 == 1``) are still reported as
        # duplicates. Values that cannot be hashed fall through to
        # _validate_term_id, which raises the descriptive TypeError.
        try:
            previous = id_to_term.get(raw_id)
        except TypeError:
            previous = None
        if previous is not None:
            raise ValueError(
                f"vocabulary has duplicate term ID {raw_id!r} for terms "
                f"{term!r} and {previous!r}"
            )
        term_id = _validate_term_id(term, raw_id)
        id_to_term[term_id] = term
        term_to_id[term] = term_id

    return term_to_id


class TermVocabulary(MappingABC):
    """A validated, reusable ``term -> uint32 ID`` mapping.

    :func:`term_weights_to_csr` validates vocabulary IDs on every call, so
    passing a raw mapping would rescan the whole vocabulary for each query.
    Building a :class:`TermVocabulary` once validates the IDs a single time
    and lets every later conversion cost only the terms in the converted
    row. This matters for query-time conversion against a large corpus
    vocabulary, where a one-term query would otherwise iterate over the
    entire vocabulary.

    The instance is a read-only :class:`collections.abc.Mapping`, so it can
    be iterated, compared with a ``dict``, and passed anywhere a
    ``str -> int`` mapping is expected.

    Args:
        vocabulary: Mapping returned by :func:`build_term_vocabulary` (or
            an equivalent ``str -> int`` mapping). Term IDs must be unique
            and fit in ``uint32``.

    Raises:
        TypeError: If ``vocabulary`` is not a mapping, a term is not a
            string, or an ID is not an integer (``bool`` included).
        ValueError: If an ID is outside the ``uint32`` range or two terms
            share the same ID.

            Duplicate IDs are detected before an ID is type-checked, so a
            value that compares equal to an earlier ID (``1 == 1.0 ==
            True``) raises ``ValueError`` rather than ``TypeError``.

    Example:
        >>> from pyvsag.sparse import TermVocabulary, build_term_vocabulary
        >>> vocabulary = TermVocabulary(
        ...     build_term_vocabulary([{"b": 1.0, "a": 0.5}])
        ... )
        >>> vocabulary["a"], len(vocabulary)
        (0, 2)
    """

    __slots__ = ("_term_to_id",)

    def __init__(self, vocabulary: Mapping[str, int]) -> None:
        self._term_to_id = _validated_term_ids(vocabulary)

    def __getitem__(self, term: str) -> int:
        return self._term_to_id[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._term_to_id)

    def __len__(self) -> int:
        return len(self._term_to_id)

    def __repr__(self) -> str:
        return f"TermVocabulary({self._term_to_id!r})"


def term_weights_to_csr(
    documents: Iterable[Mapping[str, float]],
    vocabulary: Mapping[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert term-weight mappings into pyvsag sparse CSR arrays.

    Each document becomes one CSR row. Terms are translated to their ID in
    ``vocabulary`` and sorted by ID, so the output is deterministic even
    when the input mappings have different insertion orders.

    Args:
        documents: Iterable (or sequence) of ``{term: weight}`` mappings.
            Empty mappings are rejected with ``ValueError`` (see the
            ``Raises`` section).
        vocabulary: Mapping returned by :func:`build_term_vocabulary` (or
            an equivalent ``str -> int`` mapping). Every term used by a
            document must be present; IDs must fit in ``uint32``. IDs must
            be unique — duplicate IDs in the vocabulary will cause a
            ``ValueError``. Pass a :class:`TermVocabulary` to validate the
            IDs once and reuse them across calls, which keeps query-time
            conversion proportional to the query size instead of the
            vocabulary size.

    Returns:
        A tuple ``(index_pointers, indices, values)``:

        * ``index_pointers``: ``uint32`` array of shape ``(num_documents + 1,)``
          with the start/end offsets of each row.
        * ``indices``: ``uint32`` array of shape ``(nnz,)`` with the term
          IDs, ordered within each row.
        * ``values``: ``float32`` array of shape ``(nnz,)`` with the
          corresponding weights.

        The returned arrays can be passed directly to
        ``index.build(index_pointers=..., indices=..., values=..., ids=...)``
        and ``index.knn_search(...)``. No row is ever empty: SINDI silently
        drops empty documents on ``add`` and rejects empty queries in
        ``knn_search``, so empty mappings are rejected above.

    Raises:
        TypeError: If a document or the vocabulary is not a mapping, a
            term is not a string, a vocabulary ID is not an integer, or a
            weight is not numeric.
        KeyError: If a document uses a term absent from ``vocabulary``.
        ValueError: If a vocabulary ID is outside the ``uint32`` range, a
            weight is non-finite or does not fit in ``float32``, the
            total number of stored terms exceeds the ``uint32`` range,
            the vocabulary contains duplicate term IDs (detected before
            the integer check, so ``1`` and ``1.0`` collide as duplicates),
            or a document is empty (empty documents are not supported by
            SINDI).

    Example:
        >>> from pyvsag.sparse import (
        ...     TermVocabulary,
        ...     build_term_vocabulary,
        ...     term_weights_to_csr,
        ... )
        >>> documents = [{"vector": 1.0}, {"search": 0.5, "vector": 0.8}]
        >>> vocabulary = TermVocabulary(build_term_vocabulary(documents))
        >>> index_pointers, indices, values = term_weights_to_csr(
        ...     documents, vocabulary
        ... )
        >>> index_pointers.tolist(), indices.tolist()
        ([0, 1, 3], [1, 0, 1])
        >>> [round(float(weight), 2) for weight in values]
        [1.0, 0.5, 0.8]
    """
    # A TermVocabulary has already validated its IDs; anything else is
    # validated once here so the per-document loop stays cheap.
    if isinstance(vocabulary, TermVocabulary):
        term_to_id: Mapping[str, int] = vocabulary
    else:
        term_to_id = _validated_term_ids(vocabulary)

    index_pointers = [0]
    indices: List[int] = []
    values: List[float] = []

    for position, document in enumerate(documents):
        if not isinstance(document, MappingABC):
            raise TypeError(
                f"documents[{position}] must be a mapping of str to numeric "
                f"weight, got {type(document).__name__}"
            )

        # Reject empty documents - SINDI::add skips empty documents
        # and SINDI::KnnSearch rejects empty queries
        if len(document) == 0:
            raise ValueError(
                f"documents[{position}] is empty; empty documents are not "
                f"supported by SINDI (they are silently dropped during "
                f"add and rejected by knn_search)"
            )

        row = []
        for term, weight in document.items():
            if not isinstance(term, str):
                raise TypeError(
                    f"documents[{position}] contains a non-string term "
                    f"{term!r} ({type(term).__name__})"
                )
            try:
                term_id = term_to_id[term]
            except KeyError:
                raise KeyError(
                    f"term {term!r} in documents[{position}] is missing from "
                    "the vocabulary"
                ) from None
            row.append((term_id, _validate_weight(term, weight)))

        # Sort by ID so rows are deterministic. Duplicate IDs are rejected
        # above, so every ID in a row is unique.
        row.sort()
        for term_id, weight in row:
            indices.append(term_id)
            values.append(weight)
        index_pointers.append(len(indices))

    if len(indices) > _UINT32_MAX:
        raise ValueError(
            f"CSR has {len(indices)} stored terms, exceeding the uint32 "
            f"offset space (maximum {_UINT32_MAX})"
        )

    return (
        np.asarray(index_pointers, dtype=np.uint32),
        np.asarray(indices, dtype=np.uint32),
        np.asarray(values, dtype=np.float32),
    )


def _validate_term_id(term: str, value: Any) -> int:
    """Return ``value`` as an int, rejecting IDs outside the uint32 range."""
    if isinstance(value, bool):
        raise TypeError(
            f"vocabulary ID for term {term!r} must be an integer, not bool"
        )
    try:
        term_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"vocabulary ID for term {term!r} must be an integer, got "
            f"{type(value).__name__}"
        ) from exc

    if term_id < 0 or term_id > _UINT32_MAX:
        raise ValueError(
            f"vocabulary ID for term {term!r} is {term_id}, outside the "
            f"uint32 range [0, {_UINT32_MAX}]"
        )
    return term_id


def _validate_weight(term: str, value: Any) -> float:
    """Return ``value`` as a finite float that fits in float32."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"weight for term {term!r} must be numeric, got "
            f"{type(value).__name__}"
        )
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"weight for term {term!r} must be numeric, got "
            f"{type(value).__name__}"
        ) from exc
    except OverflowError as exc:
        raise ValueError(
            f"weight for term {term!r} does not fit in float32: {value!r}"
        ) from exc

    if not math.isfinite(weight):
        raise ValueError(
            f"weight for term {term!r} must be finite, got {weight!r}"
        )
    if weight < -_FLOAT32_MAX or weight > _FLOAT32_MAX:
        raise ValueError(
            f"weight for term {term!r} does not fit in float32: {weight!r}"
        )
    return weight


def decode_sparse_bytes(buffer: Any) -> List[Dict[int, float]]:
    """Decode a raw eval-format sparse byte stream into a list of dicts.

    Args:
        buffer: Any object that supports the buffer protocol and exposes
            a contiguous byte view (``bytes``, ``bytearray``,
            ``memoryview``, or a contiguous 1-D ``numpy.ndarray``). The
            buffer is reinterpreted as raw bytes via
            ``memoryview(...).cast("B")`` and read without copying.

    Returns:
        A list where each element is ``{feature_index: value}`` for one
        sparse vector. Keys are returned in ascending order (the eval
        tool also stores them ordered after parsing).

    Raises:
        ValueError: If the byte stream is truncated or has trailing bytes
            that cannot be parsed as a complete record, or if ``buffer``
            cannot be cast to a contiguous byte view.
    """
    try:
        # Force a byte-granular view: for ndarrays / typed memoryviews
        # whose ``itemsize`` is > 1, ``len(view)`` would otherwise be an
        # element count rather than a byte count, which would mis-parse
        # the stream.
        view = memoryview(buffer).cast("B")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "decode_sparse_bytes requires a contiguous byte buffer "
            "(bytes, bytearray, or a contiguous int8/uint8 ndarray)"
        ) from exc
    total = view.nbytes
    pos = 0
    result: List[Dict[int, float]] = []

    while pos < total:
        if pos + _LEN_SIZE > total:
            raise ValueError(
                f"sparse stream truncated at offset {pos}: expected 4-byte length header"
            )
        (length,) = _LEN_STRUCT.unpack_from(view, pos)
        pos += _LEN_SIZE

        if length == 0:
            result.append({})
            continue

        ids_size = length * 4
        vals_size = length * 4
        if pos + ids_size + vals_size > total:
            raise ValueError(
                f"sparse stream truncated at offset {pos}: "
                f"expected {ids_size + vals_size} bytes for len={length}"
            )

        # Use explicit little-endian dtypes so the decode is correct on
        # big-endian hosts (``np.uint32`` / ``np.float32`` are native-
        # endian, while the on-disk layout is always little-endian).
        ids = np.frombuffer(view, dtype="<u4", count=length, offset=pos)
        pos += ids_size
        vals = np.frombuffer(view, dtype="<f4", count=length, offset=pos)
        pos += vals_size

        # Build dict; sorting by key keeps the output stable regardless
        # of input order (the C++ side also sorts on load).
        is_sorted = length <= 1 or bool(np.all(ids[:-1] <= ids[1:]))
        if is_sorted:
            result.append({int(k): float(v) for k, v in zip(ids, vals)})
        else:
            order = np.argsort(ids, kind="stable")
            result.append({int(ids[i]): float(vals[i]) for i in order})

    return result


def load_sparse_hdf5(
    path: str,
    splits: Sequence[str] = ("train", "test"),
    *,
    require_sparse: bool = True,
) -> Dict[str, Any]:
    """Load an eval-format HDF5 file containing sparse vectors.

    Args:
        path: Path to the ``.hdf5`` file produced by the eval tool.
        splits: Which sparse splits to decode. Defaults to both
            ``"train"`` and ``"test"``. Pass e.g. ``("test",)`` to skip
            decoding the (usually much larger) base set.
        require_sparse: If ``True`` (default), raise :class:`ValueError`
            when the file is not marked as sparse. Set to ``False`` to
            allow loading dense files (in which case the requested
            ``splits`` are returned as numpy arrays instead of dict
            lists).

    Returns:
        A dict with the following keys:

        * ``"type"``: ``"sparse"`` or ``"dense"`` (the value of the file's
          ``type`` attribute; ``"dense"`` is also returned when the
          attribute is absent, for backwards compatibility).
        * ``"distance"``: value of the ``distance`` attribute, or
          ``None`` if missing.
        * For each requested split (e.g. ``"train"``, ``"test"``): a
          ``list[dict[int, float]]`` for sparse files, or a
          ``numpy.ndarray`` for dense files (when
          ``require_sparse=False``).
        * ``"neighbors"``: ``numpy.ndarray`` of shape ``(Q, K)``,
          ``int64``, if the dataset exists.
        * ``"distances"``: ``numpy.ndarray`` of shape ``(Q, K)``,
          ``float32``, if the dataset exists.

    Raises:
        ImportError: If ``h5py`` is not installed.
        ValueError: If the file is not sparse and ``require_sparse`` is
            ``True``, or if a sparse byte stream is malformed.

    Example:
        >>> from pyvsag.sparse import load_sparse_hdf5
        >>> data = load_sparse_hdf5("sparse.hdf5")
        >>> data["type"]
        'sparse'
        >>> data["train"][0]
        {0: 1.0, 3: 2.0}
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "load_sparse_hdf5 requires the 'h5py' package; "
            "install it with `pip install h5py`."
        ) from exc

    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        # ``type`` attribute. Older public datasets omit it and are
        # implicitly dense.
        vec_type = _read_str_attr(f, "type")
        if vec_type is None:
            vec_type = "dense"
        elif vec_type not in ("dense", "sparse"):
            raise ValueError(
                f"{path!r} has unknown type attribute {vec_type!r}; "
                "expected 'dense' or 'sparse'."
            )
        out["type"] = vec_type
        out["distance"] = _read_str_attr(f, "distance")

        if vec_type != "sparse" and require_sparse:
            raise ValueError(
                f"{path!r} is not a sparse dataset (type={vec_type!r}); "
                "pass require_sparse=False to load dense files."
            )

        for split in splits:
            if split not in f:
                continue
            dset = f[split]
            if vec_type == "sparse":
                # Sparse splits are 1-D INT8 byte streams. ``dset[()]``
                # materializes the dataset as a numpy ndarray (one copy
                # out of HDF5); decode_sparse_bytes then reinterprets
                # that buffer as bytes without an additional copy.
                out[split] = decode_sparse_bytes(dset[()])
            else:
                out[split] = dset[()]

        for name in ("neighbors", "distances"):
            if name in f:
                out[name] = f[name][()]

    return out


def _read_str_attr(h5obj, name: str) -> Optional[str]:
    """Read a string attribute, normalizing the various forms h5py may
    return (``str``, ``bytes``, ``numpy.bytes_``, ``numpy.str_``, 0-d
    arrays, or 1-element arrays) into a plain Python ``str``."""
    if name not in h5obj.attrs:
        return None
    value = h5obj.attrs[name]
    # Unwrap 0-d / single-element numpy arrays first; ``.item()`` yields
    # a Python ``str`` or ``bytes`` (numpy scalars also support it).
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"attribute {name!r} is not a scalar string (shape={value.shape})"
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
