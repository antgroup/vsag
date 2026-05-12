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

"""Helpers for reading sparse-vector HDF5 datasets produced by the
``eval_performance`` C++ tool.

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

import struct
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "load_sparse_hdf5",
    "decode_sparse_bytes",
]

# Each record header is a single little-endian uint32 storing the number
# of non-zero entries in the vector.
_LEN_STRUCT = struct.Struct("<I")
_LEN_SIZE = _LEN_STRUCT.size  # 4


def decode_sparse_bytes(buffer: Any) -> List[Dict[int, float]]:
    """Decode a raw eval-format sparse byte stream into a list of dicts.

    Args:
        buffer: Any object that supports the buffer protocol (``bytes``,
            ``bytearray``, ``memoryview``, or a 1-D ``numpy.ndarray`` of
            ``int8`` / ``uint8``). The buffer is wrapped in a
            ``memoryview`` and read without copying.

    Returns:
        A list where each element is ``{feature_index: value}`` for one
        sparse vector. Keys are returned in ascending order (the eval
        tool also stores them ordered after parsing).

    Raises:
        ValueError: If the byte stream is truncated or has trailing bytes
            that cannot be parsed as a complete record.
    """
    view = memoryview(buffer)
    total = len(view)
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

        ids = np.frombuffer(view, dtype=np.uint32, count=length, offset=pos)
        pos += ids_size
        vals = np.frombuffer(view, dtype=np.float32, count=length, offset=pos)
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
                # Sparse splits are 1-D INT8 byte streams. Pass the
                # ndarray straight through; decode_sparse_bytes wraps it
                # in a memoryview and reads it without copying.
                out[split] = decode_sparse_bytes(dset[()])
            else:
                out[split] = dset[()]

        for name in ("neighbors", "distances"):
            if name in f:
                out[name] = f[name][()]

    return out


def _read_str_attr(h5obj, name: str) -> Optional[str]:
    """Read a string attribute, handling both Python ``str`` and any
    bytes-like value (``bytes``, ``numpy.bytes_``) that h5py may return
    depending on the on-disk string type."""
    if name not in h5obj.attrs:
        return None
    value = h5obj.attrs[name]
    # ``hasattr('decode')`` catches both ``bytes`` and ``numpy.bytes_``
    # (the latter is a ``bytes`` subclass on numpy 2.x but not on 1.x),
    # while plain ``str`` / ``numpy.str_`` falls through to ``str(value)``.
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)
