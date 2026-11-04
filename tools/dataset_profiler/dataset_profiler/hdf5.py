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

"""Readers for the HDF5 layouts consumed by VSAG's evaluation tooling.

Two layouts are supported, matching ``docs/docs/en/src/resources/dataset_format.md``:

* **dense** (``type`` missing or ``"dense"``): ``/<split>`` is an ``(N, D)``
  matrix of ``INT8`` or ``FLOAT32`` values.  One-byte unsigned integers are
  reinterpreted as ``INT8`` to match the evaluator's ``ALPHA_I8`` reads.
* **sparse** (``type="sparse"``): ``/<split>`` is a flat ``INT8`` byte stream.
  Each record is ``uint32 len | uint32 ids[len] | float32 vals[len]`` in
  little-endian order.  ``/<split>_offsets`` is an optional ``UINT64`` array of
  record byte offsets with a trailing sentinel.

The reader deliberately streams data in bounded chunks.  Callers never need to
materialize a complete dataset in memory.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_LEN_STRUCT = struct.Struct("<I")
_LEN_SIZE = _LEN_STRUCT.size


class DatasetFormatError(ValueError):
    """Raised when an HDF5 file does not follow a supported VSAG layout."""


@dataclass
class DatasetMetadata:
    path: Path
    file_name: str
    file_size_bytes: int
    vector_type: str
    split: str
    dtype: str
    count: Optional[int]
    dim: Optional[int]
    dim_inferred: bool
    distance: Optional[str]
    has_offsets: bool = False
    datasets: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def read_str_attr(h5obj, name: str) -> Optional[str]:
    """Read a scalar string attribute from an HDF5 object."""
    if name not in h5obj.attrs:
        return None
    value = h5obj.attrs[name]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise DatasetFormatError(
                "attribute %r is not a scalar string (shape=%s)" % (name, value.shape)
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class HDF5Dataset:
    """Streaming reader for one split of a VSAG-compatible HDF5 dataset."""

    def __init__(self, path: str, split: str = "train") -> None:
        import h5py

        self.path = Path(path)
        if not self.path.is_file():
            raise DatasetFormatError("input file does not exist: %s" % self.path)
        self.split = split
        self._file = h5py.File(str(self.path), "r")
        self._dataset = None
        self._offsets = None
        self._metadata: Optional[DatasetMetadata] = None
        try:
            self._metadata = self._read_metadata()
        except Exception:
            self._file.close()
            raise

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> "HDF5Dataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def metadata(self) -> DatasetMetadata:
        assert self._metadata is not None
        return self._metadata

    # -- metadata ------------------------------------------------------
    def _read_metadata(self) -> DatasetMetadata:
        f = self._file
        vector_type = read_str_attr(f, "type") or "dense"
        vector_type = vector_type.strip().lower()
        if vector_type == "multi_vector":
            raise DatasetFormatError(
                "multi_vector datasets are not supported by the dataset profiler; "
                "profile the underlying flat sub-vector matrix or a dense/sparse export"
            )
        if vector_type not in ("dense", "sparse"):
            raise DatasetFormatError(
                "unsupported HDF5 dataset type %r (expected 'dense' or 'sparse')"
                % vector_type
            )
        if self.split not in f:
            raise DatasetFormatError(
                "missing required HDF5 dataset '/%s'" % self.split
            )

        dset = f[self.split]
        dtype = _describe_dtype(dset.dtype)
        attributes = {}
        for key in ("type", "distance", "multi_vector_dim"):
            value = read_str_attr(f, key)
            if value is not None:
                attributes[key] = value
        distance = attributes.get("distance")
        warnings: List[str] = []

        count: Optional[int] = None
        dim: Optional[int] = None
        dim_inferred = False
        has_offsets = False

        if vector_type == "dense":
            if len(dset.shape) != 2:
                raise DatasetFormatError(
                    "dense dataset '/%s' must be 2-D, got shape %s"
                    % (self.split, dset.shape)
                )
            count, dim = int(dset.shape[0]), int(dset.shape[1])
            if count <= 0:
                raise DatasetFormatError("dense dataset '/%s' is empty" % self.split)
            if dim <= 0:
                raise DatasetFormatError(
                    "dense dataset '/%s' has non-positive dimensionality" % self.split
                )
        else:
            if len(dset.shape) != 1:
                raise DatasetFormatError(
                    "sparse dataset '/%s' must be a flat 1-D byte stream, got shape %s"
                    % (self.split, dset.shape)
                )
            offsets_name = self.split + "_offsets"
            if offsets_name in f:
                offsets = np.asarray(f[offsets_name][()], dtype=np.uint64)
                if len(offsets) < 1:
                    raise DatasetFormatError(
                        "'/%s' must contain at least the trailing sentinel" % offsets_name
                    )
                if int(offsets[0]) != 0:
                    raise DatasetFormatError(
                        "'/%s' must start at byte offset 0" % offsets_name
                    )
                if np.any(offsets[1:] < offsets[:-1]):
                    raise DatasetFormatError(
                        "'/%s' must be non-decreasing" % offsets_name
                    )
                if int(offsets[-1]) != int(dset.shape[0]):
                    raise DatasetFormatError(
                        "'/%s' sentinel (%d) does not match '/%s' size (%d)"
                        % (offsets_name, int(offsets[-1]), self.split, int(dset.shape[0]))
                    )
                count = len(offsets) - 1
                has_offsets = True
                self._offsets = offsets
            else:
                warnings.append(
                    "sparse dataset has no '/%s_offsets'; records are counted by scanning"
                    % self.split
                )
            if count == 0:
                raise DatasetFormatError("sparse dataset '/%s' is empty" % self.split)
            dim_inferred = True
            warnings.append(
                "sparse dimensionality is inferred as max(feature id) + 1 during profiling"
            )

        return DatasetMetadata(
            path=self.path,
            file_name=self.path.name,
            file_size_bytes=int(self.path.stat().st_size),
            vector_type=vector_type,
            split=self.split,
            dtype=dtype,
            count=count,
            dim=dim,
            dim_inferred=dim_inferred,
            distance=distance,
            has_offsets=has_offsets,
            datasets=sorted(str(name) for name in f.keys()),
            attributes=attributes,
            warnings=warnings,
        )

    # -- dense ---------------------------------------------------------
    def iter_dense_blocks(
        self, block_rows: int = 65536
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(start_row, block)`` slices of the dense matrix."""
        if self.metadata.vector_type != "dense":
            raise DatasetFormatError("iter_dense_blocks requires a dense dataset")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        dset = self._file[self.split]
        total = int(dset.shape[0])
        for start in range(0, total, block_rows):
            end = min(start + block_rows, total)
            yield start, _evaluator_dense_view(dset[start:end])

    def read_dense_rows(self, indices: Sequence[int]) -> np.ndarray:
        """Read selected rows from a dense matrix (indices must be sorted)."""
        if self.metadata.vector_type != "dense":
            raise DatasetFormatError("read_dense_rows requires a dense dataset")
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0, int(self.metadata.dim or 0)), dtype=np.float64)
        if np.any(idx[1:] <= idx[:-1]):
            raise ValueError("dense row indices must be strictly increasing")
        dset = self._file[self.split]
        return _evaluator_dense_view(dset[idx.tolist()])

    # -- sparse --------------------------------------------------------
    def iter_sparse_records(
        self, max_chunk_bytes: int = DEFAULT_CHUNK_BYTES
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(index, ids, values)`` for every sparse vector in the split."""
        if self.metadata.vector_type != "sparse":
            raise DatasetFormatError("iter_sparse_records requires a sparse dataset")
        if max_chunk_bytes < 16:
            raise ValueError("max_chunk_bytes is too small")
        if self._offsets is not None:
            yield from self._iter_sparse_with_offsets(max_chunk_bytes)
        else:
            yield from self._iter_sparse_without_offsets(max_chunk_bytes)

    def _iter_sparse_with_offsets(
        self, max_chunk_bytes: int
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        dset = self._file[self.split]
        offsets = self._offsets
        assert offsets is not None
        total_records = len(offsets) - 1
        start = 0
        while start < total_records:
            limit = int(offsets[start]) + int(max_chunk_bytes)
            end = int(np.searchsorted(offsets, limit, side="right")) - 1
            if end <= start:
                end = start + 1
            end = min(end, total_records)
            byte_start = int(offsets[start])
            byte_end = int(offsets[end])
            chunk = np.asarray(dset[byte_start:byte_end])
            view = memoryview(chunk).cast("B")
            pos = 0
            for index in range(start, end):
                record_size = int(offsets[index + 1]) - int(offsets[index])
                ids, values, consumed = _decode_one_record(view, pos, record_size)
                yield index, ids, values
                pos += consumed
            if pos != view.nbytes:
                raise DatasetFormatError(
                    "sparse byte stream is inconsistent with '/%s_offsets'" % self.split
                )
            start = end

    def _iter_sparse_without_offsets(
        self, max_chunk_bytes: int
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        dset = self._file[self.split]
        total_bytes = int(dset.shape[0])
        buffer = bytearray()
        index = 0
        position = 0
        while position < total_bytes:
            end = min(position + max_chunk_bytes, total_bytes)
            buffer.extend(np.asarray(dset[position:end]).tobytes())
            position = end
            offset = 0
            while True:
                if offset + _LEN_SIZE > len(buffer):
                    break
                (length,) = _LEN_STRUCT.unpack_from(buffer, offset)
                record_size = _LEN_SIZE + 8 * int(length)
                record_start = position - len(buffer) + offset
                remaining = total_bytes - record_start
                if record_size > remaining:
                    raise DatasetFormatError(
                        "sparse record is truncated: len=%d requires %d byte(s), "
                        "but only %d byte(s) remain"
                        % (int(length), record_size, remaining)
                    )
                if offset + record_size > len(buffer):
                    break
                ids, values, consumed = _decode_one_record(buffer, offset, record_size)
                yield index, ids, values
                index += 1
                offset += consumed
            del buffer[:offset]
        if buffer:
            raise DatasetFormatError(
                "sparse byte stream has %d trailing byte(s) that do not form a "
                "complete record" % len(buffer)
            )

    def read_sparse_rows(
        self, indices: Sequence[int], max_chunk_bytes: int = DEFAULT_CHUNK_BYTES
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Read selected sparse vectors into a mapping ``index -> (ids, values)``."""
        wanted = set(int(i) for i in indices)
        if not wanted:
            return {}
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for index, ids, values in self.iter_sparse_records(max_chunk_bytes):
            if index in wanted:
                out[index] = (ids.copy(), values.copy())
                if len(out) == len(wanted):
                    break
        missing = wanted.difference(out)
        if missing:
            raise DatasetFormatError(
                "sparse dataset ended before sampled indices: %s"
                % sorted(missing)[:5]
            )
        return out


def _evaluator_dense_view(values) -> np.ndarray:
    """Match the evaluator's signed interpretation of one-byte dense values."""
    array = np.asarray(values)
    if array.dtype.kind == "u" and array.dtype.itemsize == 1:
        return array.view(np.int8)
    return array


def _describe_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype.kind in ("i", "u") and dtype.itemsize == 1:
        return "INT8"
    if dtype.kind == "f" and dtype.itemsize == 4:
        return "FLOAT32"
    raise DatasetFormatError(
        "unsupported HDF5 element type %r; expected INT8 or FLOAT32" % dtype
    )


def _decode_one_record(
    buffer, offset: int, expected_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decode one sparse record starting at ``offset``.

    Returns ``(ids, values, consumed_bytes)``.  ``expected_size`` is supplied
    when byte offsets are available and must match the encoded record exactly.
    """
    view = memoryview(buffer).cast("B")
    if offset + _LEN_SIZE > view.nbytes:
        raise DatasetFormatError("sparse record is truncated before its length header")
    (length,) = _LEN_STRUCT.unpack_from(view, offset)
    record_size = _LEN_SIZE + 8 * int(length)
    if expected_size is not None and record_size != expected_size:
        raise DatasetFormatError(
            "sparse record size mismatch: offsets say %d byte(s), record header "
            "requires %d" % (expected_size, record_size)
        )
    if offset + record_size > view.nbytes:
        raise DatasetFormatError(
            "sparse record is truncated: len=%d requires %d byte(s)"
            % (int(length), record_size)
        )
    ids_start = offset + _LEN_SIZE
    vals_start = ids_start + 4 * int(length)
    if length:
        ids = np.frombuffer(view, dtype="<u4", count=int(length), offset=ids_start).copy()
        values = np.frombuffer(
            view, dtype="<f4", count=int(length), offset=vals_start
        ).copy()
    else:
        ids = np.empty(0, dtype=np.uint32)
        values = np.empty(0, dtype=np.float32)
    return ids, values, record_size
