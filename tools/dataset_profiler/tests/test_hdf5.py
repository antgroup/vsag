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

"""Tests for the HDF5 readers covering both supported layouts."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_profiler.hdf5 import DatasetFormatError, HDF5Dataset


def test_dense_metadata_and_blocks(tmp_path, dense_matrix):
    import h5py

    path = tmp_path / "dense.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["distance"] = "euclidean"
        f.create_dataset("train", data=dense_matrix)
    with HDF5Dataset(str(path), split="train") as dataset:
        meta = dataset.metadata
        assert meta.vector_type == "dense"
        assert meta.dtype == "FLOAT32"
        assert meta.count == dense_matrix.shape[0]
        assert meta.dim == dense_matrix.shape[1]
        assert meta.distance == "euclidean"
        assert meta.dim_inferred is False
        blocks = list(dataset.iter_dense_blocks(block_rows=25))
        assert sum(block.shape[0] for _, block in blocks) == dense_matrix.shape[0]
        np.testing.assert_allclose(blocks[0][1], dense_matrix[:25])


def test_dense_row_selection_matches_source(tmp_path, rng):
    import h5py

    matrix = rng.integers(-127, 128, size=(50, 12)).astype(np.int8)
    path = tmp_path / "dense.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("train", data=matrix)
    with HDF5Dataset(str(path)) as dataset:
        assert dataset.metadata.dtype == "INT8"
        rows = dataset.read_dense_rows([0, 5, 40])
        np.testing.assert_array_equal(rows, matrix[[0, 5, 40]])


def test_unsigned_dense_is_rejected(tmp_path):
    import h5py

    path = tmp_path / "unsigned_dense.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset(
            "train", data=np.array([[128], [255]], dtype=np.uint8)
        )
    with pytest.raises(DatasetFormatError, match="unsigned dense"):
        HDF5Dataset(str(path))


def test_dense_rejects_unsorted_row_indices(tmp_path, dense_matrix):
    import h5py

    path = tmp_path / "dense.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("train", data=dense_matrix)
    with HDF5Dataset(str(path)) as dataset:
        with pytest.raises(ValueError):
            dataset.read_dense_rows([5, 1])


def test_unsupported_float64_rejected(tmp_path):
    import h5py

    path = tmp_path / "bad_dtype.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("train", data=np.zeros((4, 3), dtype=np.float64))
    with pytest.raises(DatasetFormatError):
        HDF5Dataset(str(path))


def test_missing_split_rejected(tmp_path, dense_matrix):
    import h5py

    path = tmp_path / "dense.hdf5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("train", data=dense_matrix)
    with pytest.raises(DatasetFormatError, match="missing required HDF5 dataset"):
        HDF5Dataset(str(path), split="test")


def test_empty_or_wrong_shape_dense_rejected(tmp_path):
    import h5py

    empty = tmp_path / "empty.hdf5"
    with h5py.File(str(empty), "w") as f:
        f.create_dataset("train", data=np.zeros((0, 4), dtype=np.float32))
    with pytest.raises(DatasetFormatError, match="empty"):
        HDF5Dataset(str(empty))

    flat = tmp_path / "flat.hdf5"
    with h5py.File(str(flat), "w") as f:
        f.create_dataset("train", data=np.zeros(10, dtype=np.float32))
    with pytest.raises(DatasetFormatError, match="must be 2-D"):
        HDF5Dataset(str(flat))


def test_multi_vector_rejected_and_unknown_type_rejected(tmp_path):
    import h5py

    mv = tmp_path / "mv.hdf5"
    with h5py.File(str(mv), "w") as f:
        f.attrs["type"] = "multi_vector"
        f.create_dataset("train", data=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(DatasetFormatError, match="multi_vector"):
        HDF5Dataset(str(mv))

    weird = tmp_path / "weird.hdf5"
    with h5py.File(str(weird), "w") as f:
        f.attrs["type"] = "ternary"
        f.create_dataset("train", data=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(DatasetFormatError, match="unsupported HDF5 dataset type"):
        HDF5Dataset(str(weird))


def test_sparse_with_offsets_round_trip(sparse_path, sparse_records):
    with HDF5Dataset(str(sparse_path)) as dataset:
        meta = dataset.metadata
        assert meta.vector_type == "sparse"
        assert meta.count == len(sparse_records)
        assert meta.has_offsets is True
        records = list(dataset.iter_sparse_records())
        assert len(records) == len(sparse_records)
        for position, ((index, ids, values), (ref_ids, ref_values)) in enumerate(
            zip(records, sparse_records)
        ):
            assert index == position
            np.testing.assert_array_equal(ids, ref_ids)
            np.testing.assert_allclose(values, ref_values)


def test_sparse_without_offsets_matches_offsets_version(
    sparse_path, sparse_no_offsets_path, sparse_records
):
    with HDF5Dataset(str(sparse_path)) as with_offsets:
        a = [(ids, values) for _, ids, values in with_offsets.iter_sparse_records()]
    with HDF5Dataset(str(sparse_no_offsets_path)) as without_offsets:
        b = [(ids, values) for _, ids, values in without_offsets.iter_sparse_records()]
    assert len(a) == len(b) == len(sparse_records)
    for (ida, va), (idb, vb) in zip(a, b):
        np.testing.assert_array_equal(ida, idb)
        np.testing.assert_allclose(va, vb)
    assert without_offsets.metadata.count is None
    assert any("no '/train_offsets'" in w for w in without_offsets.metadata.warnings)


def test_sparse_trailing_bytes_rejected(tmp_path, sparse_records):
    import h5py

    from conftest import encode_sparse_record

    stream = b"".join(encode_sparse_record(ids, values) for ids, values in sparse_records)
    path = tmp_path / "truncated.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["type"] = "sparse"
        f.create_dataset("train", data=np.frombuffer(stream + b"\x01\x02", dtype=np.int8).copy())
    with HDF5Dataset(str(path)) as dataset:
        with pytest.raises(DatasetFormatError, match="trailing"):
            list(dataset.iter_sparse_records())


def test_sparse_offset_validation_errors(tmp_path, sparse_records):
    import h5py

    from conftest import encode_sparse_record

    stream = b"".join(encode_sparse_record(ids, values) for ids, values in sparse_records)
    good_offsets = [0]
    for ids, values in sparse_records:
        good_offsets.append(good_offsets[-1] + 4 + 8 * ids.size)

    bad = tmp_path / "bad_offsets.hdf5"
    with h5py.File(str(bad), "w") as f:
        f.attrs["type"] = "sparse"
        f.create_dataset("train", data=np.frombuffer(stream, dtype=np.int8).copy())
        offsets = np.asarray(good_offsets, dtype=np.uint64)
        offsets[-1] += 1  # sentinel no longer matches the byte size
        f.create_dataset("train_offsets", data=offsets)
    with pytest.raises(DatasetFormatError, match="sentinel"):
        HDF5Dataset(str(bad))

    unsorted = tmp_path / "unsorted_offsets.hdf5"
    with h5py.File(str(unsorted), "w") as f:
        f.attrs["type"] = "sparse"
        f.create_dataset("train", data=np.frombuffer(stream, dtype=np.int8).copy())
        offsets = np.asarray(good_offsets, dtype=np.uint64)
        offsets[1], offsets[2] = offsets[2], offsets[1]
        f.create_dataset("train_offsets", data=offsets)
    with pytest.raises(DatasetFormatError, match="non-decreasing"):
        HDF5Dataset(str(unsorted))


def test_read_sparse_rows_returns_requested_indices(sparse_path):
    with HDF5Dataset(str(sparse_path)) as dataset:
        rows = dataset.read_sparse_rows([2, 10, 13])
    assert set(rows) == {2, 10, 13}


def test_sparse_without_offsets_handles_records_across_tiny_chunks(
    sparse_no_offsets_path, sparse_records
):
    with HDF5Dataset(str(sparse_no_offsets_path)) as dataset:
        records = list(dataset.iter_sparse_records(max_chunk_bytes=16))
    assert len(records) == len(sparse_records)
    for (_, ids, values), (ref_ids, ref_values) in zip(records, sparse_records):
        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_allclose(values, ref_values)


def test_sparse_impossible_length_is_rejected_before_buffering(tmp_path):
    import struct

    import h5py

    path = tmp_path / "impossible_length.hdf5"
    stream = np.frombuffer(struct.pack("<I", 2**32 - 1), dtype=np.int8).copy()
    with h5py.File(str(path), "w") as handle:
        handle.attrs["type"] = "sparse"
        handle.create_dataset("train", data=stream)
    with HDF5Dataset(str(path)) as dataset:
        with pytest.raises(DatasetFormatError, match="truncated"):
            list(dataset.iter_sparse_records(max_chunk_bytes=16))
