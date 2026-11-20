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

"""Shared fixtures for the dataset profiler tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def encode_sparse_record(ids, values) -> bytes:
    """Encode one VSAG sparse record: uint32 len | uint32 ids | float32 vals."""
    ids = np.asarray(ids, dtype="<u4")
    values = np.asarray(values, dtype="<f4")
    assert ids.size == values.size
    return b"".join(
        [
            np.uint32(ids.size).tobytes(),
            ids.tobytes(),
            values.tobytes(),
        ]
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(20260101))


@pytest.fixture
def dense_matrix(rng) -> np.ndarray:
    """Small deterministic dense matrix with duplicates, NaN, and an outlier."""
    base = rng.normal(0.0, 1.0, size=(60, 12)).astype(np.float32)
    # exact duplicates of rows 0..1
    base[2] = base[0]
    base[5] = base[1]
    # NaN and Inf land in row 7
    base[7, 0] = np.float32("nan")
    base[7, 1] = np.float32("inf")
    # one huge outlier row
    base[9] = base[9] * 500.0
    # an all-zero row
    base[11] = 0.0
    return base


@pytest.fixture
def dense_path(tmp_path, dense_matrix) -> Path:
    import h5py

    path = tmp_path / "dense_toy.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["distance"] = "l2"
        f.create_dataset("train", data=dense_matrix)
        f.create_dataset("test", data=dense_matrix[:6])
    return path


@pytest.fixture
def sparse_records():
    """Deterministic list of (ids, values) sparse vectors."""
    rows = []
    gen = np.random.Generator(np.random.PCG64(99))
    for _ in range(40):
        length = int(gen.integers(2, 8))
        ids = np.sort(gen.choice(48, size=length, replace=False)).astype(np.uint32)
        values = gen.uniform(0.1, 1.0, size=length).astype(np.float32)
        rows.append((ids, values))
    # duplicate of the first record and one empty record
    rows.append((rows[0][0].copy(), rows[0][1].copy()))
    rows.append((np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)))
    return rows


def _sparse_bytes_and_offsets(rows):
    stream = bytearray()
    offsets = []
    for ids, values in rows:
        offsets.append(len(stream))
        stream += encode_sparse_record(ids, values)
    offsets.append(len(stream))
    return bytes(stream), offsets


@pytest.fixture
def sparse_path(tmp_path, sparse_records) -> Path:
    import h5py

    stream, offsets = _sparse_bytes_and_offsets(sparse_records)
    path = tmp_path / "sparse_toy.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["type"] = "sparse"
        f.attrs["distance"] = "ip"
        f.create_dataset("train", data=np.frombuffer(stream, dtype=np.int8).copy())
        f.create_dataset("train_offsets", data=np.asarray(offsets, dtype=np.uint64))
    return path


@pytest.fixture
def sparse_no_offsets_path(tmp_path, sparse_records) -> Path:
    import h5py

    stream, _ = _sparse_bytes_and_offsets(sparse_records)
    path = tmp_path / "sparse_no_offsets.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["type"] = "sparse"
        f.create_dataset("train", data=np.frombuffer(stream, dtype=np.int8).copy())
    return path
