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

"""Generate reproducible toy dense/sparse HDF5 datasets and profile reports.

This is the demo entry point for the dataset profiler.  It writes:

* ``toy_dense.hdf5``  -- clustered float32 vectors with duplicates, an
  all-zero vector, an outlier, and one NaN value;
* ``toy_sparse.hdf5`` -- sparse records with ``train_offsets``, a duplicate
  record, and one empty record;
* ``toy_dense.profile.{json,html}`` and ``toy_sparse.profile.{json,html}``
  produced from the two files with a fixed seed.

Run from anywhere::

    python tools/dataset_profiler/examples/make_toy_datasets.py --outdir out

Everything is deterministic for a given NumPy build; committed example reports
under ``examples/output/`` were produced with NumPy 2.4 / Python 3.11.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_profiler.html_report import render_html  # noqa: E402
from dataset_profiler.profiler import profile_dataset  # noqa: E402
from dataset_profiler.stats import dump_json  # noqa: E402

SEED = 20260107
_PROFILE_ARGS = dict(seed=42, sample_size=64, pairwise_sample_size=32, clusters=4)


def write_dense_dataset(path: Path) -> None:
    import h5py

    rng = np.random.Generator(np.random.PCG64(SEED))
    centers = np.asarray(
        [
            rng.normal(0.0, 1.0, size=16),
            rng.normal(0.0, 1.0, size=16) + 3.5,
            rng.normal(0.0, 1.0, size=16) - 3.5,
        ],
        dtype=np.float32,
    )
    blocks = []
    for center in centers:
        blocks.append((center[None, :] + 0.4 * rng.normal(0.0, 1.0, size=(40, 16))).astype(np.float32))
    data = np.concatenate(blocks, axis=0)
    data[5] = data[0]  # exact duplicate
    data[9] = data[0]
    data[11] = 0.0  # all-zero vector
    data[13] *= 300.0  # outlier norms
    data[17, 3] = np.float32("nan")  # one invalid value
    with h5py.File(str(path), "w") as f:
        f.attrs["distance"] = "euclidean"
        f.create_dataset("train", data=data)
        f.create_dataset("test", data=data[:8])


def _sparse_records(rng: np.random.Generator):
    records = []
    for _ in range(80):
        length = int(rng.integers(3, 12))
        ids = np.sort(rng.choice(128, size=length, replace=False)).astype(np.uint32)
        values = rng.uniform(0.05, 1.0, size=length).astype(np.float32)
        records.append((ids, values))
    return records


def write_sparse_dataset(path: Path) -> None:
    import h5py

    rng = np.random.Generator(np.random.PCG64(SEED + 1))
    records = _sparse_records(rng)
    records.append((records[0][0].copy(), records[0][1].copy()))  # duplicate record
    records.append((np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)))  # empty record

    stream = bytearray()
    offsets = []
    for ids, values in records:
        offsets.append(len(stream))
        stream += struct.pack("<I", ids.size)
        stream += ids.astype("<u4").tobytes()
        stream += values.astype("<f4").tobytes()
    offsets.append(len(stream))
    with h5py.File(str(path), "w") as f:
        f.attrs["type"] = "sparse"
        f.attrs["distance"] = "ip"
        f.create_dataset("train", data=np.frombuffer(bytes(stream), dtype=np.int8).copy())
        f.create_dataset("train_offsets", data=np.asarray(offsets, dtype=np.uint64))


def profile_to(path: Path, outdir: Path) -> None:
    report = profile_dataset(str(path), **_PROFILE_ARGS)
    stem = path.with_suffix("").name
    json_path = outdir / (stem + ".profile.json")
    html_path = outdir / (stem + ".profile.html")
    json_path.write_text(dump_json(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print("wrote %s (%d warnings)" % (json_path, len(report["warnings"])))
    print("wrote %s" % html_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="directory for the generated .hdf5 files and reports (default: examples/output)",
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dense = outdir / "toy_dense.hdf5"
    sparse = outdir / "toy_sparse.hdf5"
    write_dense_dataset(dense)
    write_sparse_dataset(sparse)
    profile_to(dense, outdir)
    profile_to(sparse, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
