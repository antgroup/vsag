#!/usr/bin/env python3
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

import argparse
import json
from typing import Any

import h5py


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def dataset_info(dataset: h5py.Dataset) -> dict[str, Any]:
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "ndim": dataset.ndim,
        "logical_bytes": int(dataset.size * dataset.dtype.itemsize),
        "storage_bytes": int(dataset.id.get_storage_size()),
        "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VSAG eval HDF5 dataset metadata.")
    parser.add_argument("path", help="Path to the HDF5 dataset.")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as file:
        datasets = {
            name: dataset_info(obj)
            for name, obj in file.items()
            if isinstance(obj, h5py.Dataset)
        }
        attrs = {key: to_jsonable(value) for key, value in file.attrs.items()}

        train = datasets.get("train")
        test = datasets.get("test")
        neighbors = datasets.get("neighbors")
        distances = datasets.get("distances")

        inferred = {
            "path": args.path,
            "vector_type": attrs.get("type", "dense"),
            "base_count": train["shape"][0] if train is not None and train["shape"] else None,
            "query_count": test["shape"][0] if test is not None and test["shape"] else None,
            "dim": train["shape"][1] if train is not None and len(train["shape"]) > 1 else None,
            "train_dtype": train["dtype"] if train is not None else None,
            "test_dtype": test["dtype"] if test is not None else None,
            "groundtruth_k": (
                neighbors["shape"][1]
                if neighbors is not None and len(neighbors["shape"]) > 1
                else None
            ),
            "has_distances": distances is not None,
            "has_labels": "train_labels" in datasets and "test_labels" in datasets,
            "has_valid_ratios": "valid_ratios" in datasets,
        }

        print(json.dumps({"attrs": attrs, "inferred": inferred, "datasets": datasets}, indent=2))


if __name__ == "__main__":
    main()
