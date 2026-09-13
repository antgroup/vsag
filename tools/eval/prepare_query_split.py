#!/usr/bin/env python3
# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
"""Prepare dense HDF5 calibration and validation query subsets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


ROLES = ("calibration", "validation")
DATASETS = {"train", "test", "neighbors", "distances"}
BLOCK_BYTES = 4 * 1024 * 1024


def text_attribute(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_dataset(source):
    if text_attribute(source.attrs.get("type", "dense")) != "dense":
        raise ValueError("only dense datasets are supported")
    if set(source.keys()) != DATASETS:
        raise ValueError("expected only train, test, neighbors and distances (no filters)")
    for name in DATASETS:
        data = source[name]
        if not isinstance(data, h5py.Dataset) or data.ndim != 2 or min(data.shape) == 0:
            raise ValueError(f"{name} must be a non-empty matrix")
    train, queries = source["train"], source["test"]
    vector_type = train.dtype.kind, train.dtype.itemsize
    if vector_type not in {("f", 4), ("i", 1)} or (queries.dtype.kind, queries.dtype.itemsize) != vector_type:
        raise ValueError("train/test must have matching float32 or int8 types")
    if train.shape[1] != queries.shape[1]:
        raise ValueError("train/test dimensions differ")
    neighbors, distances = source["neighbors"], source["distances"]
    if neighbors.shape != distances.shape or neighbors.shape[0] != queries.shape[0]:
        raise ValueError("ground truth must align with query rows")
    if (neighbors.dtype.kind, neighbors.dtype.itemsize) != ("i", 8):
        raise ValueError("neighbors must contain int64 row indices")
    if (distances.dtype.kind, distances.dtype.itemsize) != ("f", 4):
        raise ValueError("distances must contain float32 values")
    if text_attribute(source.attrs.get("distance", "")) not in {"euclidean", "angular", "ip"}:
        raise ValueError("distance must be euclidean, angular or ip")
    for item in [source, *(source[name] for name in DATASETS)]:
        for name in item.attrs:
            dtype = item.attrs.get_id(name).dtype
            if dtype.hasobject and h5py.check_string_dtype(dtype) is None:
                raise ValueError("reference and non-string variable-length attributes are unsupported")


def validate_rows(rows, query_count):
    if not isinstance(rows, dict) or set(rows) != set(ROLES):
        raise ValueError("query rows must contain calibration and validation lists")
    seen = set()
    for role in ROLES:
        selected = rows[role]
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"{role} must be a non-empty list")
        if any(type(i) is not int or not 0 <= i < query_count for i in selected):
            raise ValueError(f"{role} contains an invalid query row")
        if len(set(selected)) != len(selected) or seen.intersection(selected):
            raise ValueError("query row indices must be unique within and across splits")
        seen.update(selected)


def read_rows(dataset, indices):
    """Read bounded blocks, retaining the caller's order despite h5py's sorted indexing."""
    batch = max(1, BLOCK_BYTES // (dataset.shape[1] * dataset.dtype.itemsize))
    for start in range(0, len(indices), batch):
        selected = np.asarray(indices[start:start + batch], dtype=np.int64)
        order = np.argsort(selected)
        yield start, dataset[selected[order]][np.argsort(order)]


def validate_content(source, rows):
    seen = {}
    for role in ROLES:
        for offset, values in read_rows(source["test"], rows[role]):
            if not np.isfinite(values).all():
                raise ValueError(f"{role} contains non-finite query values")
            canonical = values.astype(values.dtype.newbyteorder("<"), copy=True)
            canonical[canonical == 0] = 0  # Treat +0 and -0 as the same query value.
            for i, query in enumerate(canonical):
                row = rows[role][offset + i]
                key = hashlib.sha256(query.tobytes()).digest()
                previous = seen.setdefault(key, (role, row))
                if previous[0] != role:
                    raise ValueError(f"identical queries cross splits: rows {previous[1]} and {row}")
        for _, neighbors in read_rows(source["neighbors"], rows[role]):
            if np.any(neighbors < 0) or np.any(neighbors >= source["train"].shape[0]):
                raise ValueError(f"{role} has ground-truth indices outside train")
        for _, distances in read_rows(source["distances"], rows[role]):
            if not np.isfinite(distances).all():
                raise ValueError(f"{role} has non-finite ground-truth distances")


def copy_attributes(source, destination):
    for name, value in source.attrs.items():
        destination.attrs.create(name, value, dtype=source.attrs.get_id(name).dtype)


def prepare_split(source_path, rows, output_path):
    source_path = Path(source_path).resolve(strict=True)
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_path}")
    output_path = output_path.resolve()
    with h5py.File(source_path, "r") as source:
        validate_dataset(source)
        validate_rows(rows, source["test"].shape[0])
        validate_content(source, rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        relative_source = os.path.relpath(source_path, output_path)
        with tempfile.TemporaryDirectory(prefix=".query-split-", dir=output_path.parent) as temp:
            staging = Path(temp)
            for role in ROLES:
                with h5py.File(staging / f"{role}.hdf5", "x") as target:
                    copy_attributes(source, target)
                    target["train"] = h5py.ExternalLink(relative_source, "/train")
                    for name in ("test", "neighbors", "distances"):
                        original = source[name]
                        selected = target.create_dataset(
                            name, shape=(len(rows[role]), original.shape[1]), dtype=original.dtype
                        )
                        copy_attributes(original, selected)
                        for offset, values in read_rows(original, rows[role]):
                            selected[offset:offset + len(values)] = values
            (staging / "query_rows.json").write_text(
                json.dumps({"source": relative_source, "query_rows": rows}, indent=2) + "\n",
                encoding="utf-8",
            )
            # Reserve the destination exclusively; never replace an existing directory.
            output_path.mkdir()
            for path in staging.iterdir():
                path.rename(output_path / path.name)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="unfiltered dense ANN-benchmarks HDF5 file")
    parser.add_argument("query_rows", type=Path, help="JSON calibration/validation row lists")
    parser.add_argument("output", type=Path, help="new output directory")
    args = parser.parse_args()
    try:
        rows = json.loads(args.query_rows.read_text(encoding="utf-8"))
        output = prepare_split(args.dataset, rows, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared {output / 'calibration.hdf5'} and {output / 'validation.hdf5'}")


if __name__ == "__main__":
    main()
