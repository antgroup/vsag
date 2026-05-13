# Metric Semantics in VSAG

This page explains how VSAG treats `l2`, `ip`, and `cosine` in practice.

> Warning: VSAG's internal metric implementations are optimized for performance
> and consistency. Their behavior may differ from the textbook mathematical
> definitions, so use the semantics described here when comparing results or
> preparing ground truth.

VSAG keeps all search APIs in a "smaller is better" distance model. For that
reason, several internal implementations reuse squared distances, normalized
vectors, or cached norms to keep behavior fast and consistent across index
types.

## `l2`

- The distance is `L2Sqr` (squared L2 distance).
- Internally, many kernels work with `L2Sqr` for speed.
- The squared form is used for performance; ranking remains consistent with
  L2 distance. Returned distance values and range-search thresholds are
  squared.

## `ip`

- The distance is `1 - inner_product`.
- Larger inner product means smaller distance.

## `cosine`

- The distance is `1 - cosine_similarity`.
- For performance, implementations may normalize vectors or store extra norm
  information so cosine can reuse IP-oriented kernels.

## Return Value Range

- `l2`: `0` to `+infinity`
- `ip`: unbounded; values may be negative when `inner_product > 1`
- `cosine`: `0` to `2` for cosine similarity in `[-1, 1]`

## Why this matters

- Dataset ground truth, query semantics, and index internals need to agree on
  the same metric family.
- `l2`, `ip`, and `cosine` are not interchangeable after an index is built.
- When comparing results across tools, check whether the tool uses a distance
  or a similarity convention.

## Related Pages

- [Creating an Index](../guide/create_index.md)
- [Index Parameters](index_parameters.md)
- [HDF5 Dataset Format](dataset_format.md)
