# Adaptive neighbor pruning in HGraph

HGraph offers an experimental, opt-in adaptive selector for NSW bottom-layer construction. It changes the graph built by `Build` and `Add`, including the optimized RaBitQ split/fused build path. It does not run during queries. See [HGraph configuration](hgraph.md#experimental-adaptive-pruning) for the public parameters and defaults.

## Scope and compatibility

- `adaptive_pruning=false` preserves the original selector, including its shortcut for fewer candidates than the degree limit.
- `adaptive_pruning=true` applies to new-node forward selection. `adaptive_pruning_apply_to_reverse=true` additionally applies when an existing bottom-layer reverse neighbor list is full. Lists with free slots append directly.
- Reverse selection uses the existing neighbor as its center, with its old neighbors and the incoming node as candidates. Selection and publication remain under that neighbor's existing lock.
- Upper layers and update/refinement operations retain the existing selector. ODescent, imported-cache builds, non-L2 metrics, and enabled `adaptive_pruning_apply_to_upper=true` are rejected.
- Serialization records the policy. Reload requires the same enabled state and, when enabled, the same baseline alpha, step and scope. Older parameter objects without this policy mean disabled mode. This is not a guarantee that older VSAG binaries can read newly written indexes.

## Distance rule

Let `u` be the center, `c` a candidate, `s` an already accepted neighbor, and `K` the target degree of the current graph. Distances come from the existing build distance provider; enabling this policy does not switch quantizers or retain raw vectors. For the supported L2 path, the scores are squared L2 distances, possibly estimated by the build quantizer.

Reject `c` if any accepted `s` satisfies:

$$
\alpha\,d(s,c) < d(u,c).
$$

Equality does not reject. Increasing alpha relaxes this individual comparison; decreasing it tightens the comparison. This does not imply that every accepted neighbor set is nested as alpha changes, because selection is greedy and earlier choices affect later candidates.

Candidates are sorted by `(distance, internal ID)`. Self edges and duplicate IDs are removed, retaining the closest occurrence of each ID. Distances must be finite and nonnegative. Equal vectors with different IDs are not deduplicated by this selector.

## Adaptive schedule

Write `alpha0` for the existing HGraph `alpha` parameter and `delta` for `adaptive_pruning_adjust_step`. A scan processes candidates in order, accepting non-rejected entries until `K` is reached or the input is exhausted. Rejected entries are recorded separately; unscanned entries are not counted as rejects.

1. Scan the normalized candidates at `alpha0`, obtaining accepted list `A` and rejected list `B`.
2. If `delta=0`, return the first-pass result without filling unused capacity. This is not equivalent to disabling the policy.
3. If `0 < |A| < K`, retain `A` and rescan `B` at a relaxed alpha:

| Ratio `K / |A|` | Relaxed alpha |
| --- | --- |
| `<= 1.5` | `alpha0 + delta` |
| `> 1.5` and `<= 3` | `alpha0 + 2*delta` |
| `> 3` | `alpha0 + 3*delta` |

Remaining rejected candidates are not appended, even when capacity remains.

4. If `|A| = K`, compute the following alpha using the first scan's rejected count:

| Ratio `|B| / K` | Tightened alpha |
| --- | --- |
| `>= 5` | `alpha0` |
| `>= 2.5` and `< 5` | `alpha0 - delta` |
| `< 2.5` | `alpha0 - 2*delta` |

Clear `A` and rescan the **entire original normalized list**, including its previously unscanned tail, at the tightened alpha. If fewer than `K` entries survive, retain them and rescan the new rejects at `alpha0`. This branch never fills unconditionally.

Return the accepted list sorted by `(distance, internal ID)`. Empty inputs, self-only inputs and `K=0` produce no neighbors. The algorithm can return fewer than `K` neighbors; it does not invent candidates.

Configuration requires finite `alpha0` and `delta`, `delta >= 0`, `alpha0 - 2*delta > 0`, and finite `alpha0 + 3*delta`. The GIST reference configuration `alpha0=1.06, delta=0.06` yields relaxed values 1.12/1.18/1.24 and tightened values 1.06/1.00/0.94. These are explicit experiment settings, not dimension-specific defaults.

## Worked example

Use squared L2, center `u=(0,0)`, candidates `a=(1,0)`, `b=(0.5,1)`, `c=(-2,0)`, `K=2`, `alpha0=1.06`, and `delta=0.06`.

The first scan accepts `a,b` because `d(a,b)=d(u,b)=1.25`; it stops before `c`. There are no rejects, so the tightened alpha is 0.94. The second scan accepts `a`, rejects `b` because `0.94*1.25 < 1.25`, then accepts the previously unscanned `c` because `0.94*9 >= 4`. The final neighbors are `a,c`. Retaining the unscanned tail is therefore essential.

## Cost and performance interpretation

For `C` candidates, normalization takes `O(C log C)` time. A constant number of scans perform `O(CK)` pairwise comparisons, whose individual cost depends on the build distance provider. Scratch is local to one selection: candidate/reject lists, an ID set, and per-source distance computers. No global adaptive state or query-time policy is introduced. The helper caches distance computers, not every pairwise distance result.

More passes increase construction work, particularly when full reverse lists are repeatedly pruned. At query time, equal `efSearch` does not fix the number of expanded nodes, scanned edges, or RaBitQ refinements. A changed graph may improve recall while lowering QPS at the same `efSearch`; compare recall–QPS curves, construction cost and memory separately.

## Related work and provenance

The concrete two-sided count-based schedule above was adapted from behavior recovered from a Descartes `ClassicSelector` binary. The VSAG integration scope, configuration, validation and deterministic candidate normalization are implementation choices here. This is not a reproduction of the full Descartes index, nor a claim of inventing adaptive neighbor selection.

- [Descartes](https://github.com/01-ai/Descartes) publicly describes adaptive neighbor selection, but its README does not specify the numerical schedule above.
- [HNSW, Algorithm 4](https://arxiv.org/pdf/1603.09320) describes optional filling from rejected candidates (`keepPrunedConnections`), without this alpha schedule.
- [Vamana construction](https://intel.github.io/ScalableVectorSearch/advanced/graph_search.html) uses alpha-controlled pruning and two graph-wide passes; this differs from per-node feedback.
- [SymphonyQG, section 3.2.2](https://arxiv.org/html/2411.12229v1#S3.SS2.SSS2) adjusts a per-node angular threshold to supplement edges and align out-degree with FastScan batches.
- [Alpha-CNG, section 4.2 and Algorithm 4](https://arxiv.org/html/2510.05975v1#S4.SS2) increases alpha locally until a degree threshold is reached or the configured alpha range is exhausted. It uses a different pruning inequality with an additional tau term. Its published algorithm does not use this accept/reject-ratio tightening schedule.

These establish related public techniques, not a proven derivation chain or priority claim for the exact constants used here.

## Validation

Tests tagged `[adaptive_pruning]` cover ratio boundaries, strict equality, unused capacity without filling, zero step, self/duplicate normalization, invalid inputs, a real L2 tightening example across input permutations, forward/reverse scope, cache rejection, parameter compatibility, and FP32/RaBitQ split/fused build–serialize–reload–Add with one and four construction threads. `[pruning_strategy]`, `[HGraphParameter]`, `[hgraph]`, and `[build_cache]` provide surrounding regression coverage.

Parameters are flat fields under `index_param`; `adaptive_pruning` is a boolean. The former experimental nested object and removed `fill_rejected` option are rejected; rebuild indexes created with that experimental schema.
