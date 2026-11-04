# Optimizer (Tune)

For graph-based indexes (HGraph), VSAG exposes the `Tune` interface, which automatically
adjusts runtime parameters based on a representative query set to get a better trade-off between
**recall** and **latency**. Internally this is the historical "ELP Optimizer".

## Basic Usage

```cpp
#include <vsag/vsag.h>

auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
index->Build(base_dataset);

std::string tune_params = R"(
{
    "queries_dataset": "path/or/inline/queries",
    "target_recall": 0.95,
    "top_k": 10
}
)";
auto ret = index->Tune(tune_params);
```

The second argument `disable_future_tuning` defaults to `false`, allowing repeated calls to keep
refining. Set it to `true` to freeze the parameters.

## Relationship with the ELP Optimizer

Older literature (see [Research Papers](../resources/research_papers.md)) refers to the "ELP
Optimizer". Its implementation key is `use_elp_optimizer`, which now lives behind the unified
`Tune` API — users no longer need to flip it directly.

## Supported Indexes

| Index type | Supports Tune |
|------------|---------------|
| hgraph | yes |
| ivf / sindi / brute_force | no |

## Example

`examples/cpp/318_feature_tune.cpp` walks through an end-to-end tuning flow:

1. Create the index and `Build`.
2. Call `Tune` with a representative query set.
3. Serialize the tuned index for production use.

## Notes

- When rebuilding HGraph vector codes, `Tune` needs a raw or FP32 source covering the full internal ID range. If a failed `Add` leaves that source incomplete and no complete fallback exists, `Tune` returns `false` without changing the index. Later successful Adds do not fill earlier unwritten IDs. Deduplicated storage must have a published mapping for every ID; ordinary storage retains unfinished-write information and reserved IDs across serialization, so an Add after restore cannot reuse a missing vector’s ID. This uses existing count metadata without changing the binary format; older snapshots without unfinished-write metadata cannot recover that information. `Merge` rejects a destination or any input with unfinished code writes with `INVALID_ARGUMENT`, before copying any input; rebuild incomplete indexes from original data before merging. Restores reject reservation counts above the serialized capacity or internal ID limit with `INVALID_BINARY`; legacy metadata without a reservation count still falls back to the written-code count. It does not recover vectors from the failed `Add`; rebuild the index from the original data before tuning again.
- Tuning is sensitive to the query distribution — use samples that reflect real traffic.
- Tuned parameters are persisted together with the index metadata via `Serialize`/`Deserialize`
  and remain in effect after deployment.
