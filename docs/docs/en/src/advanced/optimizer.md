# Optimizer (Tune)

For graph-based indexes (HNSW, HGraph), VSAG exposes the `Tune` interface, which automatically
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
| hnsw | yes |
| hgraph | yes |
| diskann | partial |
| ivf / sindi / brute_force | no |

## Example

`examples/cpp/318_feature_tune.cpp` walks through an end-to-end tuning flow:

1. Create the index and `Build`.
2. Call `Tune` with a representative query set.
3. Serialize the tuned index for production use.

## Notes

- Tuning is sensitive to the query distribution — use samples that reflect real traffic.
- Tuned parameters are persisted together with the index metadata via `Serialize`/`Deserialize`
  and remain in effect after deployment.
