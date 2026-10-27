// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vsag/index.h"

namespace vsag {

/// Static, no-lock router over one existing FP32/L2 HGraph with a global MCI companion.
/// No partition index is used. The caller must not mutate/reload the index while this router lives.
class GlobalMCIHybrid {
public:
    static tl::expected<std::shared_ptr<GlobalMCIHybrid>, Error>
    Create(IndexPtr index);

    /// Retention selects HGraph first. Otherwise, the configured predictor selects MCI or a scan.
    /// Defaults to distance_samples: 512 global samples and 256 legal samples, seed 42.
    /// Explicit clique_probe uses HGraph top-10 instead of distance sampling.
    /// Legal distance samples become MCI seeds without repeating their distance calculations.
    [[nodiscard]] tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const;

private:
    struct State;
    explicit GlobalMCIHybrid(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

}  // namespace vsag
