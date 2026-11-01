// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hdf5_source.h"
#include "linear_algebra.h"
#include "neighbor_search.h"
#include "profile_types.h"

namespace vsag::dataset_profile {

using ProgressFn = std::function<void(const std::string&)>;

struct SplitContext {
    const DatasetInfo* info{nullptr};
    std::string name;
    const ProfileOptions* options{nullptr};
    Metric metric{Metric::L2};
    ProgressFn progress;
};

/// Sampling bookkeeping shared by dense and sparse splits.
struct SamplePlan {
    uint64_t population{0};
    /// Sampled row ids, ascending (Floyd's algorithm on stream "sample/<split>").
    std::vector<uint64_t> rows;
    /// Positions into `rows` in a seeded random order (stream "order/<split>") restricted
    /// to rows that are valid for distance computations. Every sample-based analysis uses
    /// a prefix of this order, so each subset is itself a uniform random sample.
    std::vector<uint64_t> analysis_order;
};

JsonType
sample_json(const SamplePlan& plan, uint64_t used, const std::string& subset);

/// Histogram range for a streamed value distribution: robust_range() applied to at most
/// ~1M finite values taken with a fixed stride from `sample`.
bool
streamed_value_range(
    const std::vector<float>& sample, double minimum, double maximum, double& lower, double& upper);

/// Profiles one dense split with two streaming passes:
///  1. full scan: invalid values, norms, per-dimension moments, sparsity, duplicate hashes,
///     and capture of the sampled rows;
///  2. full scan: value histogram, 6-sigma outliers and the exact k-NN of the sampled
///     queries (and optional queries from another split).
/// Sample-based analyses then run on the in-memory sample only.
class DenseSplitProfiler {
public:
    explicit DenseSplitProfiler(SplitContext context);

    /// `cross_queries` (may be null) are searched against this split in pass 2.
    void
    Run(const DenseQuerySet* cross_queries);

    [[nodiscard]] const JsonType&
    Json() const {
        return json_;
    }

    /// The first `count` valid analysis rows as a query set for another split.
    [[nodiscard]] DenseQuerySet
    MakeQuerySet(uint64_t count) const;

    /// Original row ids of the first `count` valid analysis rows.
    [[nodiscard]] std::vector<uint64_t>
    AnalysisRowIds(uint64_t count) const;

    [[nodiscard]] const std::vector<float>&
    SortedNorms() const {
        return sorted_norms_;
    }

    [[nodiscard]] const std::vector<double>&
    DimensionMeans() const {
        return dim_mean_;
    }

    [[nodiscard]] const std::vector<double>&
    DimensionStds() const {
        return dim_std_;
    }

    /// k-NN results of the cross queries against this split.
    [[nodiscard]] const std::vector<std::vector<Neighbor>>&
    CrossNeighbors() const {
        return cross_neighbors_;
    }

    [[nodiscard]] const std::vector<std::vector<double>>&
    CrossGeometric() const {
        return cross_geometric_;
    }

    /// First-neighbor geometric distances of this split's own sampled queries.
    [[nodiscard]] const std::vector<double>&
    SelfFirstNeighborGeometric() const {
        return self_first_geometric_;
    }

    [[nodiscard]] const PcaResult&
    Pca() const {
        return pca_;
    }

    [[nodiscard]] bool
    HasPca() const {
        return has_pca_;
    }

    /// Normalizes rows first when profiling cosine; used for projections of other splits.
    [[nodiscard]] std::vector<float>
    AnalysisRows(uint64_t count) const;

private:
    void
    prepare_sample();

    void
    first_pass();

    void
    second_pass(const DenseQuerySet* cross_queries);

    void
    finalize();

    [[nodiscard]] const float*
    sample_row(uint64_t analysis_position) const;

    SplitContext context_;
    const ProfileOptions& options_;
    uint64_t rows_{0};
    uint64_t dim_{0};
    uint64_t chunk_rows_{0};
    SamplePlan plan_;
    JsonType json_;

    std::vector<float> sample_data_;
    std::vector<float> row_norms_;
    std::vector<uint32_t> row_nonzeros_;
    std::vector<uint8_t> row_valid_;
    bool full_duplicates_{false};
    std::vector<uint64_t> hash_high_;
    std::vector<uint64_t> hash_low_;

    std::vector<double> dim_mean_;
    std::vector<double> dim_std_;
    std::vector<float> sorted_norms_;
    double global_min_{0.0};
    double global_max_{0.0};
    /// Value-histogram range chosen from the sample (see robust_range()).
    double value_low_{0.0};
    double value_high_{0.0};
    bool value_clipped_{false};

    std::vector<std::vector<Neighbor>> self_neighbors_;
    std::vector<std::vector<double>> self_geometric_;
    std::vector<std::vector<Neighbor>> cross_neighbors_;
    std::vector<std::vector<double>> cross_geometric_;
    std::vector<double> self_first_geometric_;
    PcaResult pca_;
    bool has_pca_{false};
};

/// Profiles one sparse split (records of uint32 ids and float values). Pass structure mirrors
/// the dense profiler; per-dimension statistics are replaced by term statistics.
class SparseSplitProfiler {
public:
    explicit SparseSplitProfiler(SplitContext context);

    void
    Run(const SparseQuerySet* cross_queries);

    [[nodiscard]] const JsonType&
    Json() const {
        return json_;
    }

    [[nodiscard]] SparseQuerySet
    MakeQuerySet(uint64_t count) const;

    [[nodiscard]] std::vector<uint64_t>
    AnalysisRowIds(uint64_t count) const;

    [[nodiscard]] const std::vector<float>&
    SortedNorms() const {
        return sorted_norms_;
    }

    [[nodiscard]] const std::vector<double>&
    SortedNonzeros() const {
        return sorted_nonzeros_;
    }

    /// Document frequency of a term id (0 when absent).
    [[nodiscard]] uint64_t
    DocumentFrequency(uint32_t term) const;

    [[nodiscard]] const std::vector<std::vector<Neighbor>>&
    CrossNeighbors() const {
        return cross_neighbors_;
    }

    [[nodiscard]] const std::vector<double>&
    SelfFirstNeighborDistance() const {
        return self_first_distance_;
    }

private:
    void
    prepare_sample();

    void
    first_pass();

    void
    second_pass(const SparseQuerySet* cross_queries);

    void
    finalize();

    SplitContext context_;
    const ProfileOptions& options_;
    uint64_t rows_{0};
    SamplePlan plan_;
    JsonType json_;

    SparseQuerySet sample_;
    std::vector<float> row_norms_;
    std::vector<uint8_t> row_valid_;
    std::vector<uint64_t> hash_high_;
    std::vector<uint64_t> hash_low_;
    bool full_duplicates_{false};
    std::vector<uint64_t> term_frequency_;
    std::vector<float> sorted_norms_;
    std::vector<double> sorted_nonzeros_;
    double value_min_{0.0};
    double value_max_{0.0};
    double value_low_{0.0};
    double value_high_{0.0};
    bool value_clipped_{false};

    std::vector<std::vector<Neighbor>> self_neighbors_;
    std::vector<std::vector<Neighbor>> cross_neighbors_;
    std::vector<double> self_first_distance_;
    uint64_t self_few_overlaps_{0};
};

/// Groups 128-bit row hashes and reports exact duplicates: duplicate rows, group count, the
/// largest group and up to five example groups (row ids). `rows` maps hash entries to ids.
JsonType
duplicate_report(const std::vector<uint64_t>& high,
                 const std::vector<uint64_t>& low,
                 const std::vector<uint64_t>& rows,
                 uint64_t population);

/// Two independent 64-bit hashes of a word sequence (rows of canonical float bits).
void
hash_words(const uint32_t* words, uint64_t count, uint64_t& high, uint64_t& low);

}  // namespace vsag::dataset_profile
