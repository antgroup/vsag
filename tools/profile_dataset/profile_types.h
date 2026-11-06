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
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace vsag::dataset_profile {

/// Reports keep insertion order so that sections read top-down; dumps are still deterministic.
using JsonType = nlohmann::ordered_json;

/// Identifier and version of the JSON report layout. Additive changes bump the minor version;
/// renamed or removed keys bump the major version.
constexpr const char* REPORT_SCHEMA = "vsag.dataset_profile";
constexpr const char* REPORT_SCHEMA_VERSION = "1.0";

/// Rows per statistics block. Full-scan partial results are merged in block order, so every
/// metric is independent of the thread count and of the I/O chunk size.
constexpr uint64_t BLOCK_ROWS = 4096;

/// Thrown for invalid input files or options; the CLI reports the message and exits with 1.
class ProfileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class VectorKind {
    DENSE = 0,
    SPARSE,
};

/// Distances follow VSAG conventions: l2 is squared L2, ip is 1 - <x, y> and cosine is
/// 1 - cos(x, y). See docs/docs/en/src/resources/metric_semantics.md.
enum class Metric {
    L2 = 0,
    IP,
    COSINE,
};

std::string
metric_name(Metric metric);

/// Parses "l2", "ip" or "cosine"; throws ProfileError otherwise.
Metric
parse_metric(const std::string& name);

/// Maps the HDF5 `distance` attribute ("euclidean", "ip", "angular") to a metric.
/// Returns false when the attribute is empty or unknown.
bool
metric_from_distance_attribute(const std::string& attribute, Metric& metric);

/// Optional analyses that can be switched off with --disable.
constexpr const char* SECTION_DUPLICATES = "duplicates";
constexpr const char* SECTION_KNN = "knn";
constexpr const char* SECTION_PCA = "pca";
constexpr const char* SECTION_CLUSTERING = "clustering";
constexpr const char* SECTION_HUBNESS = "hubness";
constexpr const char* SECTION_QUANTIZATION = "quantization";
constexpr const char* SECTION_GROUND_TRUTH = "ground_truth";
constexpr const char* SECTION_LABELS = "labels";

const std::vector<std::string>&
optional_sections();

/// Deterministic profiling options. Everything except `threads`, `chunk_mb` and `quiet`
/// is recorded in the report because it can change metric values.
struct ProfileOptions {
    std::string data_path;
    /// Splits to profile; empty means every present split among train and test.
    std::vector<std::string> splits;
    /// "auto" reads the HDF5 `distance` attribute; otherwise l2, ip or cosine.
    std::string metric{"auto"};
    uint64_t seed{42};
    /// Rows drawn uniformly without replacement for every sample-based metric.
    uint64_t sample_size{10000};
    /// Sampled rows whose exact k nearest neighbors are searched in the full split.
    uint64_t knn_queries{1000};
    uint64_t knn_k{20};
    /// Random row pairs (drawn from the sample) for the pairwise-distance distribution.
    uint64_t pair_count{100000};
    uint64_t histogram_bins{50};
    /// k for the sampled k-means summary; 0 disables clustering.
    uint64_t clusters{16};
    uint64_t kmeans_iterations{25};
    /// PCA is skipped above this dimensionality to bound the O(dim^3) eigen-decomposition.
    uint64_t pca_max_dim{2048};
    /// Sampled rows drawn in the 2-D projection of the HTML report.
    uint64_t projection_points{2000};
    uint64_t hubness_sample{2000};
    uint64_t hubness_k{10};
    uint64_t quantization_base{5000};
    uint64_t quantization_queries{200};
    uint64_t quantization_k{10};
    /// Nearest-neighbor distance (relative to the median pairwise distance) below which a
    /// sampled row counts as a near duplicate.
    double near_duplicate_threshold{0.01};
    /// Exact duplicate detection keeps 16 bytes per row; above this budget it falls back to
    /// the sample.
    uint64_t duplicate_memory_mb{4096};
    std::set<std::string> disabled_sections;

    /// Execution-only settings; they never change metric values.
    uint64_t threads{0};
    uint64_t chunk_mb{64};
    bool quiet{false};

    [[nodiscard]] bool
    Enabled(const std::string& section) const {
        return disabled_sections.count(section) == 0;
    }
};

/// Throws ProfileError when an option is out of range.
void
validate_options(const ProfileOptions& options);

/// Options that can change metric values, as recorded in the report `config` section.
JsonType
options_to_json(const ProfileOptions& options);

}  // namespace vsag::dataset_profile
