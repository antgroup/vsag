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

#include "profile_types.h"

#include <algorithm>
#include <cmath>

namespace vsag::dataset_profile {

std::string
metric_name(Metric metric) {
    switch (metric) {
        case Metric::L2:
            return "l2";
        case Metric::IP:
            return "ip";
        case Metric::COSINE:
            return "cosine";
    }
    return "l2";
}

Metric
parse_metric(const std::string& name) {
    if (name == "l2") {
        return Metric::L2;
    }
    if (name == "ip") {
        return Metric::IP;
    }
    if (name == "cosine") {
        return Metric::COSINE;
    }
    throw ProfileError("unsupported metric '" + name + "'; expected auto, l2, ip or cosine");
}

bool
metric_from_distance_attribute(const std::string& attribute, Metric& metric) {
    if (attribute == "euclidean" || attribute == "l2") {
        metric = Metric::L2;
        return true;
    }
    if (attribute == "ip") {
        metric = Metric::IP;
        return true;
    }
    if (attribute == "angular" || attribute == "cosine") {
        metric = Metric::COSINE;
        return true;
    }
    return false;
}

const std::vector<std::string>&
optional_sections() {
    static const std::vector<std::string> sections = {SECTION_DUPLICATES,
                                                      SECTION_KNN,
                                                      SECTION_PCA,
                                                      SECTION_CLUSTERING,
                                                      SECTION_HUBNESS,
                                                      SECTION_QUANTIZATION,
                                                      SECTION_GROUND_TRUTH,
                                                      SECTION_LABELS};
    return sections;
}

namespace {

void
require_range(const char* name, uint64_t value, uint64_t lower, uint64_t upper) {
    if (value < lower || value > upper) {
        throw ProfileError(std::string("--") + name + " must be in [" + std::to_string(lower) +
                           ", " + std::to_string(upper) + "], got " + std::to_string(value));
    }
}

}  // namespace

void
validate_options(const ProfileOptions& options) {
    if (options.data_path.empty()) {
        throw ProfileError("--datapath is required");
    }
    if (options.metric != "auto") {
        (void)parse_metric(options.metric);
    }
    for (const auto& split : options.splits) {
        if (split != "train" && split != "test") {
            throw ProfileError("unsupported split '" + split + "'; expected train or test");
        }
    }
    for (const auto& section : options.disabled_sections) {
        const auto& known = optional_sections();
        if (std::find(known.begin(), known.end(), section) == known.end()) {
            throw ProfileError("unknown section '" + section + "' in --disable");
        }
    }
    constexpr uint64_t max_rows = 1000000;
    require_range("sample-size", options.sample_size, 2, max_rows);
    require_range("knn-queries", options.knn_queries, 1, 100000);
    require_range("knn-k", options.knn_k, 2, 1000);
    require_range("pair-count", options.pair_count, 1, 100000000);
    require_range("histogram-bins", options.histogram_bins, 2, 10000);
    require_range("clusters", options.clusters, 0, 1024);
    require_range("kmeans-iterations", options.kmeans_iterations, 1, 1000);
    require_range("pca-max-dim", options.pca_max_dim, 2, 16384);
    require_range("projection-points", options.projection_points, 0, 100000);
    require_range("hubness-sample", options.hubness_sample, 2, 100000);
    require_range("hubness-k", options.hubness_k, 1, 1000);
    require_range("quantization-base", options.quantization_base, 2, max_rows);
    require_range("quantization-queries", options.quantization_queries, 1, 100000);
    require_range("quantization-k", options.quantization_k, 1, 1000);
    require_range("chunk-mb", options.chunk_mb, 1, 65536);
    require_range("duplicate-memory-mb", options.duplicate_memory_mb, 0, 1048576);
    const double threshold = options.near_duplicate_threshold;
    if (std::isnan(threshold) || threshold < 0.0 || threshold >= 1.0) {
        throw ProfileError("--near-duplicate-threshold must be in [0, 1)");
    }
}

JsonType
options_to_json(const ProfileOptions& options) {
    JsonType disabled = JsonType::array();
    for (const auto& section : options.disabled_sections) {
        disabled.push_back(section);
    }
    JsonType splits = JsonType::array();
    for (const auto& split : options.splits) {
        splits.push_back(split);
    }
    return JsonType{{"seed", options.seed},
                    {"requested_metric", options.metric},
                    {"requested_splits", splits},
                    {"sample_size", options.sample_size},
                    {"knn_queries", options.knn_queries},
                    {"knn_k", options.knn_k},
                    {"pair_count", options.pair_count},
                    {"histogram_bins", options.histogram_bins},
                    {"clusters", options.clusters},
                    {"kmeans_iterations", options.kmeans_iterations},
                    {"pca_max_dim", options.pca_max_dim},
                    {"projection_points", options.projection_points},
                    {"hubness_sample", options.hubness_sample},
                    {"hubness_k", options.hubness_k},
                    {"quantization_base", options.quantization_base},
                    {"quantization_queries", options.quantization_queries},
                    {"quantization_k", options.quantization_k},
                    {"near_duplicate_threshold", options.near_duplicate_threshold},
                    {"duplicate_memory_mb", options.duplicate_memory_mb},
                    {"disabled_sections", disabled}};
}

}  // namespace vsag::dataset_profile
