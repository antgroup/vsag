// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file 322_feature_build_cache.cpp
 * @brief Demonstrates the Build Cache feature for accelerated HGraph index rebuilding.
 *
 * Usage:
 *   ./322_feature_build_cache <day1_data_dir> <day2_data_dir>
 *
 * Where each data_dir contains:
 *   - id_map:  Text file, one FeatureID per line (e.g., "438b28dd3e02e8325f752dcf61e1d2d6_NEWS")
 *   - label.bin: Binary file of int64_t labels (sequential 0,1,2,...)
 *   - vector_data.bin: Binary file of float32 vectors (dim=768)
 *
 * The program:
 *   1. Loads day1 data, builds an HGraph index (Phase 1)
 *   2. Exports the Build Cache from the day1 index
 *   3. Loads day2 data, uses the cache to warm-start a new index build (Phase 2)
 *   4. Prints cache hit statistics
 *   5. Performs a sample search to verify correctness
 */

#include <vsag/vsag.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Utility: Load data from disk
// ============================================================================

/**
 * Load id_map file: one FeatureID string per line.
 * Returns a vector of strings.
 */
static std::vector<std::string>
load_id_map(const std::string& path) {
    std::vector<std::string> result;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Cannot open id_map file: " << path << std::endl;
        return result;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    return result;
}

/**
 * Load label.bin file: array of int64_t.
 * Returns a vector of int64_t labels.
 */
static std::vector<int64_t>
load_labels(const std::string& path, int64_t expected_count) {
    std::vector<int64_t> labels(expected_count);
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open label.bin file: " << path << std::endl;
        return {};
    }
    fin.read(reinterpret_cast<char*>(labels.data()), expected_count * sizeof(int64_t));
    return labels;
}

/**
 * Load vector_data.bin file: array of float32 vectors.
 * Returns a vector of float values.
 */
static std::vector<float>
load_vectors(const std::string& path, int64_t num_vectors, int64_t dim) {
    std::vector<float> vectors(num_vectors * dim);
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Cannot open vector_data.bin file: " << path << std::endl;
        return {};
    }
    fin.read(reinterpret_cast<char*>(vectors.data()), num_vectors * dim * sizeof(float));
    return vectors;
}

// ============================================================================
// Utility: Create dataset for building
// ============================================================================

static vsag::DatasetPtr
create_dataset(int64_t num_vectors,
               int64_t dim,
               const std::vector<int64_t>& ids,
               const std::vector<float>& vectors,
               const std::vector<std::string>* feature_ids = nullptr) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(num_vectors)->Dim(dim)->Ids(ids.data())->Float32Vectors(vectors.data());
    if (feature_ids != nullptr) {
        dataset->FeatureIds(feature_ids->data());
    }
    dataset->Owner(false);
    return dataset;
}

// ============================================================================
// Main
// ============================================================================

int
main(int argc, char** argv) {
    vsag::init();

    // -------------------------------------------------------------------------
    // Configuration (change these paths for your environment)
    // -------------------------------------------------------------------------
    std::string day1_dir = (argc > 1) ? argv[1] : "./whole-30m-768-euclidean-day1";
    std::string day2_dir = (argc > 2) ? argv[2] : "./whole-30m-768-euclidean-day2";

    // For quick testing, limit the number of vectors to load (-1 means all)
    int64_t max_load = 10000;

    int64_t dim = 768;

    // -------------------------------------------------------------------------
    // Phase 1: Build index with day1 data and export cache
    // -------------------------------------------------------------------------
    std::cout << "=== Phase 1: Building index with day1 data ===" << std::endl;

    auto day1_ids = load_id_map(day1_dir + "/id_map");

    // Determine actual data count from label.bin size.
    // id_map may have more entries than label.bin/vector_data.bin, so we use the
    // minimum of all three sources to ensure alignment.
    int64_t label_count = static_cast<int64_t>(
        std::filesystem::file_size(day1_dir + "/label.bin") / sizeof(int64_t));
    int64_t vector_count = static_cast<int64_t>(
        std::filesystem::file_size(day1_dir + "/vector_data.bin") / (dim * sizeof(float)));
    int64_t id_count = static_cast<int64_t>(day1_ids.size());
    int64_t day1_count = std::min({label_count, vector_count, id_count});

    if (max_load > 0 && max_load < day1_count) {
        day1_count = max_load;
    }

    auto day1_labels = load_labels(day1_dir + "/label.bin", day1_count);
    auto day1_vectors = load_vectors(day1_dir + "/vector_data.bin", day1_count, dim);

    if (day1_labels.empty() || day1_vectors.empty()) {
        std::cerr << "Failed to load day1 data" << std::endl;
        return 1;
    }

    // Limit feature ids to loaded count
    if (static_cast<int64_t>(day1_ids.size()) > day1_count) {
        day1_ids.resize(day1_count);
    }

    std::string hgraph_params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 768,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha": 1.2
        }
    }
    )";

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);

    auto day1_index = engine.CreateIndex("hgraph", hgraph_params).value();

    auto day1_base =
        create_dataset(day1_count, dim, day1_labels, day1_vectors, &day1_ids);

    auto build_start = std::chrono::steady_clock::now();
    auto build_result = day1_index->Build(day1_base);
    auto build_end = std::chrono::steady_clock::now();

    if (!build_result.has_value()) {
        std::cerr << "Failed to build day1 index: " << build_result.error().message << std::endl;
        return 1;
    }
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Day1 index built in " << build_ms << " ms, "
              << "num_elements=" << day1_index->GetNumElements() << std::endl;

    // Check Build Cache support
    std::cout << "SupportsBuildCache: " << (day1_index->SupportsBuildCache() ? "true" : "false")
              << std::endl;

    // Export Build Cache to file
    std::string cache_file = "/tmp/vsag_build_cache.bin";
    {
        std::ofstream ofs(cache_file, std::ios::binary);
        auto export_result = day1_index->ExportBuildCache(ofs);
        if (!export_result.has_value()) {
            std::cerr << "Failed to export build cache: " << export_result.error().message
                      << std::endl;
            return 1;
        }
        ofs.close();
        std::cout << "Build cache exported to " << cache_file << std::endl;
    }

    // -------------------------------------------------------------------------
    // Phase 2: Build index with day2 data using cache warm start
    // -------------------------------------------------------------------------
    std::cout << "\n=== Phase 2: Building index with day2 data (using cache) ===" << std::endl;

    auto day2_ids = load_id_map(day2_dir + "/id_map");
    int64_t label_count2 = static_cast<int64_t>(
        std::filesystem::file_size(day2_dir + "/label.bin") / sizeof(int64_t));
    int64_t vector_count2 = static_cast<int64_t>(
        std::filesystem::file_size(day2_dir + "/vector_data.bin") / (dim * sizeof(float)));
    int64_t id_count2 = static_cast<int64_t>(day2_ids.size());
    int64_t day2_count = std::min({label_count2, vector_count2, id_count2});

    if (max_load > 0 && max_load < day2_count) {
        day2_count = max_load;
    }

    auto day2_labels = load_labels(day2_dir + "/label.bin", day2_count);
    auto day2_vectors = load_vectors(day2_dir + "/vector_data.bin", day2_count, dim);

    if (day2_labels.empty() || day2_vectors.empty()) {
        std::cerr << "Failed to load day2 data" << std::endl;
        return 1;
    }

    if (static_cast<int64_t>(day2_ids.size()) > day2_count) {
        day2_ids.resize(day2_count);
    }

    auto day2_index = engine.CreateIndex("hgraph", hgraph_params).value();

    auto day2_base =
        create_dataset(day2_count, dim, day2_labels, day2_vectors, &day2_ids);

    // Build with cache
    vsag::BuildCacheOptions cache_options;
    cache_options.enable_warm_start = true;
    cache_options.hit_refine_rounds = 2;
    cache_options.missed_refine_rounds = 4;
    cache_options.drop_invalid_neighbors = true;

    {
        std::ifstream ifs(cache_file, std::ios::binary);
        auto cache_build_start = std::chrono::steady_clock::now();
        auto cache_build_result = day2_index->BuildWithCache(day2_base, ifs, cache_options);
        auto cache_build_end = std::chrono::steady_clock::now();

        if (!cache_build_result.has_value()) {
            std::cerr << "BuildWithCache failed: " << cache_build_result.error().message
                      << std::endl;
            std::cerr << "Falling back to full Build..." << std::endl;

            // Fallback to normal Build
            auto fallback_index = engine.CreateIndex("hgraph", hgraph_params).value();
            auto fb_result = fallback_index->Build(day2_base);
            if (!fb_result.has_value()) {
                std::cerr << "Fallback Build also failed: " << fb_result.error().message
                          << std::endl;
                return 1;
            }
            day2_index = fallback_index;
        } else {
            auto cache_build_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(cache_build_end -
                                                                       cache_build_start)
                    .count();
            std::cout << "Day2 index built with cache in " << cache_build_ms << " ms, "
                      << "num_elements=" << day2_index->GetNumElements() << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3: Print Build Cache statistics
    // -------------------------------------------------------------------------
    auto stats_result = day2_index->GetBuildCacheStats();
    if (stats_result.has_value()) {
        const auto& stats = stats_result.value();
        std::cout << "\n=== Build Cache Statistics ===" << std::endl;
        std::cout << "  total_nodes:        " << stats.total_nodes << std::endl;
        std::cout << "  cached_nodes:       " << stats.cached_nodes << std::endl;
        std::cout << "  hit_nodes:          " << stats.hit_nodes << std::endl;
        std::cout << "  missed_nodes:       " << stats.missed_nodes << std::endl;
        std::cout << "  cache_hit_rate:     " << stats.cache_hit_rate << std::endl;
        std::cout << "  dropped_neighbors:  " << stats.dropped_neighbors << std::endl;
        std::cout << "  invalid_neighbors:  " << stats.invalid_neighbors << std::endl;
        std::cout << "  hit_refine_rounds:  " << stats.hit_refine_rounds << std::endl;
        std::cout << "  missed_refine_rounds: " << stats.missed_refine_rounds << std::endl;
        std::cout << "  cache_load_us:      " << stats.cache_load_us << std::endl;
        std::cout << "  warm_start_apply_us: " << stats.warm_start_apply_us << std::endl;
        std::cout << "  hit_refine_us:      " << stats.hit_refine_us << std::endl;
        std::cout << "  missed_refine_us:   " << stats.missed_refine_us << std::endl;
    }

    // -------------------------------------------------------------------------
    // Phase 4: Verify search works
    // -------------------------------------------------------------------------
    std::cout << "\n=== Verification: KnnSearch ===" << std::endl;
    std::vector<float> query_vec(dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < dim; ++i) {
        query_vec[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec.data())->Owner(false);

    std::string search_params = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";

    auto search_result = day2_index->KnnSearch(query, 10, search_params);
    if (search_result.has_value()) {
        std::cout << "Search returned " << search_result.value()->GetDim() << " results:" << std::endl;
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(5), search_result.value()->GetDim());
             ++i) {
            std::cout << "  id=" << search_result.value()->GetIds()[i]
                      << " dist=" << search_result.value()->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search failed: " << search_result.error().message << std::endl;
    }

    // -------------------------------------------------------------------------
    // Phase 5: Export new cache for next cycle
    // -------------------------------------------------------------------------
    std::cout << "\n=== Phase 5: Exporting cache for next build cycle ===" << std::endl;
    {
        std::string cache_file_v2 = "/tmp/vsag_build_cache_v2.bin";
        std::ofstream ofs(cache_file_v2, std::ios::binary);
        auto export_result = day2_index->ExportBuildCache(ofs);
        if (export_result.has_value()) {
            std::cout << "New build cache exported to " << cache_file_v2 << std::endl;
        } else {
            std::cerr << "Failed to export new cache: " << export_result.error().message
                      << std::endl;
        }
    }

    engine.Shutdown();
    return 0;
}
