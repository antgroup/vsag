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

#include <vsag/vsag.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t NUM_VECTORS = 1000;
constexpr int64_t DIM = 32;

}  // namespace

int
main() {
    vsag::init();

    /******************* Prepare Dataset *****************/
    std::vector<int64_t> ids(NUM_VECTORS);
    std::vector<float> vectors(NUM_VECTORS * DIM);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distribution;
    for (int64_t i = 0; i < NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = distribution(rng);
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(NUM_VECTORS)
                    ->Dim(DIM)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);

    /******************* Create Disk-Backed Precise Codes *****************/
    const auto unique_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temp_dir = std::filesystem::temp_directory_path() /
                          ("vsag-ivf-precise-bucket-" + std::to_string(unique_id));
    std::filesystem::create_directories(temp_dir);
    const auto precise_file_path = (temp_dir / "precise.codes").generic_string();

    // precise_codes_layout is the only parameter introduced by bucket-aligned precise storage.
    // The existing precise quantizer and IO settings are reused.
    const auto build_params = std::string(R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "buckets_count": 16,
            "buckets_per_data": 1,
            "base_quantization_type": "sq8",
            "partition_strategy_type": "ivf",
            "ivf_train_type": "kmeans",
            "use_reorder": true,
            "precise_quantization_type": "fp32",
            "precise_codes_layout": "bucket",
            "precise_io_type": "buffer_io",
            "precise_file_path": ")") +
                              precise_file_path + R"("
        }
    }
    )";

    auto create_result = vsag::Factory::CreateIndex("ivf", build_params);
    if (not create_result.has_value()) {
        std::cerr << "Create index failed: " << create_result.error().message << std::endl;
        std::filesystem::remove_all(temp_dir);
        return EXIT_FAILURE;
    }
    auto index = std::move(create_result.value());

    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << "Build failed: " << build_result.error().message << std::endl;
        index.reset();
        std::filesystem::remove_all(temp_dir);
        return EXIT_FAILURE;
    }

    /******************* Search With Precise Reordering *****************/
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(DIM)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    const auto search_params = R"(
    {
        "ivf": {
            "scan_buckets_count": 16,
            "factor": 4.0
        }
    }
    )";
    {
        auto search_result = index->KnnSearch(query, 10, search_params);
        if (not search_result.has_value()) {
            std::cerr << "Search failed: " << search_result.error().message << std::endl;
            index.reset();
            std::filesystem::remove_all(temp_dir);
            return EXIT_FAILURE;
        }

        std::cout << "Top-" << search_result.value()->GetDim() << " results:" << std::endl;
        for (int64_t i = 0; i < search_result.value()->GetDim(); ++i) {
            std::cout << "  id=" << search_result.value()->GetIds()[i]
                      << "  dist=" << search_result.value()->GetDistances()[i] << std::endl;
        }
    }

    index.reset();
    std::filesystem::remove_all(temp_dir);
    return EXIT_SUCCESS;
}
