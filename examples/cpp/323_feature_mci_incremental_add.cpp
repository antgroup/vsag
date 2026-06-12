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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim = 16;
constexpr int64_t kTotalCount = 256;
constexpr int64_t kBuildCount = 128;
constexpr int64_t kTopK = 5;

const std::string kSearchParams = R"(
{
    "mci": {
        "ef_search": 32,
        "seed_count": 32
    },
    "hgraph": {
        "ef_search": 32
    }
}
)";

void
check(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename E>
void
check_expected(const tl::expected<T, E>& result, const std::string& prefix) {
    if (not result.has_value()) {
        throw std::runtime_error(prefix + result.error().message);
    }
}

std::string
make_mci_params() {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": )"
            << kDim << R"(,
    "index_param": {
        "base_quantization_type": "sq8",
        "base_codes_type": "flatten",
        "max_degree": 16,
        "mcs": 64,
        "clique_max": 12,
        "alpha": 1.2,
        "join_ratio_threshold": 0.6,
        "added_mct": 3,
        "build_thread_count": 2,
        "use_hgraph_hybrid": true,
        "hgraph_valid_ratio_threshold": 0.5,
        "hgraph_ef_search": 32,
        "hgraph_index_param": {
            "base_quantization_type": "fp32",
            "graph_type": "odescent",
            "max_degree": 16,
            "alpha": 1.2,
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.2
        }
    }
})";
    return builder.str();
}

std::vector<float>
generate_vectors() {
    std::vector<float> vectors(kTotalCount * kDim, 0.0F);
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> noise(0.0F, 0.02F);
    for (int64_t row = 0; row < kTotalCount; ++row) {
        for (int64_t col = 0; col < kDim; ++col) {
            const auto cluster = static_cast<float>(row / 32);
            const auto offset = static_cast<float>(row % 32) * 0.03F;
            vectors[row * kDim + col] =
                cluster + offset + static_cast<float>(col) * 0.01F + noise(rng);
        }
    }
    return vectors;
}

vsag::DatasetPtr
make_slice(std::vector<int64_t>& ids, std::vector<float>& vectors, int64_t begin, int64_t count) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(kDim)
        ->Ids(ids.data() + begin)
        ->Float32Vectors(vectors.data() + begin * kDim)
        ->Owner(false);
    return dataset;
}

void
serialize_index(const vsag::IndexPtr& index, const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    check(output.good(), "failed to open index file for writing: " + path);
    auto result = index->Serialize(output);
    check_expected(result, "failed to serialize MCI index: ");
}

void
deserialize_index(const vsag::IndexPtr& index, const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "failed to open index file for reading: " + path);
    auto result = index->Deserialize(input);
    check_expected(result, "failed to deserialize MCI index: ");
}

}  // namespace

int
main() {
    vsag::init();

    std::vector<int64_t> ids(kTotalCount);
    for (int64_t id = 0; id < kTotalCount; ++id) {
        ids[id] = id;
    }
    auto vectors = generate_vectors();
    const auto create_params = make_mci_params();

    auto index = vsag::Factory::CreateIndex("mci", create_params);
    check_expected(index, "failed to create MCI index: ");

    auto build_result = index.value()->Build(make_slice(ids, vectors, 0, kBuildCount));
    check_expected(build_result, "failed to build MCI prefix: ");

    auto add_result =
        index.value()->Add(make_slice(ids, vectors, kBuildCount, kTotalCount - kBuildCount));
    check_expected(add_result, "failed to incrementally add vectors: ");
    check(add_result.value().empty(), "unexpected duplicate ids during MCI Add");
    check(index.value()->GetNumElements() == kTotalCount, "MCI element count mismatch after Add");

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(kDim)->Float32Vectors(vectors.data() + 180 * kDim)->Owner(false);

    auto search_result = index.value()->KnnSearch(query, kTopK, kSearchParams);
    check_expected(search_result, "failed to search incremental MCI index: ");

    const auto work_dir = std::filesystem::path("/tmp/vsag_mci_incremental_example");
    std::filesystem::create_directories(work_dir);
    const auto index_path = (work_dir / "mci_incremental.index").string();
    serialize_index(index.value(), index_path);

    auto reloaded = vsag::Factory::CreateIndex("mci", create_params);
    check_expected(reloaded, "failed to create reload target: ");
    deserialize_index(reloaded.value(), index_path);

    auto reload_result = reloaded.value()->KnnSearch(query, kTopK, kSearchParams);
    check_expected(reload_result, "failed to search reloaded MCI index: ");
    auto stats = reload_result.value()->GetStatistics({"mci_hybrid_route"});
    check(stats.size() == 1, "MCI search did not return route statistics");
    check(stats[0].find("hgraph") != std::string::npos,
          "expected reloaded MCI to restore embedded HGraph, got: " + stats[0]);

    std::cout << "Built prefix vectors: " << kBuildCount << std::endl;
    std::cout << "Incrementally added vectors: " << kTotalCount - kBuildCount << std::endl;
    std::cout << "Serialized MCI index with embedded HGraph: " << index_path << std::endl;
    std::cout << "Reloaded hybrid route: " << stats[0] << std::endl;
    std::cout << "Top-" << kTopK << " ids:" << std::endl;
    for (int64_t rank = 0; rank < reload_result.value()->GetDim(); ++rank) {
        std::cout << reload_result.value()->GetIds()[rank] << " "
                  << reload_result.value()->GetDistances()[rank] << std::endl;
    }

    return 0;
}
