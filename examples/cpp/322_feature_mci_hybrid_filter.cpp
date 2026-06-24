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

#include <algorithm>
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
constexpr int64_t kCount = 128;
constexpr uint64_t kKnngDegree = 8;
constexpr int64_t kTopK = 5;

const std::string kHGraphSearchParams = R"(
{
    "hgraph": {
        "ef_search": 32
    }
}
)";

const std::string kMciSearchParams = R"(
{
    "mci": {
        "ef_search": 16,
        "seed_count": 8
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
make_hgraph_params() {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": )"
            << kDim << R"(,
    "index_param": {
        "base_quantization_type": "fp32",
        "graph_type": "odescent",
        "max_degree": 8,
        "alpha": 1.2,
        "graph_iter_turn": 10,
        "neighbor_sample_rate": 0.2
    }
})";
    return builder.str();
}

std::string
make_mci_params(const std::string& knng_path) {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": )"
            << kDim << R"(,
    "index_param": {
        "max_degree": 8,
        "mcs": )"
            << kKnngDegree << R"(,
        "clique_max": 8,
        "build_thread_count": 2,
        "knng_path": ")"
            << knng_path << R"("
    }
})";
    return builder.str();
}

std::string
make_mci_hybrid_params(const std::string& knng_path, const std::string& hgraph_index_path) {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": )"
            << kDim << R"(,
    "index_param": {
        "max_degree": 8,
        "mcs": )"
            << kKnngDegree << R"(,
        "clique_max": 8,
        "build_thread_count": 2,
        "knng_path": ")"
            << knng_path << R"(",
        "use_hgraph_hybrid": true,
        "hgraph_valid_ratio_threshold": 0.4,
        "hgraph_index_path": ")"
            << hgraph_index_path << R"(",
        "hgraph_ef_search": 32,
        "hgraph_index_param": {
            "base_quantization_type": "fp32",
            "graph_type": "odescent",
            "max_degree": 8,
            "alpha": 1.2,
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.2
        }
    }
})";
    return builder.str();
}

class MinIdFilter : public vsag::Filter {
public:
    explicit MinIdFilter(int64_t min_id) : min_id_(min_id) {
        valid_ids_.reserve(static_cast<uint64_t>(kCount - min_id));
        for (int64_t id = min_id; id < kCount; ++id) {
            valid_ids_.push_back(id);
        }
        valid_ratio_ = static_cast<float>(valid_ids_.size()) / static_cast<float>(kCount);
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= min_id_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return valid_ratio_;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

private:
    int64_t min_id_;
    float valid_ratio_{1.0F};
    std::vector<int64_t> valid_ids_{};
};

void
serialize_index(const vsag::IndexPtr& index, const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    check(output.good(), "failed to open index file for writing: " + path);
    auto serialize_result = index->Serialize(output);
    check_expected(serialize_result, "failed to serialize index: ");
}

void
export_knng_from_hgraph(const vsag::IndexPtr& hgraph,
                        const std::vector<float>& vectors,
                        const std::string& knng_path) {
    std::ofstream output(knng_path, std::ios::binary);
    check(output.good(), "failed to open KNNG file for writing: " + knng_path);

    const auto query_k = static_cast<int64_t>(std::min<uint64_t>(kCount, kKnngDegree + 1));
    for (int64_t row = 0; row < kCount; ++row) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(kDim)
            ->Float32Vectors(const_cast<float*>(vectors.data() + row * kDim))
            ->Owner(false);

        auto search_result = hgraph->KnnSearch(query, query_k, kHGraphSearchParams);
        check_expected(search_result, "failed to export KNNG row " + std::to_string(row) + ": ");

        std::vector<uint32_t> neighbors;
        neighbors.reserve(kKnngDegree);
        auto result = search_result.value();
        for (int64_t rank = 0;
             rank < result->GetDim() && static_cast<uint64_t>(neighbors.size()) < kKnngDegree;
             ++rank) {
            auto neighbor_id = result->GetIds()[rank];
            if (neighbor_id < 0 || neighbor_id >= kCount || neighbor_id == row) {
                continue;
            }
            auto neighbor = static_cast<uint32_t>(neighbor_id);
            if (std::find(neighbors.begin(), neighbors.end(), neighbor) != neighbors.end()) {
                continue;
            }
            neighbors.push_back(neighbor);
        }

        check(not neighbors.empty(),
              "HGraph search produced no usable neighbors for row " + std::to_string(row));
        while (static_cast<uint64_t>(neighbors.size()) < kKnngDegree) {
            neighbors.push_back(neighbors.back());
        }

        output.write(reinterpret_cast<const char*>(neighbors.data()),
                     static_cast<std::streamsize>(kKnngDegree * sizeof(uint32_t)));
        check(output.good(), "failed to write KNNG row " + std::to_string(row));
    }
}

std::vector<float>
generate_vectors() {
    std::vector<float> vectors(kCount * kDim, 0.0F);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> noise(0.0F, 0.02F);
    for (int64_t row = 0; row < kCount; ++row) {
        for (int64_t col = 0; col < kDim; ++col) {
            const auto block = static_cast<float>(row / 16);
            const auto lane = static_cast<float>(row % 16) * 0.05F;
            vectors[row * kDim + col] = block + lane + static_cast<float>(col) * 0.01F + noise(rng);
        }
    }
    return vectors;
}

}  // namespace

int
main() {
    vsag::init();

    const auto work_dir = std::filesystem::path("/tmp/vsag_mci_hybrid_example");
    std::filesystem::create_directories(work_dir);
    const auto hgraph_path = (work_dir / "hgraph.index").string();
    const auto knng_path = (work_dir / "hgraph.knng").string();
    const auto mci_path = (work_dir / "mci.index").string();

    std::vector<int64_t> ids(kCount);
    for (int64_t id = 0; id < kCount; ++id) {
        ids[id] = id;
    }
    auto vectors = generate_vectors();

    auto base = vsag::Dataset::Make();
    base->NumElements(kCount)
        ->Dim(kDim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto hgraph = vsag::Factory::CreateIndex("hgraph", make_hgraph_params());
    check_expected(hgraph, "failed to create HGraph index: ");
    auto hgraph_build = hgraph.value()->Build(base);
    check_expected(hgraph_build, "failed to build HGraph index: ");
    serialize_index(hgraph.value(), hgraph_path);
    export_knng_from_hgraph(hgraph.value(), vectors, knng_path);

    auto mci = vsag::Factory::CreateIndex("mci", make_mci_params(knng_path));
    check_expected(mci, "failed to create MCI index: ");
    auto mci_build = mci.value()->Build(base);
    check_expected(mci_build, "failed to build MCI index: ");
    serialize_index(mci.value(), mci_path);

    auto hybrid = vsag::Factory::CreateIndex("mci", make_mci_hybrid_params(knng_path, hgraph_path));
    check_expected(hybrid, "failed to create Hybrid overlay: ");
    std::ifstream input(mci_path, std::ios::binary);
    check(input.good(), "failed to open MCI index for loading: " + mci_path);
    auto deserialize_result = hybrid.value()->Deserialize(input);
    check_expected(deserialize_result, "failed to load MCI index into Hybrid overlay: ");

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(kDim)->Float32Vectors(vectors.data() + 96 * kDim)->Owner(false);

    auto filter = std::make_shared<MinIdFilter>(64);
    auto search_result = hybrid.value()->KnnSearch(query, kTopK, kMciSearchParams, filter);
    check_expected(search_result, "failed to run Hybrid filtered search: ");

    auto result = search_result.value();
    auto stats = result->GetStatistics({"mci_hybrid_route", "mci_hybrid_valid_ratio"});
    check(stats.size() == 2, "Hybrid search did not return expected statistics");
    check(stats[0].find("hgraph") != std::string::npos,
          "expected Hybrid query to route to HGraph, got: " + stats[0]);

    std::cout << "HGraph index: " << hgraph_path << std::endl;
    std::cout << "Derived KNNG: " << knng_path << std::endl;
    std::cout << "MCI index: " << mci_path << std::endl;
    std::cout << "Hybrid route: " << stats[0] << std::endl;
    std::cout << "Hybrid valid ratio: " << stats[1] << std::endl;
    std::cout << "Filtered results:" << std::endl;
    for (int64_t rank = 0; rank < result->GetDim(); ++rank) {
        check(result->GetIds()[rank] >= 64, "filtered result contains an invalid id");
        std::cout << result->GetIds()[rank] << " " << result->GetDistances()[rank] << std::endl;
    }

    return 0;
}