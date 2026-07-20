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

#include <vsag/filter.h>
#include <vsag/vsag.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kDim = 32;
constexpr int64_t kBaseCount = 512;
constexpr int64_t kTopK = 10;
constexpr const char* kSerializedIndex = "/tmp/vsag_hgraph_mci_companion.index";

class ModuloFilter : public vsag::Filter {
public:
    explicit ModuloFilter(int64_t divisor) : divisor_(divisor) {
        for (int64_t id = 0; id < kBaseCount; ++id) {
            if (this->CheckValid(id)) {
                valid_ids_.push_back(id);
            }
        }
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id >= 0 and id % divisor_ == 0;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return static_cast<float>(valid_ids_.size()) / static_cast<float>(kBaseCount);
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

private:
    int64_t divisor_{1};
    std::vector<int64_t> valid_ids_{};
};

template <typename T, typename E>
void
check_expected(const tl::expected<T, E>& result, const std::string& prefix) {
    if (not result.has_value()) {
        std::cerr << prefix << result.error().message << std::endl;
        std::exit(1);
    }
}

vsag::DatasetPtr
make_base_dataset(std::vector<int64_t>& ids, std::vector<float>& vectors) {
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    ids.resize(kBaseCount);
    vectors.resize(kBaseCount * kDim);
    for (int64_t i = 0; i < kBaseCount; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = distrib_real(rng);
    }

    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(kBaseCount)
        ->Dim(kDim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

vsag::DatasetPtr
make_query_dataset(std::vector<float>& query_vector) {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> distrib_real;
    query_vector.resize(kDim);
    for (auto& value : query_vector) {
        value = distrib_real(rng);
    }

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(kDim)->Float32Vectors(query_vector.data())->Owner(false);
    return query;
}

std::string
make_hgraph_mci_params() {
    return R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "base_io_type": "memory_io",
            "store_raw_vector": true,
            "raw_vector_io_type": "memory_io",
            "graph_type": "odescent",
            "max_degree": 16,
            "ef_construction": 80,
            "build_thread_count": 4,
            "alpha": 1.2,
            "graph_iter_turn": 4,
            "neighbor_sample_rate": 0.2,
            "mci_mcs": 32,
            "mci_clique_max": 16,
            "mci_alpha": 1.2
        }
    }
    )";
}

void
build_and_save(vsag::Engine& engine, const vsag::DatasetPtr& base) {
    auto index_result = engine.CreateIndex("hgraph", make_hgraph_mci_params());
    check_expected(index_result, "create HGraph MCI index failed: ");
    auto index = index_result.value();

    auto build_result = index->Build(base);
    check_expected(build_result, "build HGraph MCI index failed: ");
    std::cout << "[build] vectors=" << index->GetNumElements() << std::endl;
    std::cout << "[build] stats:\n" << index->GetStats() << std::endl;

    std::ofstream out(kSerializedIndex, std::ios::binary);
    auto serialize_result = index->Serialize(out);
    check_expected(serialize_result, "serialize HGraph MCI index failed: ");
    std::cout << "[build] saved to " << kSerializedIndex << std::endl;
}

vsag::IndexPtr
load_index(vsag::Engine& engine) {
    auto index_result = engine.CreateIndex("hgraph", make_hgraph_mci_params());
    check_expected(index_result, "create HGraph MCI index for load failed: ");
    auto index = index_result.value();

    std::ifstream in(kSerializedIndex, std::ios::binary);
    auto deserialize_result = index->Deserialize(in);
    check_expected(deserialize_result, "deserialize HGraph MCI index failed: ");
    std::cout << "[load] vectors=" << index->GetNumElements() << std::endl;
    return index;
}

void
run_filtered_search(const vsag::IndexPtr& index, const vsag::DatasetPtr& query) {
    const std::string search_params = R"(
    {
        "hgraph": {
            "ef_search": 80,
            "use_mci": true,
            "mci_seed_ratio": 0.1,
            "hgraph_valid_ratio_threshold": 0.5
        }
    }
    )";
    auto filter = std::make_shared<ModuloFilter>(8);
    auto result = index->KnnSearch(query, kTopK, search_params, filter);
    check_expected(result, "filtered HGraph MCI search failed: ");

    auto dataset = result.value();
    std::cout << "[search] top-" << kTopK << " filtered results:" << std::endl;
    for (int64_t i = 0; i < dataset->GetDim(); ++i) {
        std::cout << "    rank " << i << ": id=" << dataset->GetIds()[i]
                  << ", dist=" << dataset->GetDistances()[i] << std::endl;
    }
    std::cout << "[search] statistics: " << dataset->GetStatistics() << std::endl;
}

}  // namespace

int
main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    vsag::init();
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);

    std::vector<int64_t> ids;
    std::vector<float> vectors;
    auto base = make_base_dataset(ids, vectors);

    std::vector<float> query_vector;
    auto query = make_query_dataset(query_vector);

    build_and_save(engine, base);
    auto loaded = load_index(engine);
    run_filtered_search(loaded, query);

    engine.Shutdown();
    return 0;
}
