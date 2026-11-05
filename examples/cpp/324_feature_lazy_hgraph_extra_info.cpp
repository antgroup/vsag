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
//
// Example: LazyHGraph with extra_info filtering.
//
// LazyHGraph starts in an exact BruteForce-backed flat phase and converts to
// HGraph after transition_threshold is reached. The extra_info API and search
// parameters are the same in both phases.

#include <vsag/vsag.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct ItemExtraInfo {
    uint32_t category_id{0};

    void
    Serialize(char* buffer) const {
        std::memcpy(buffer, &category_id, sizeof(category_id));
    }

    void
    Deserialize(const char* buffer) {
        std::memcpy(&category_id, buffer, sizeof(category_id));
    }
};

class CategoryFilter : public vsag::Filter {
public:
    explicit CategoryFilter(uint32_t category_id) : category_id_(category_id) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t) const override {
        // When use_extra_info_filter is enabled, LazyHGraph calls CheckValid(const char*).
        return false;
    }

    [[nodiscard]] bool
    CheckValid(const char* data) const override {
        if (data == nullptr) {
            return false;
        }
        ItemExtraInfo info;
        info.Deserialize(data);
        return info.category_id == category_id_;
    }

private:
    uint32_t category_id_;
};

vsag::DatasetPtr
MakeDataset(int64_t first_id,
            int64_t count,
            int64_t dim,
            uint64_t extra_info_size,
            std::vector<int64_t>& ids,
            std::vector<float>& vectors,
            std::vector<char>& extra_infos) {
    ids.resize(count);
    vectors.resize(count * dim);
    extra_infos.resize(static_cast<uint64_t>(count) * extra_info_size);

    for (int64_t i = 0; i < count; ++i) {
        ids[i] = first_id + i;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = static_cast<float>((first_id + i) * 0.01 + j);
        }

        ItemExtraInfo info;
        info.category_id = (i % 2 == 0) ? 1U : 7U;
        info.Serialize(extra_infos.data() + static_cast<uint64_t>(i) * extra_info_size);
    }

    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->ExtraInfos(extra_infos.data())
        ->ExtraInfoSize(static_cast<int64_t>(extra_info_size))
        ->Owner(false);
}

bool
PrintSearchResult(const std::string& phase_name,
                  const vsag::DatasetPtr& result,
                  uint64_t extra_info_size) {
    if (result == nullptr) {
        std::cerr << phase_name << " search returned a null result" << std::endl;
        return false;
    }

    std::cout << phase_name << " filtered results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        ItemExtraInfo info;
        info.Deserialize(result->GetExtraInfos() + static_cast<uint64_t>(i) * extra_info_size);
        std::cout << "  id=" << result->GetIds()[i] << ", distance=" << result->GetDistances()[i]
                  << ", category_id=" << info.category_id << std::endl;
    }
    return true;
}

}  // namespace

int
main() {
    vsag::init();

    constexpr int64_t dim = 8;
    constexpr uint64_t extra_info_size = sizeof(ItemExtraInfo);
    constexpr uint64_t transition_threshold = 8;

    std::string lazy_hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "extra_info_size": 4,
        "lazy_hgraph": {
            "transition_threshold": 8,
            "hgraph": {
                "base_quantization_type": "fp32",
                "max_degree": 8,
                "ef_construction": 40,
                "build_thread_count": 1
            }
        }
    }
    )";

    auto index_res = vsag::Factory::CreateIndex("lazy_hgraph", lazy_hgraph_build_parameters);
    if (not index_res.has_value()) {
        std::cerr << "Failed to create LazyHGraph: " << index_res.error().message << std::endl;
        return -1;
    }
    auto index = index_res.value();

    if (not index->CheckFeature(vsag::SUPPORT_KNN_SEARCH_WITH_EX_FILTER)) {
        std::cerr << "LazyHGraph extra_info filtering is not available" << std::endl;
        return -1;
    }

    std::vector<int64_t> small_ids;
    std::vector<float> small_vectors;
    std::vector<char> small_extra_infos;
    auto small_batch =
        MakeDataset(0, 4, dim, extra_info_size, small_ids, small_vectors, small_extra_infos);
    if (auto add_result = index->Add(small_batch); not add_result.has_value()) {
        std::cerr << "Failed to add flat-phase data: " << add_result.error().message << std::endl;
        return -1;
    }
    std::cout << "After small Add(), LazyHGraph contains " << index->GetNumElements()
              << " vectors and remains in the flat phase." << std::endl;

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(small_vectors.data())
                     ->Owner(false);
    auto category_filter = std::make_shared<CategoryFilter>(7);
    const char* search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 40,
            "use_extra_info_filter": true
        }
    }
    )";

    auto flat_result = index->KnnSearch(query, 2, search_parameters, category_filter);
    if (not flat_result.has_value() or
        not PrintSearchResult("Flat phase", flat_result.value(), extra_info_size)) {
        std::cerr << "Flat-phase filtered search failed" << std::endl;
        return -1;
    }

    std::vector<int64_t> grow_ids;
    std::vector<float> grow_vectors;
    std::vector<char> grow_extra_infos;
    auto grow_batch =
        MakeDataset(4,
                    static_cast<int64_t>(transition_threshold) - small_batch->GetNumElements(),
                    dim,
                    extra_info_size,
                    grow_ids,
                    grow_vectors,
                    grow_extra_infos);
    if (auto add_result = index->Add(grow_batch); not add_result.has_value()) {
        std::cerr << "Failed to add graph-phase data: " << add_result.error().message << std::endl;
        return -1;
    }
    std::cout << "After reaching transition_threshold, LazyHGraph contains "
              << index->GetNumElements() << " vectors and has converted to HGraph internally."
              << std::endl;

    auto graph_result = index->KnnSearch(query, 2, search_parameters, category_filter);
    if (not graph_result.has_value() or
        not PrintSearchResult("Graph phase", graph_result.value(), extra_info_size)) {
        std::cerr << "Graph-phase filtered search failed" << std::endl;
        return -1;
    }

    std::vector<int64_t> extra_info_ids = {graph_result.value()->GetIds()[0]};
    std::vector<char> fetched_extra_info(extra_info_size);
    auto get_result = index->GetExtraInfoByIds(extra_info_ids.data(), 1, fetched_extra_info.data());
    if (not get_result.has_value()) {
        std::cerr << "Failed to get extra_info: " << get_result.error().message << std::endl;
        return -1;
    }

    ItemExtraInfo fetched_info;
    fetched_info.Deserialize(fetched_extra_info.data());
    std::cout << "Fetched extra_info by id: id=" << extra_info_ids[0]
              << ", category_id=" << fetched_info.category_id << std::endl;

    return 0;
}
