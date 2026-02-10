
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

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Define extra_info structure for user data
// In this example, we store: uint32_t category_id + uint32_t timestamp + float score
struct ItemExtraInfo {
    uint32_t category_id;
    uint32_t timestamp;
    float score;

    void
    Serialize(char* buffer) const {
        memcpy(buffer, &category_id, sizeof(category_id));
        memcpy(buffer + sizeof(category_id), &timestamp, sizeof(timestamp));
        memcpy(buffer + sizeof(category_id) + sizeof(timestamp), &score, sizeof(score));
    }

    void
    Deserialize(const char* buffer) {
        memcpy(&category_id, buffer, sizeof(category_id));
        memcpy(&timestamp, buffer + sizeof(category_id), sizeof(timestamp));
        memcpy(&score, buffer + sizeof(category_id) + sizeof(timestamp), sizeof(score));
    }
};

// Custom filter based on extra_info
// When use_extra_info_filter is enabled in HGraph search parameters,
// HGraph will call CheckValid(const char* data) with the extra_info data
// of each candidate vector during the search process.
class CategoryFilter : public vsag::Filter {
public:
    CategoryFilter(uint32_t min_category, uint32_t max_category)
        : min_category_(min_category), max_category_(max_category) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        // This method is not used when use_extra_info_filter is enabled
        // HGraph uses CheckValid(const char* data) instead
        return true;
    }

    [[nodiscard]] bool
    CheckValid(const char* data) const override {
        // This method is called by HGraph when use_extra_info_filter is enabled.
        // 'data' points to the extra_info bytes of the candidate vector.
        // We deserialize the category_id from the extra_info and check if it's in range.
        uint32_t category_id;
        memcpy(&category_id, data, sizeof(category_id));
        return category_id >= min_category_ && category_id <= max_category_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return 0.5F;  // Assume 50% valid ratio
    }

    // Helper method to manually check if category is in range (for demonstration)
    [[nodiscard]] bool
    IsCategoryInRange(uint32_t category_id) const {
        return category_id >= min_category_ && category_id <= max_category_;
    }

private:
    uint32_t min_category_;
    uint32_t max_category_;
};

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Configuration *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    const uint64_t extra_info_size = sizeof(ItemExtraInfo);

    /******************* Prepare Base Dataset *****************/
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::vector<char> extra_infos(num_vectors * extra_info_size);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    std::uniform_int_distribution<uint32_t> category_dist(1, 10);
    std::uniform_int_distribution<uint32_t> timestamp_dist(1609459200, 1704067200);  // 2021-2024

    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;

        // Generate extra_info for each item
        ItemExtraInfo info;
        info.category_id = category_dist(rng);
        info.timestamp = timestamp_dist(rng);
        info.score = distrib_real(rng);
        info.Serialize(extra_infos.data() + i * extra_info_size);
    }

    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->ExtraInfos(extra_infos.data())
        ->ExtraInfoSize(static_cast<int64_t>(extra_info_size))
        ->Owner(false);

    std::cout << "Dataset prepared with " << num_vectors << " vectors\n";
    std::cout << "Extra info size: " << extra_info_size << " bytes per item\n";

    /******************* Create HGraph Index with Extra Info Support *****************/
    std::string hgraph_build_parameters = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "extra_info_size": 12,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "alpha": 1.2
        }
    })";

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    vsag::Options::Instance().set_block_size_limit(2 * 1024 * 1024);

    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build HGraph Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "Build successful! Index contains: " << index->GetNumElements()
                  << " elements\n";
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << "\n";
        exit(-1);
    }

    /******************* Feature 1: Retrieve Extra Info by ID *****************/
    std::cout << "\n=== Feature 1: Retrieve Extra Info by ID ===\n";

    // Check if index supports getting extra_info
    if (index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
        // Query extra_info for first 5 items
        int64_t query_count = 5;
        std::vector<int64_t> query_ids(query_count);
        for (int64_t i = 0; i < query_count; ++i) {
            query_ids[i] = i;
        }

        std::vector<char> retrieved_extra_infos(
            static_cast<size_t>(query_count * static_cast<int64_t>(extra_info_size)));
        auto result =
            index->GetExtraInfoByIds(query_ids.data(), query_count, retrieved_extra_infos.data());

        if (result.has_value()) {
            std::cout << "Successfully retrieved extra info for " << query_count << " items:\n";
            for (int64_t i = 0; i < query_count; ++i) {
                ItemExtraInfo info;
                info.Deserialize(retrieved_extra_infos.data() +
                                 i * static_cast<int64_t>(extra_info_size));
                std::cout << "  Item " << query_ids[i] << ": category_id=" << info.category_id
                          << ", timestamp=" << info.timestamp << ", score=" << info.score << "\n";
            }
        } else {
            std::cerr << "Failed to get extra info: " << result.error().message << "\n";
        }
    } else {
        std::cout << "Index does not support getting extra info by ID\n";
    }

    /******************* Feature 2: Search with Extra Info Filter *****************/
    std::cout << "\n=== Feature 2: Search with Extra Info Filter ===\n";

    // Prepare query
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    int64_t topk = 10;
    uint32_t min_category = 3;
    uint32_t max_category = 7;

    // Create a custom filter that filters by category range
    auto category_filter = std::make_shared<CategoryFilter>(min_category, max_category);

    // First, let's demonstrate manual filtering without use_extra_info_filter
    std::cout << "\n--- Manual filtering demonstration (without use_extra_info_filter) ---\n";
    const char* hgraph_search_parameters = R"({
        "hgraph": {
            "ef_search": 100
        }
    })";

    auto search_result = index->KnnSearch(query, topk * 2, hgraph_search_parameters);
    if (search_result.has_value() && index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
        auto result = search_result.value();

        // Manually filter results based on extra_info
        std::cout << "Search results (top " << result->GetDim() << ") and manual category filter ["
                  << min_category << "-" << max_category << "]:\n";

        int64_t result_count = result->GetDim();
        std::vector<int64_t> result_ids(result_count);
        for (int64_t i = 0; i < result_count; ++i) {
            result_ids[i] = result->GetIds()[i];
        }

        std::vector<char> result_extra_infos(
            static_cast<size_t>(result_count * static_cast<int64_t>(extra_info_size)));
        auto extra_info_result =
            index->GetExtraInfoByIds(result_ids.data(), result_count, result_extra_infos.data());

        if (extra_info_result.has_value()) {
            int filtered_count = 0;
            for (int64_t i = 0; i < result_count && filtered_count < topk; ++i) {
                ItemExtraInfo info;
                info.Deserialize(result_extra_infos.data() +
                                 i * static_cast<int64_t>(extra_info_size));
                bool in_range = category_filter->IsCategoryInRange(info.category_id);
                std::cout << "  Item " << result_ids[i]
                          << ": distance=" << result->GetDistances()[i]
                          << ", category_id=" << info.category_id << " (in range [" << min_category
                          << "-" << max_category << "]: " << (in_range ? "YES" : "NO") << ")\n";
                if (in_range) {
                    filtered_count++;
                }
            }
        }
    }

    // Now demonstrate using use_extra_info_filter parameter
    // Note: When use_extra_info_filter is true, HGraph uses extra_info data internally
    // during search to filter candidates, which is more efficient than post-filtering
    std::cout << "\n--- Using use_extra_info_filter parameter ---\n";
    const char* hgraph_filter_search_parameters = R"({
        "hgraph": {
            "ef_search": 100,
            "use_extra_info_filter": true
        }
    })";

    std::cout << "Searching with use_extra_info_filter enabled...\n";
    std::cout << "Note: When enabled, HGraph internally uses extra_info to filter\n";
    std::cout << "      candidates during the search process.\n\n";

    auto filter_search_result =
        index->KnnSearch(query, topk, hgraph_filter_search_parameters, category_filter);
    if (filter_search_result.has_value()) {
        auto result = filter_search_result.value();
        std::cout << "Search with extra_info filter - Top " << result->GetDim() << " results:\n";
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << "  " << result->GetIds()[i] << ": " << result->GetDistances()[i] << "\n";
        }

        // Verify filtered results by retrieving extra_info
        if (index->CheckFeature(vsag::SUPPORT_GET_EXTRA_INFO_BY_ID)) {
            int64_t result_count = result->GetDim();
            std::vector<int64_t> result_ids(result_count);
            for (int64_t i = 0; i < result_count; ++i) {
                result_ids[i] = result->GetIds()[i];
            }

            std::vector<char> result_extra_infos(
                static_cast<size_t>(result_count * static_cast<int64_t>(extra_info_size)));
            auto extra_info_result = index->GetExtraInfoByIds(
                result_ids.data(), result_count, result_extra_infos.data());

            if (extra_info_result.has_value()) {
                std::cout << "\nExtra info for filtered search results:\n";
                for (int64_t i = 0; i < result_count; ++i) {
                    ItemExtraInfo info;
                    info.Deserialize(result_extra_infos.data() +
                                     i * static_cast<int64_t>(extra_info_size));
                    std::cout << "  Item " << result_ids[i] << ": category_id=" << info.category_id
                              << ", timestamp=" << info.timestamp << ", score=" << info.score
                              << "\n";
                }
            }
        }
    } else {
        std::cerr << "Filtered search failed: " << filter_search_result.error().message << "\n";
    }

    /******************* Feature 3: Update Extra Info *****************/
    std::cout << "\n=== Feature 3: Update Extra Info ===\n";

    if (index->CheckFeature(vsag::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT)) {
        // Update extra_info for item 0
        ItemExtraInfo updated_info;
        updated_info.category_id = 99;
        updated_info.timestamp = 999999999;
        updated_info.score = 0.99F;

        std::vector<char> updated_extra_info(extra_info_size);
        updated_info.Serialize(updated_extra_info.data());

        auto update_dataset = vsag::Dataset::Make();
        update_dataset->NumElements(1)
            ->Ids(ids.data())
            ->ExtraInfos(updated_extra_info.data())
            ->ExtraInfoSize(static_cast<int64_t>(extra_info_size))
            ->Owner(false);

        auto update_result = index->UpdateExtraInfo(update_dataset);
        if (update_result.has_value()) {
            std::cout << "Successfully updated extra info for item 0\n";

            // Verify the update
            std::vector<int64_t> verify_id = {0};
            std::vector<char> verify_extra_info(extra_info_size);
            auto verify_result =
                index->GetExtraInfoByIds(verify_id.data(), 1, verify_extra_info.data());
            if (verify_result.has_value()) {
                ItemExtraInfo info;
                info.Deserialize(verify_extra_info.data());
                std::cout << "Verified - Item 0: category_id=" << info.category_id
                          << ", timestamp=" << info.timestamp << ", score=" << info.score << "\n";
            }
        } else {
            std::cerr << "Failed to update extra info: " << update_result.error().message << "\n";
        }
    } else {
        std::cout << "Index does not support updating extra info\n";
    }

    std::cout << "\n=== Example completed successfully! ===\n";

    engine.Shutdown();
    return 0;
}
