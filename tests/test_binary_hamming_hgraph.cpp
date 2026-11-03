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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "vsag/vsag.h"

namespace {

uint64_t
ReferenceHamming(const uint8_t* x, const uint8_t* y, uint64_t num_bytes) {
    uint64_t result = 0;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        auto diff = static_cast<uint8_t>(x[i] ^ y[i]);
        while (diff != 0) {
            diff = static_cast<uint8_t>(diff & (diff - 1));
            ++result;
        }
    }
    return result;
}

std::vector<uint8_t>
GeneratePacked(uint64_t count, uint64_t bits, uint64_t seed) {
    const uint64_t bytes = (bits + 7) / 8;
    std::vector<uint8_t> data(count * bytes);
    uint64_t state = seed;
    for (auto& byte : data) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        byte = static_cast<uint8_t>(state >> 33);
    }
    if (bits % 8 != 0) {
        const auto mask = static_cast<uint8_t>((1U << (bits % 8U)) - 1U);
        for (uint64_t i = 0; i < count; ++i) {
            data[(i + 1) * bytes - 1] &= mask;
        }
    }
    return data;
}

vsag::DatasetPtr
MakeDataset(const std::vector<uint8_t>& data, const std::vector<int64_t>& ids, int64_t dim) {
    return vsag::Dataset::Make()
        ->NumElements(static_cast<int64_t>(ids.size()))
        ->Dim(dim)
        ->Ids(ids.data())
        ->BinaryVectors(data.data())
        ->Owner(false);
}

}  // namespace

TEST_CASE("(PR) HGraph Binary Hamming Build Search Add Remove Serialize",
          "[ft][build][hgraph][pr]") {
    constexpr int64_t dim = 129;  // non-byte-aligned logical bit dimension
    constexpr int64_t count = 600;
    constexpr uint64_t num_bytes = (dim + 7) / 8;

    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 129,
        "index_param": {
            "base_quantization_type": "binary",
            "support_remove": true
        }
    })";
    const std::string search_params = R"({"hgraph":{"ef_search":200}})";

    auto base_data = GeneratePacked(count, dim, 20260102);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);

    auto build_dataset = MakeDataset(base_data, ids, dim);

    auto brute_made = vsag::Factory::CreateIndex("brute_force", params);
    REQUIRE(brute_made.has_value());
    auto brute = brute_made.value();
    REQUIRE(brute->Build(build_dataset).has_value());

    auto graph_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(graph_made.has_value());
    auto graph = graph_made.value();
    REQUIRE(graph->Build(build_dataset).has_value());
    REQUIRE(graph->GetNumElements() == count);

    auto queries = GeneratePacked(16, dim, 777);
    uint64_t recall_hits = 0;
    uint64_t expected_hits = 0;
    for (int64_t q = 0; q < 16; ++q) {
        const auto* query_vec = queries.data() + static_cast<uint64_t>(q) * num_bytes;
        auto query =
            vsag::Dataset::Make()->NumElements(1)->Dim(dim)->BinaryVectors(query_vec)->Owner(false);

        auto exact = brute->KnnSearch(query, 10, "{}");
        auto approx = graph->KnnSearch(query, 10, search_params);
        REQUIRE(exact.has_value());
        REQUIRE(approx.has_value());
        std::unordered_set<int64_t> exact_ids;
        for (int64_t i = 0; i < exact.value()->GetDim(); ++i) {
            exact_ids.insert(exact.value()->GetIds()[i]);
        }
        for (int64_t i = 0; i < approx.value()->GetDim(); ++i) {
            const auto id = approx.value()->GetIds()[i];
            if (exact_ids.count(id) != 0) {
                ++recall_hits;
            }
            ++expected_hits;
            auto expected =
                ReferenceHamming(query_vec, base_data.data() + static_cast<uint64_t>(id) * num_bytes,
                                 num_bytes);
            REQUIRE(approx.value()->GetDistances()[i] == static_cast<float>(expected));
        }

        // Range search uses "distance <= threshold" for Hamming.
        auto range = graph->RangeSearch(query, 40.0F, search_params);
        REQUIRE(range.has_value());
        for (int64_t i = 0; i < range.value()->GetDim(); ++i) {
            const auto id = range.value()->GetIds()[i];
            auto expected =
                ReferenceHamming(query_vec, base_data.data() + static_cast<uint64_t>(id) * num_bytes,
                                 num_bytes);
            REQUIRE(expected <= 40);
            REQUIRE(range.value()->GetDistances()[i] == static_cast<float>(expected));
        }

        auto dist = graph->CalcDistanceById(query, q * 31);
        REQUIRE(dist.has_value());
        auto expected = ReferenceHamming(query_vec,
                                         base_data.data() + static_cast<uint64_t>(q * 31) * num_bytes,
                                         num_bytes);
        REQUIRE(dist.value() == static_cast<float>(expected));
    }
    const auto recall = static_cast<double>(recall_hits) / static_cast<double>(expected_hits);
    REQUIRE(recall >= 0.75);

    // Incremental add.
    constexpr int64_t extra = 40;
    auto extra_data = GeneratePacked(extra, dim, 888);
    std::vector<int64_t> extra_ids(extra);
    std::iota(extra_ids.begin(), extra_ids.end(), count);
    auto add_dataset = MakeDataset(extra_data, extra_ids, dim);
    REQUIRE(graph->Add(add_dataset).has_value());
    REQUIRE(graph->GetNumElements() == count + extra);

    // Remove some labels and make sure they no longer surface in search results.
    std::vector<int64_t> removed_ids = {3, 77, count + 11};
    auto removed = graph->Remove(removed_ids);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() == removed_ids.size());

    auto remove_query = vsag::Dataset::Make()
                            ->NumElements(1)
                            ->Dim(dim)
                            ->BinaryVectors(queries.data())
                            ->Owner(false);
    auto after_remove = graph->KnnSearch(remove_query, 20, search_params);
    REQUIRE(after_remove.has_value());
    for (int64_t i = 0; i < after_remove.value()->GetDim(); ++i) {
        const auto id = after_remove.value()->GetIds()[i];
        REQUIRE(std::find(removed_ids.begin(), removed_ids.end(), id) == removed_ids.end());
    }

    // Serialize and deserialize restore the same exact binary distances.
    std::stringstream stream;
    REQUIRE(graph->SerializeStreaming(stream).has_value());
    auto restored_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(restored_made.has_value());
    auto restored = restored_made.value();
    std::stringstream restore_stream(stream.str());
    REQUIRE(restored->DeserializeStreaming(restore_stream).has_value());
    REQUIRE(restored->GetNumElements() == graph->GetNumElements());

    auto original_result = graph->KnnSearch(remove_query, 10, search_params);
    auto restored_result = restored->KnnSearch(remove_query, 10, search_params);
    REQUIRE(original_result.has_value());
    REQUIRE(restored_result.has_value());
    REQUIRE(restored_result.value()->GetDim() == original_result.value()->GetDim());
    for (int64_t i = 0; i < restored_result.value()->GetDim(); ++i) {
        REQUIRE(restored_result.value()->GetIds()[i] == original_result.value()->GetIds()[i]);
        REQUIRE(restored_result.value()->GetDistances()[i] ==
                original_result.value()->GetDistances()[i]);
    }
}
