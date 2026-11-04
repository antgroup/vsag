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
#include <cmath>
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
    const std::string brute_params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 129
    })";
    const std::string search_params = R"({"hgraph":{"ef_search":200}})";

    auto base_data = GeneratePacked(count, dim, 20260102);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);

    auto build_dataset = MakeDataset(base_data, ids, dim);

    auto brute_made = vsag::Factory::CreateIndex("brute_force", brute_params);
    REQUIRE(brute_made.has_value());
    auto brute = brute_made.value();
    REQUIRE(brute->Build(build_dataset).has_value());

    const std::vector<int64_t> retrieved_ids = {0, 7, count - 1};
    auto retrieved =
        brute->GetDataByIds(retrieved_ids.data(), static_cast<int64_t>(retrieved_ids.size()));
    REQUIRE(retrieved.has_value());
    REQUIRE(retrieved.value()->GetFloat32Vectors() == nullptr);
    REQUIRE(retrieved.value()->GetBinaryVectors() != nullptr);
    for (uint64_t i = 0; i < retrieved_ids.size(); ++i) {
        const auto* expected = base_data.data() + retrieved_ids[i] * num_bytes;
        const auto* actual = retrieved.value()->GetBinaryVectors() + i * num_bytes;
        REQUIRE(std::memcmp(expected, actual, num_bytes) == 0);
    }
    auto raw_retrieved =
        brute->GetRawVectorByIds(retrieved_ids.data(), static_cast<int64_t>(retrieved_ids.size()));
    REQUIRE(raw_retrieved.has_value());
    REQUIRE(raw_retrieved.value()->GetFloat32Vectors() == nullptr);
    REQUIRE(raw_retrieved.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(base_data.data() + 7 * num_bytes,
                        raw_retrieved.value()->GetBinaryVectors() + num_bytes,
                        num_bytes) == 0);

    // Auxiliary float storage must not divert a binary query to the float-only overload.
    std::vector<float> auxiliary_float(dim, 0.0F);
    auto mixed_query = vsag::Dataset::Make()
                           ->NumElements(1)
                           ->Dim(dim)
                           ->BinaryVectors(base_data.data())
                           ->Float32Vectors(auxiliary_float.data())
                           ->Owner(false);
    auto mixed_distance = brute->CalcDistanceById(mixed_query, 7);
    REQUIRE(mixed_distance.has_value());
    const auto mixed_expected =
        ReferenceHamming(base_data.data(), base_data.data() + 7 * num_bytes, num_bytes);
    REQUIRE(mixed_distance.value() == static_cast<float>(mixed_expected));

    auto binary_query = vsag::Dataset::Make()
                            ->NumElements(1)
                            ->Dim(dim)
                            ->BinaryVectors(base_data.data())
                            ->Owner(false);
    auto batch_distances = brute->CalcDistancesById(
        binary_query, retrieved_ids.data(), static_cast<int64_t>(retrieved_ids.size()));
    REQUIRE(batch_distances.has_value());
    for (uint64_t i = 0; i < retrieved_ids.size(); ++i) {
        const auto expected = ReferenceHamming(
            base_data.data(), base_data.data() + retrieved_ids[i] * num_bytes, num_bytes);
        REQUIRE(batch_distances.value()->GetDistances()[i] == static_cast<float>(expected));
    }

    auto graph_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(graph_made.has_value());
    auto graph = graph_made.value();
    REQUIRE_FALSE(graph->CheckFeature(vsag::IndexFeature::NEED_TRAIN));
    REQUIRE(graph->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH));
    REQUIRE(graph->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER));
    REQUIRE(graph->Build(build_dataset).has_value());
    REQUIRE(graph->GetNumElements() == count);
    auto graph_raw =
        graph->GetRawVectorByIds(retrieved_ids.data(), static_cast<int64_t>(retrieved_ids.size()));
    REQUIRE(graph_raw.has_value());
    REQUIRE(graph_raw.value()->GetFloat32Vectors() == nullptr);
    REQUIRE(graph_raw.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(base_data.data() + 7 * num_bytes,
                        graph_raw.value()->GetBinaryVectors() + num_bytes,
                        num_bytes) == 0);

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
            auto expected = ReferenceHamming(
                query_vec, base_data.data() + static_cast<uint64_t>(id) * num_bytes, num_bytes);
            REQUIRE(approx.value()->GetDistances()[i] == static_cast<float>(expected));
        }

        // Range search uses "distance <= threshold" for Hamming.
        auto range = graph->RangeSearch(query, 40.0F, search_params);
        REQUIRE(range.has_value());
        for (int64_t i = 0; i < range.value()->GetDim(); ++i) {
            const auto id = range.value()->GetIds()[i];
            auto expected = ReferenceHamming(
                query_vec, base_data.data() + static_cast<uint64_t>(id) * num_bytes, num_bytes);
            REQUIRE(expected <= 40);
            REQUIRE(range.value()->GetDistances()[i] == static_cast<float>(expected));
        }

        auto dist = graph->CalcDistanceById(query, q * 31);
        REQUIRE(dist.has_value());
        auto expected = ReferenceHamming(
            query_vec, base_data.data() + static_cast<uint64_t>(q * 31) * num_bytes, num_bytes);
        REQUIRE(dist.value() == static_cast<float>(expected));
    }
    const auto recall = static_cast<double>(recall_hits) / static_cast<double>(expected_hits);
    REQUIRE(recall >= 0.75);

    // Forced updates must retain packed extraction and encoding.
    auto replacement = GeneratePacked(1, dim, 999);
    const std::vector<int64_t> replacement_id = {4};
    auto replacement_dataset = MakeDataset(replacement, replacement_id, dim);
    auto updated = graph->UpdateVector(replacement_id[0], replacement_dataset, true);
    REQUIRE(updated.has_value());
    REQUIRE(updated.value());
    auto replacement_query = vsag::Dataset::Make()
                                 ->NumElements(1)
                                 ->Dim(dim)
                                 ->BinaryVectors(replacement.data())
                                 ->Owner(false);
    auto updated_distance = graph->CalcDistanceById(replacement_query, replacement_id[0]);
    REQUIRE(updated_distance.has_value());
    REQUIRE(updated_distance.value() == 0.0F);

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

TEST_CASE("(PR) HGraph Binary Hamming Ordinary Update And Tune", "[ft][hgraph][pr]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    constexpr uint64_t num_bytes = (dim + 7) / 8;
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 64,
        "index_param": {}
    })";
    auto base_data = GeneratePacked(count, dim, 2468);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto index_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index_made.has_value());
    auto index = index_made.value();
    REQUIRE(index->Build(MakeDataset(base_data, ids, dim)).has_value());

    const std::vector<int64_t> first_id = {0};
    std::vector<uint8_t> unchanged(base_data.begin(), base_data.begin() + num_bytes);
    auto unchanged_result = index->UpdateVector(first_id[0], MakeDataset(unchanged, first_id, dim));
    REQUIRE(unchanged_result.has_value());
    REQUIRE(unchanged_result.value());

    const std::vector<int64_t> second_id = {1};
    std::vector<uint8_t> changed(base_data.begin() + num_bytes, base_data.begin() + 2 * num_bytes);
    changed[0] ^= 0x01U;
    auto changed_result = index->UpdateVector(second_id[0], MakeDataset(changed, second_id, dim));
    REQUIRE(changed_result.has_value());

    auto raw = index->GetRawVectorByIds(first_id.data(), 1);
    REQUIRE(raw.has_value());
    REQUIRE(raw.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(base_data.data(), raw.value()->GetBinaryVectors(), num_bytes) == 0);

    auto tuned = index->Tune(R"({"index_param":{"use_reorder":true}})");
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->BinaryVectors(base_data.data())
                     ->Owner(false);
    auto distance = index->CalcDistanceById(query, first_id[0]);
    REQUIRE(distance.has_value());
    REQUIRE(distance.value() == 0.0F);
}

TEST_CASE("(PR) HGraph Binary Raw Store Follows Forced Updates", "[ft][hgraph][pr]") {
    constexpr int64_t dim = 8;
    constexpr int64_t count = 32;
    constexpr uint64_t num_bytes = (dim + 7) / 8;
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 8,
        "index_param": {
            "store_raw_vector": true
        }
    })";
    auto base_data = GeneratePacked(count, dim, 1357);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto index_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index_made.has_value());
    auto index = index_made.value();
    REQUIRE(index->Build(MakeDataset(base_data, ids, dim)).has_value());

    const std::vector<int64_t> update_id = {3};
    std::vector<uint8_t> replacement(base_data.begin() + update_id[0] * num_bytes,
                                     base_data.begin() + (update_id[0] + 1) * num_bytes);
    replacement[0] ^= 0x01U;
    auto updated =
        index->UpdateVector(update_id[0], MakeDataset(replacement, update_id, dim), true);
    REQUIRE(updated.has_value());
    REQUIRE(updated.value());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->BinaryVectors(replacement.data())
                     ->Owner(false);
    auto distance = index->CalcDistanceById(query, update_id[0]);
    REQUIRE(distance.has_value());
    REQUIRE(distance.value() == 0.0F);

    auto stored = index->GetDataByIds(update_id.data(), 1);
    REQUIRE(stored.has_value());
    REQUIRE(stored.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(replacement.data(), stored.value()->GetBinaryVectors(), num_bytes) == 0);
    auto raw = index->GetRawVectorByIds(update_id.data(), 1);
    REQUIRE(raw.has_value());
    REQUIRE(raw.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(replacement.data(), raw.value()->GetBinaryVectors(), num_bytes) == 0);

    auto tuned = index->Tune(R"({"index_param":{"use_reorder":true}})");
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());
    replacement[0] ^= 0x02U;
    updated = index->UpdateVector(update_id[0], MakeDataset(replacement, update_id, dim), true);
    REQUIRE(updated.has_value());
    REQUIRE(updated.value());
    raw = index->GetRawVectorByIds(update_id.data(), 1);
    REQUIRE(raw.has_value());
    REQUIRE(raw.value()->GetBinaryVectors() != nullptr);
    REQUIRE(std::memcmp(replacement.data(), raw.value()->GetBinaryVectors(), num_bytes) == 0);

    // Reorder also exercises binary base/precise distances in the analyzer.
    REQUIRE_NOTHROW(index->GetStats());
}

TEST_CASE("(PR) Float HGraph SQ8 Training Ignores Auxiliary Binary Storage", "[ft][hgraph][pr]") {
    constexpr int64_t dim = 8;
    constexpr int64_t count = 1024;
    const std::string params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "index_param": {
            "base_quantization_type": "sq8",
            "train_sample_count": 512
        }
    })";
    std::vector<float> vectors(count * dim);
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>((i * 17 + d * 31) % 101) / 10.0F;
        }
    }
    auto auxiliary_binary = GeneratePacked(count, dim, 9753);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto mixed_dataset = vsag::Dataset::Make()
                             ->NumElements(count)
                             ->Dim(dim)
                             ->Ids(ids.data())
                             ->Float32Vectors(vectors.data())
                             ->BinaryVectors(auxiliary_binary.data())
                             ->Owner(false);
    auto index_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index_made.has_value());
    auto index = index_made.value();
    REQUIRE(index->Build(mixed_dataset).has_value());

    constexpr int64_t query_id = 17;
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data() + query_id * dim)
                     ->Owner(false);
    auto base_distance = index->CalcDistanceById(query, query_id, false);
    REQUIRE(base_distance.has_value());
    // Untrained SQ8 bounds produce a non-finite distance here; trained bounds must not.
    REQUIRE(std::isfinite(base_distance.value()));
    REQUIRE(base_distance.value() >= 0.0F);
}

TEST_CASE("(PR) Binary HGraph Analysis Uses Packed Queries", "[ft][hgraph][pr]") {
    constexpr int64_t dim = 65;
    constexpr int64_t count = 32;
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 65,
        "index_param": {}
    })";
    auto base_data = GeneratePacked(count, dim, 8642);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto index_made = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE(index_made.has_value());
    auto index = index_made.value();
    REQUIRE(index->Build(MakeDataset(base_data, ids, dim)).has_value());
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_STATS));

    std::string stats;
    REQUIRE_NOTHROW(stats = index->GetStats());
    REQUIRE_FALSE(stats.empty());

    auto queries = vsag::Dataset::Make()
                       ->NumElements(4)
                       ->Dim(dim)
                       ->BinaryVectors(base_data.data())
                       ->Owner(false);
    vsag::SearchRequest request;
    request.query_ = queries;
    request.topk_ = 4;
    request.params_str_ = R"({"hgraph":{"ef_search":32}})";
    std::string analysis;
    REQUIRE_NOTHROW(analysis = index->AnalyzeIndexBySearch(request));
    REQUIRE_FALSE(analysis.empty());

    auto short_data = GeneratePacked(4, dim - 1, 8643);
    auto wrong_dim_queries = vsag::Dataset::Make()
                                 ->NumElements(4)
                                 ->Dim(dim - 1)
                                 ->BinaryVectors(short_data.data())
                                 ->Owner(false);
    request.query_ = wrong_dim_queries;
    REQUIRE_THROWS(index->AnalyzeIndexBySearch(request));

    auto missing_vectors = vsag::Dataset::Make()->NumElements(4)->Dim(dim)->Owner(false);
    request.query_ = missing_vectors;
    REQUIRE_THROWS(index->AnalyzeIndexBySearch(request));
}

TEST_CASE("(PR) Binary HGraph Rejects MCI During Validation", "[ft][hgraph][pr]") {
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 64,
        "index_param": {
            "use_mci": true
        }
    })";
    auto index = vsag::Factory::CreateIndex("hgraph", params);
    REQUIRE_FALSE(index.has_value());
    REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("(PR) BruteForce Binary Hamming Update And Force Remove", "[ft][bruteforce][pr]") {
    constexpr int64_t dim = 65;
    constexpr uint64_t num_bytes = (dim + 7) / 8;
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 65
    })";
    const std::vector<int64_t> ids = {10, 20, 30, 40};
    auto base_data = GeneratePacked(ids.size(), dim, 1234);
    auto index_made = vsag::Factory::CreateIndex("brute_force", params);
    REQUIRE(index_made.has_value());
    auto index = index_made.value();
    REQUIRE(index->Build(MakeDataset(base_data, ids, dim)).has_value());

    auto replacement = GeneratePacked(1, dim, 5678);
    replacement.back() ^= 0x80U;  // Unused tail bits must be masked during re-encoding.
    const std::vector<int64_t> update_id = {20};
    auto update = index->UpdateVector(update_id[0], MakeDataset(replacement, update_id, dim));
    REQUIRE(update.has_value());
    REQUIRE(update.value());
    auto replacement_query = vsag::Dataset::Make()
                                 ->NumElements(1)
                                 ->Dim(dim)
                                 ->BinaryVectors(replacement.data())
                                 ->Owner(false);
    auto updated_distance = index->CalcDistanceById(replacement_query, update_id[0]);
    REQUIRE(updated_distance.has_value());
    REQUIRE(updated_distance.value() == 0.0F);

    // Removing a non-last slot decodes and reinserts the last packed row.
    auto removed = index->Remove(ids.front(), vsag::RemoveMode::FORCE_REMOVE);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() == 1);
    auto moved_query = vsag::Dataset::Make()
                           ->NumElements(1)
                           ->Dim(dim)
                           ->BinaryVectors(base_data.data() + 3 * num_bytes)
                           ->Owner(false);
    auto moved_distance = index->CalcDistanceById(moved_query, ids.back());
    REQUIRE(moved_distance.has_value());
    REQUIRE(moved_distance.value() == 0.0F);
}

TEST_CASE("(PR) Binary WARP Is Rejected", "[ft][warp][pr]") {
    const std::string params = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 64
    })";
    auto index = vsag::Factory::CreateIndex("warp", params);
    REQUIRE_FALSE(index.has_value());
    REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}
