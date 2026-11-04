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

// Functional tests for bit-packed binary vectors compared with the Hamming distance
// ("dtype": "binary", "metric_type": "hamming"). Every distance is checked against an
// independent bit-by-bit scalar reference.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vsag/vsag.h"

namespace {

uint64_t
row_bytes(int64_t dim) {
    return (static_cast<uint64_t>(dim) + 7) / 8;
}

// Logical bit i is (bytes[i / 8] >> (7 - i % 8)) & 1 (MSB-first packing, numpy.packbits).
float
reference_hamming(const uint8_t* x, const uint8_t* y, int64_t dim) {
    uint64_t distance = 0;
    for (int64_t i = 0; i < dim; ++i) {
        const auto shift = 7 - (i % 8);
        distance += ((x[i / 8] >> shift) & 1U) != ((y[i / 8] >> shift) & 1U) ? 1 : 0;
    }
    return static_cast<float>(distance);
}

struct BinaryData {
    int64_t dim{0};
    std::vector<uint8_t> bytes;
    std::vector<int64_t> ids;

    [[nodiscard]] int64_t
    count() const {
        return static_cast<int64_t>(ids.size());
    }

    [[nodiscard]] const uint8_t*
    row(int64_t i) const {
        return bytes.data() + static_cast<uint64_t>(i) * row_bytes(dim);
    }
};

// Clustered data: every vector flips each bit of one of `clusters` random centers with
// probability `flip`, so nearest neighbors are meaningful. Padding bits are random too.
BinaryData
make_data(int64_t dim, int64_t count, uint64_t seed, int clusters = 16, double flip = 0.1) {
    std::mt19937_64 rng(seed);
    const auto bytes_per_row = row_bytes(dim);
    std::vector<std::vector<uint8_t>> centers(clusters, std::vector<uint8_t>(bytes_per_row));
    for (auto& center : centers) {
        for (auto& value : center) {
            value = static_cast<uint8_t>(rng());
        }
    }
    std::bernoulli_distribution flip_bit(flip);
    BinaryData data;
    data.dim = dim;
    data.bytes.resize(static_cast<uint64_t>(count) * bytes_per_row);
    for (int64_t i = 0; i < count; ++i) {
        const auto& center = centers[rng() % centers.size()];
        uint8_t* row = data.bytes.data() + static_cast<uint64_t>(i) * bytes_per_row;
        for (uint64_t b = 0; b < bytes_per_row; ++b) {
            uint8_t mask = 0;
            for (int bit = 0; bit < 8; ++bit) {
                mask = static_cast<uint8_t>(mask | (flip_bit(rng) ? (1U << bit) : 0U));
            }
            row[b] = static_cast<uint8_t>(center[b] ^ mask);
        }
        data.ids.push_back(1000 + 3 * i);
    }
    return data;
}

vsag::DatasetPtr
base_dataset(const BinaryData& data, int64_t begin = 0, int64_t end = -1) {
    end = end < 0 ? data.count() : end;
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(end - begin)
        ->Dim(data.dim)
        ->Ids(data.ids.data() + begin)
        ->BinaryVectors(data.row(begin))
        ->Owner(false);
    return dataset;
}

vsag::DatasetPtr
query_dataset(const uint8_t* query, int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(1)->Dim(dim)->BinaryVectors(query)->Owner(false);
    return dataset;
}

std::string
create_parameters(int64_t dim, const nlohmann::json& index_param = nlohmann::json::object()) {
    nlohmann::json json;
    json["dtype"] = "binary";
    json["metric_type"] = "hamming";
    json["dim"] = dim;
    json["index_param"] = index_param;
    return json.dump();
}

// Exact distances from `query` to every live vector, keyed by id.
std::map<int64_t, float>
exact_distances(const BinaryData& data, const uint8_t* query, const std::set<int64_t>& removed) {
    std::map<int64_t, float> result;
    for (int64_t i = 0; i < data.count(); ++i) {
        if (removed.count(data.ids[i]) == 0) {
            result[data.ids[i]] = reference_hamming(query, data.row(i), data.dim);
        }
    }
    return result;
}

float
kth_distance(const std::map<int64_t, float>& exact, int64_t k) {
    std::vector<float> values;
    values.reserve(exact.size());
    for (const auto& [id, distance] : exact) {
        values.push_back(distance);
    }
    std::sort(values.begin(), values.end());
    return values[std::min<uint64_t>(k, values.size()) - 1];
}

// The k-NN result of an exact index: every returned distance is the exact distance of its id,
// ids are distinct and live, and the sorted distances equal the k smallest exact distances.
void
check_exact_knn(const vsag::DatasetPtr& result, const std::map<int64_t, float>& exact, int64_t k) {
    const auto returned = result->GetDim();
    REQUIRE(returned == std::min<int64_t>(k, static_cast<int64_t>(exact.size())));
    std::vector<float> expected;
    for (const auto& [id, distance] : exact) {
        expected.push_back(distance);
    }
    std::sort(expected.begin(), expected.end());
    std::set<int64_t> seen;
    for (int64_t r = 0; r < returned; ++r) {
        const auto id = result->GetIds()[r];
        REQUIRE(exact.count(id) == 1);
        REQUIRE(seen.insert(id).second);
        REQUIRE(result->GetDistances()[r] == exact.at(id));
        REQUIRE(result->GetDistances()[r] == expected[r]);
    }
}

// The range result of an exact index: precisely the live ids with distance <= radius.
void
check_exact_range(const vsag::DatasetPtr& result,
                  const std::map<int64_t, float>& exact,
                  float radius) {
    std::set<int64_t> expected;
    for (const auto& [id, distance] : exact) {
        if (distance <= radius) {
            expected.insert(id);
        }
    }
    std::set<int64_t> returned;
    for (int64_t r = 0; r < result->GetDim(); ++r) {
        const auto id = result->GetIds()[r];
        REQUIRE(result->GetDistances()[r] == exact.at(id));
        REQUIRE(returned.insert(id).second);
    }
    REQUIRE(returned == expected);
}

// Fails every allocation, to exercise out-of-memory paths.
class FailingAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "FailingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        (void)size;
        return nullptr;
    }

    void
    Deallocate(void* p) override {
        (void)p;
    }

    void*
    Reallocate(void* p, uint64_t size) override {
        (void)p;
        (void)size;
        return nullptr;
    }
};

vsag::IndexPtr
create_index(const std::string& name, const std::string& parameters) {
    auto index = vsag::Factory::CreateIndex(name, parameters);
    REQUIRE(index.has_value());
    return index.value();
}

vsag::IndexPtr
serialize_round_trip(const vsag::IndexPtr& index,
                     const std::string& name,
                     const std::string& parameters) {
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    auto restored = create_index(name, parameters);
    REQUIRE(restored->Deserialize(stream).has_value());
    return restored;
}

}  // namespace

TEST_CASE("Binary BruteForce returns exact Hamming results", "[ft][binary][brute_force]") {
    const int64_t dim = GENERATE(1, 7, 8, 9, 63, 64, 65, 127, 128, 256, 1000);
    const int64_t count = dim < 16 ? 300 : 1200;
    const int64_t k = 10;
    auto data = make_data(dim, count, static_cast<uint64_t>(dim));
    const auto parameters = create_parameters(dim);
    auto index = create_index("brute_force", parameters);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_METRIC_TYPE_HAMMING));
    REQUIRE(index->Build(base_dataset(data)).has_value());
    REQUIRE(index->GetNumElements() == count);

    std::mt19937_64 rng(dim + 1);
    std::vector<uint8_t> query(row_bytes(dim));
    const std::set<int64_t> none;
    for (int q = 0; q < 8; ++q) {
        for (auto& value : query) {
            value = static_cast<uint8_t>(rng());
        }
        if (q % 2 == 1) {
            // Queries equal to stored vectors (with garbage in the padding bits).
            const auto* row = data.row(static_cast<int64_t>(rng() % count));
            std::copy(row, row + row_bytes(dim) - 1, query.begin());
            query.back() = static_cast<uint8_t>(row[row_bytes(dim) - 1] ^ (0xFFU >> (dim % 8)));
            if (dim % 8 == 0) {
                query.back() = row[row_bytes(dim) - 1];
            }
        }
        const auto exact = exact_distances(data, query.data(), none);
        auto knn = index->KnnSearch(query_dataset(query.data(), dim), k, "{}");
        REQUIRE(knn.has_value());
        check_exact_knn(knn.value(), exact, k);

        for (float radius : {0.0F, kth_distance(exact, k), kth_distance(exact, k) + 0.5F}) {
            auto range = index->RangeSearch(query_dataset(query.data(), dim), radius, "{}");
            REQUIRE(range.has_value());
            check_exact_range(range.value(), exact, radius);
        }

        // An ID filter keeps only even positions; the result is exact over the kept ids.
        std::set<int64_t> odd_ids;
        for (int64_t i = 1; i < count; i += 2) {
            odd_ids.insert(data.ids[i]);
        }
        // A std::function filter returns true for the ids to exclude.
        auto exclude_odd = [&odd_ids](int64_t id) { return odd_ids.count(id) != 0; };
        auto filtered = index->KnnSearch(query_dataset(query.data(), dim), k, "{}", exclude_odd);
        REQUIRE(filtered.has_value());
        check_exact_knn(filtered.value(), exact_distances(data, query.data(), odd_ids), k);

        std::vector<int64_t> ids = {data.ids[0], data.ids[count / 2], data.ids[count - 1], -7};
        auto distances = index->CalcDistancesById(
            query_dataset(query.data(), dim), ids.data(), static_cast<int64_t>(ids.size()));
        REQUIRE(distances.has_value());
        for (int64_t i = 0; i < 3; ++i) {
            REQUIRE(distances.value()->GetDistances()[i] == exact.at(ids[i]));
        }
        REQUIRE(distances.value()->GetDistances()[3] < 0.0F);
        auto single = index->CalcDistanceById(query_dataset(query.data(), dim), data.ids[1]);
        REQUIRE(single.has_value());
        REQUIRE(single.value() == exact.at(data.ids[1]));
    }

    // Stored codes are the input bits with the padding cleared.
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));
    std::vector<int64_t> wanted = {data.ids[3], data.ids[count - 2]};
    auto raw = index->GetRawVectorByIds(wanted.data(), 2);
    REQUIRE(raw.has_value());
    REQUIRE(raw.value()->GetDim() == dim);
    REQUIRE(raw.value()->GetBinaryVectors() != nullptr);
    REQUIRE(raw.value()->GetFloat32Vectors() == nullptr);
    REQUIRE(reference_hamming(raw.value()->GetBinaryVectors(), data.row(3), dim) == 0.0F);
    REQUIRE(reference_hamming(raw.value()->GetBinaryVectors() + row_bytes(dim),
                              data.row(count - 2),
                              dim) == 0.0F);
    FailingAllocator failing;
    auto no_memory = index->GetRawVectorByIds(wanted.data(), 2, &failing);
    REQUIRE_FALSE(no_memory.has_value());
    REQUIRE(no_memory.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
}

TEST_CASE("Binary BruteForce adds and scans with several threads",
          "[ft][binary][brute_force][concurrent]") {
    const int64_t dim = 200;
    const int64_t count = 3000;
    const int64_t k = 20;
    auto data = make_data(dim, count, 31);
    // thread_count gives the index a thread pool, which Build and Add use for the inserts.
    auto index = create_index("brute_force", create_parameters(dim, {{"thread_count", 4}}));
    REQUIRE(index->Build(base_dataset(data, 0, 1000)).has_value());
    REQUIRE(index->Add(base_dataset(data, 1000, count)).has_value());
    REQUIRE(index->GetNumElements() == count);

    std::mt19937_64 rng(5);
    std::vector<uint8_t> query(row_bytes(dim));
    for (int q = 0; q < 5; ++q) {
        for (auto& value : query) {
            value = static_cast<uint8_t>(rng());
        }
        const auto exact = exact_distances(data, query.data(), {});
        // A parallel scan splits the ids into chunks with one heap each, then merges them.
        for (const auto* search : {"{}", R"({"brute_force": {"parallelism": 4}})"}) {
            INFO("search=" << search);
            auto knn = index->KnnSearch(query_dataset(query.data(), dim), k, search);
            REQUIRE(knn.has_value());
            check_exact_knn(knn.value(), exact, k);
        }
        const auto radius = kth_distance(exact, k);
        auto range =
            index->RangeSearch(query_dataset(query.data(), dim), radius, R"({"parallelism": 3})");
        REQUIRE(range.has_value());
        check_exact_range(range.value(), exact, radius);
    }
}

TEST_CASE("Binary indexes update vectors", "[ft][binary]") {
    const std::string name = GENERATE("brute_force", "hgraph");
    INFO("index=" << name);
    const int64_t dim = 100;  // the last byte has four padding bits
    const int64_t count = 600;
    auto data = make_data(dim, count, 2026);
    auto index = create_index(name, create_parameters(dim));
    REQUIRE(index->Build(base_dataset(data)).has_value());
    const std::string search = name == "hgraph" ? R"({"hgraph": {"ef_search": 200}})" : "{}";

    const int64_t row = 17;
    const int64_t target = data.ids[row];
    // Writing a vector's own bits back is accepted without force and keeps the vector.
    auto same = index->UpdateVector(target, base_dataset(data, row, row + 1));
    REQUIRE(same.has_value());
    REQUIRE(same.value());

    // A one-bit change keeps the vector closer to its old position than to any neighbor, so the
    // update is accepted without force (HGraph checks this with native binary distances).
    std::vector<uint8_t> nudged(data.row(row), data.row(row) + row_bytes(dim));
    nudged[0] ^= 0x80;
    auto nudge = index->UpdateVector(target, query_dataset(nudged.data(), dim));
    REQUIRE(nudge.has_value());
    REQUIRE(nudge.value());
    REQUIRE(reference_hamming(index->GetRawVectorByIds(&target, 1).value()->GetBinaryVectors(),
                              nudged.data(),
                              dim) == 0.0F);

    // A new vector with garbage in the padding bits of its last byte.
    std::vector<uint8_t> replacement(row_bytes(dim));
    std::mt19937_64 rng(7);
    for (auto& value : replacement) {
        value = static_cast<uint8_t>(rng());
    }
    replacement.back() |= 0x0F;
    auto updated = index->UpdateVector(target, query_dataset(replacement.data(), dim), true);
    REQUIRE(updated.has_value());
    REQUIRE(updated.value());

    // The stored row holds the new bits with the padding cleared.
    auto raw = index->GetRawVectorByIds(&target, 1);
    REQUIRE(raw.has_value());
    const auto* stored = raw.value()->GetBinaryVectors();
    REQUIRE(reference_hamming(stored, replacement.data(), dim) == 0.0F);
    REQUIRE((stored[row_bytes(dim) - 1] & 0x0F) == 0);

    // Searches and distances see the new vector.
    auto knn = index->KnnSearch(query_dataset(replacement.data(), dim), 1, search);
    REQUIRE(knn.has_value());
    REQUIRE(knn.value()->GetIds()[0] == target);
    REQUIRE(knn.value()->GetDistances()[0] == 0.0F);
    auto distance = index->CalcDistanceById(query_dataset(data.row(row), dim), target);
    REQUIRE(distance.has_value());
    REQUIRE(distance.value() == reference_hamming(data.row(row), replacement.data(), dim));

    // Updates must carry binary vectors of the index dimension.
    std::vector<float> floats(static_cast<uint64_t>(dim), 0.0F);
    auto float_update = vsag::Dataset::Make();
    float_update->NumElements(1)->Dim(dim)->Float32Vectors(floats.data())->Owner(false);
    REQUIRE_FALSE(index->UpdateVector(target, float_update, true).has_value());
    REQUIRE_FALSE(
        index->UpdateVector(target, query_dataset(replacement.data(), dim + 1), true).has_value());
    REQUIRE(reference_hamming(index->GetRawVectorByIds(&target, 1).value()->GetBinaryVectors(),
                              replacement.data(),
                              dim) == 0.0F);
}

TEST_CASE("Binary BruteForce add, remove and serialization", "[ft][binary][brute_force]") {
    const int64_t dim = GENERATE(65, 256);
    const int64_t count = 800;
    const int64_t k = 15;
    auto data = make_data(dim, count, 77 + static_cast<uint64_t>(dim));
    const auto parameters = create_parameters(dim);
    auto index = create_index("brute_force", parameters);
    REQUIRE(index->Build(base_dataset(data, 0, count / 2)).has_value());
    REQUIRE(index->Add(base_dataset(data, count / 2, count)).has_value());
    REQUIRE(index->GetNumElements() == count);

    // Force removal moves the last stored row into the freed slot, so it also checks that
    // packed rows are copied with the binary code size.
    std::set<int64_t> removed;
    for (int64_t i = 0; i < count; i += 9) {
        REQUIRE(index->Remove(data.ids[i], vsag::RemoveMode::FORCE_REMOVE).has_value());
        removed.insert(data.ids[i]);
    }
    REQUIRE(index->GetNumElements() == count - static_cast<int64_t>(removed.size()));
    auto restored = serialize_round_trip(index, "brute_force", parameters);
    REQUIRE(restored->GetNumElements() == index->GetNumElements());

    // Marked removals hide ids from both indexes without moving data.
    for (int64_t i = 4; i < count; i += 13) {
        if (removed.insert(data.ids[i]).second) {
            REQUIRE(index->Remove(data.ids[i]).has_value());
            REQUIRE(restored->Remove(data.ids[i]).has_value());
        }
    }
    REQUIRE(restored->GetNumElements() == count - static_cast<int64_t>(removed.size()));

    std::mt19937_64 rng(dim);
    std::vector<uint8_t> query(row_bytes(dim));
    for (int q = 0; q < 6; ++q) {
        for (auto& value : query) {
            value = static_cast<uint8_t>(rng());
        }
        const auto exact = exact_distances(data, query.data(), removed);
        for (const auto& candidate : {index, restored}) {
            auto knn = candidate->KnnSearch(query_dataset(query.data(), dim), k, "{}");
            REQUIRE(knn.has_value());
            check_exact_knn(knn.value(), exact, k);
            const auto radius = kth_distance(exact, k);
            auto range = candidate->RangeSearch(query_dataset(query.data(), dim), radius, "{}");
            REQUIRE(range.has_value());
            check_exact_range(range.value(), exact, radius);
        }
    }
}

TEST_CASE("Binary HGraph build, add, remove, search and serialization", "[ft][binary][hgraph]") {
    const int64_t dim = GENERATE(65, 256);
    const int64_t count = 4000;
    const int64_t k = 10;
    auto data = make_data(dim, count, 2872 + static_cast<uint64_t>(dim), 64, 0.12);
    nlohmann::json index_param;
    index_param["max_degree"] = 32;
    index_param["ef_construction"] = 200;
    index_param["support_force_remove"] = true;
    index_param["use_reverse_edges"] = true;  // force removal repairs in-edges through them
    index_param["build_thread_count"] = 4;
    const auto parameters = create_parameters(dim, index_param);
    auto index = create_index("hgraph", parameters);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_METRIC_TYPE_HAMMING));
    REQUIRE(index->Build(base_dataset(data, 0, count / 2)).has_value());
    REQUIRE(index->Add(base_dataset(data, count / 2, count)).has_value());
    REQUIRE(index->GetNumElements() == count);

    std::set<int64_t> removed;
    for (int64_t i = 5; i < count; i += 23) {
        REQUIRE(index->Remove(data.ids[i], vsag::RemoveMode::FORCE_REMOVE).has_value());
        removed.insert(data.ids[i]);
    }

    auto restored = serialize_round_trip(index, "hgraph", parameters);
    REQUIRE(restored->GetNumElements() == count - static_cast<int64_t>(removed.size()));
    for (int64_t i = 11; i < count; i += 31) {
        if (removed.insert(data.ids[i]).second) {
            REQUIRE(index->Remove(data.ids[i]).has_value());
            REQUIRE(restored->Remove(data.ids[i]).has_value());
        }
    }
    const auto search_parameters = R"({"hgraph": {"ef_search": 200}})";
    std::mt19937_64 rng(dim);
    std::vector<uint8_t> query(row_bytes(dim));
    for (const auto& candidate : {index, restored}) {
        double hits = 0;
        double total = 0;
        for (int q = 0; q < 50; ++q) {
            // Queries perturb stored vectors so that close neighbors exist.
            const auto* row = data.row(static_cast<int64_t>(rng() % count));
            for (uint64_t b = 0; b < row_bytes(dim); ++b) {
                query[b] =
                    static_cast<uint8_t>(row[b] ^ ((rng() % 7 == 0) ? (1U << (rng() % 8)) : 0U));
            }
            const auto exact = exact_distances(data, query.data(), removed);
            const float kth = kth_distance(exact, k);
            auto knn = candidate->KnnSearch(query_dataset(query.data(), dim), k, search_parameters);
            REQUIRE(knn.has_value());
            const auto& result = knn.value();
            REQUIRE(result->GetDim() == k);
            for (int64_t r = 0; r < result->GetDim(); ++r) {
                const auto id = result->GetIds()[r];
                REQUIRE(removed.count(id) == 0);
                // Graph search is approximate, but every reported distance is exact.
                REQUIRE(result->GetDistances()[r] == exact.at(id));
                // Ties are common with integer distances: any id within the exact k-th
                // distance counts as a true neighbor.
                hits += result->GetDistances()[r] <= kth ? 1 : 0;
            }
            total += static_cast<double>(k);

            auto exclude_odd = [](int64_t id) { return id % 2 != 0; };
            auto filtered = candidate->KnnSearch(
                query_dataset(query.data(), dim), k, search_parameters, exclude_odd);
            REQUIRE(filtered.has_value());
            for (int64_t r = 0; r < filtered.value()->GetDim(); ++r) {
                const auto id = filtered.value()->GetIds()[r];
                REQUIRE(id % 2 == 0);
                REQUIRE(filtered.value()->GetDistances()[r] == exact.at(id));
            }

            auto range =
                candidate->RangeSearch(query_dataset(query.data(), dim), kth, search_parameters);
            REQUIRE(range.has_value());
            for (int64_t r = 0; r < range.value()->GetDim(); ++r) {
                const auto id = range.value()->GetIds()[r];
                REQUIRE(removed.count(id) == 0);
                REQUIRE(range.value()->GetDistances()[r] == exact.at(id));
                REQUIRE(range.value()->GetDistances()[r] <= kth);
            }
        }
        const double recall = hits / total;
        INFO("recall@" << k << " = " << recall);
        REQUIRE(recall >= 0.95);
    }

    const auto& probe_id = data.ids[1];
    auto distance = restored->CalcDistanceById(query_dataset(data.row(1), dim), probe_id);
    REQUIRE(distance.has_value());
    REQUIRE(distance.value() == 0.0F);
    std::vector<int64_t> wanted = {data.ids[2], data.ids[count - 1]};
    auto raw = restored->GetRawVectorByIds(wanted.data(), 2);
    REQUIRE(raw.has_value());
    REQUIRE(reference_hamming(raw.value()->GetBinaryVectors(), data.row(2), dim) == 0.0F);
}

TEST_CASE("Binary vectors reject invalid configurations and inputs", "[ft][binary]") {
    const int64_t dim = 64;
    auto data = make_data(dim, 50, 9);

    SECTION("dtype and metric must match") {
        auto hamming_float = vsag::Factory::CreateIndex(
            "brute_force", R"({"dtype": "float32", "metric_type": "hamming", "dim": 64})");
        REQUIRE_FALSE(hamming_float.has_value());
        REQUIRE(hamming_float.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        auto binary_l2 = vsag::Factory::CreateIndex(
            "hgraph", R"({"dtype": "binary", "metric_type": "l2", "dim": 64})");
        REQUIRE_FALSE(binary_l2.has_value());
        REQUIRE(binary_l2.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("only brute_force and hgraph accept binary vectors") {
        for (const auto* name :
             {"ivf", "pyramid", "sindi", "sindi_v2", "warp", "simq", "lazy_hgraph"}) {
            auto index = vsag::Factory::CreateIndex(name, create_parameters(dim));
            REQUIRE_FALSE(index.has_value());
            REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
            REQUIRE_THAT(index.error().message,
                         Catch::Matchers::ContainsSubstring("supported only by brute_force"));
        }
    }

    SECTION("numeric quantizers are rejected") {
        auto hgraph = vsag::Factory::CreateIndex(
            "hgraph", create_parameters(dim, {{"base_quantization_type", "sq8"}}));
        REQUIRE_FALSE(hgraph.has_value());
        REQUIRE_THAT(hgraph.error().message,
                     Catch::Matchers::ContainsSubstring("must be binary when dtype is binary"));
        auto reorder = vsag::Factory::CreateIndex(
            "hgraph", create_parameters(dim, {{"precise_quantization_type", "fp32"}}));
        REQUIRE_FALSE(reorder.has_value());
        auto brute_force = vsag::Factory::CreateIndex(
            "brute_force", create_parameters(dim, {{"base_quantization_type", "fp32"}}));
        REQUIRE_FALSE(brute_force.has_value());
        auto explicit_binary = vsag::Factory::CreateIndex(
            "hgraph", create_parameters(dim, {{"base_quantization_type", "binary"}}));
        REQUIRE(explicit_binary.has_value());
    }

    SECTION("datasets must carry binary vectors of the index dimension") {
        for (const auto* name : {"brute_force", "hgraph"}) {
            auto index = create_index(name, create_parameters(dim));
            std::vector<float> floats(static_cast<uint64_t>(dim) * 4, 0.0F);
            std::vector<int64_t> ids = {1, 2, 3, 4};
            auto float_base = vsag::Dataset::Make();
            float_base->NumElements(4)
                ->Dim(dim)
                ->Ids(ids.data())
                ->Float32Vectors(floats.data())
                ->Owner(false);
            REQUIRE_FALSE(index->Build(float_base).has_value());

            REQUIRE(index->Build(base_dataset(data)).has_value());
            std::vector<uint8_t> query(row_bytes(dim), 0);
            auto wrong_dim = vsag::Dataset::Make();
            wrong_dim->NumElements(1)->Dim(dim + 1)->BinaryVectors(query.data())->Owner(false);
            REQUIRE_FALSE(
                index->KnnSearch(wrong_dim, 3, R"({"hgraph": {"ef_search": 20}})").has_value());

            // A float query of the right dimension is an error, not a read of missing bits.
            auto float_query = vsag::Dataset::Make();
            float_query->NumElements(1)->Dim(dim)->Float32Vectors(floats.data())->Owner(false);
            const auto* search = R"({"hgraph": {"ef_search": 20}})";
            REQUIRE_FALSE(index->KnnSearch(float_query, 3, search).has_value());
            REQUIRE_FALSE(index->RangeSearch(float_query, 5.0F, search).has_value());
        }
    }
}

TEST_CASE("Binary HGraph statistics and search analysis", "[ft][binary][hgraph]") {
    const bool reorder = GENERATE(false, true);
    INFO("use_reorder=" << reorder);
    const int64_t dim = 100;
    const int64_t count = 1500;
    auto data = make_data(dim, count, 404);
    nlohmann::json index_param = {{"max_degree", 16}, {"ef_construction", 100}};
    if (reorder) {
        index_param["use_reorder"] = true;  // precise codes are binary as well
    }
    auto index = create_index("hgraph", create_parameters(dim, index_param));
    REQUIRE(index->Build(base_dataset(data)).has_value());

    // Base-sample statistics search with the stored packed rows as queries. The ten samples are
    // random and Hamming distances often tie at the 100th neighbor, so recall is only bounded.
    std::string stats;
    REQUIRE_NOTHROW(stats = index->GetStats());
    const auto parsed = nlohmann::json::parse(stats);
    REQUIRE(parsed["total_count"].get<int64_t>() == count);
    REQUIRE(parsed["recall_base"].get<double>() >= 0.8);
    // Base and precise codes hold the same bits, so their distances agree exactly.
    REQUIRE(parsed["quantization_bias_ratio"].get<double>() == 0.0);
    REQUIRE(parsed["quantization_inversion_count_rate"].get<double>() == 0.0);

    // Search analysis with binary queries in a Dataset. The queries are base rows whose 10th and
    // 11th nearest distances differ, so each exact top 10 is unique and recall does not depend on
    // how ties are broken. The mean ground-truth distance is checked exactly.
    const int64_t topk = 10;
    const uint64_t queries = 8;
    std::vector<uint8_t> query_rows;
    double expected_avg_distance = 0.0;
    std::vector<float> distances(static_cast<uint64_t>(count));
    for (int64_t row = 0; row < count and query_rows.size() < queries * row_bytes(dim); ++row) {
        for (int64_t other = 0; other < count; ++other) {
            distances[static_cast<uint64_t>(other)] =
                reference_hamming(data.row(row), data.row(other), dim);
        }
        std::partial_sort(distances.begin(), distances.begin() + topk + 1, distances.end());
        if (distances[topk - 1] == distances[topk]) {
            continue;
        }
        query_rows.insert(query_rows.end(), data.row(row), data.row(row) + row_bytes(dim));
        for (int64_t i = 0; i < topk; ++i) {
            expected_avg_distance += distances[static_cast<uint64_t>(i)];
        }
    }
    REQUIRE(query_rows.size() == queries * row_bytes(dim));
    expected_avg_distance /= static_cast<double>(queries) * static_cast<double>(topk);

    vsag::SearchRequest request;
    request.query_ = vsag::Dataset::Make();
    request.query_->NumElements(static_cast<int64_t>(queries))
        ->Dim(dim)
        ->BinaryVectors(query_rows.data())
        ->Owner(false);
    request.topk_ = topk;
    request.params_str_ = R"({"hgraph": {"ef_search": 100}})";
    std::string analysis;
    REQUIRE_NOTHROW(analysis = index->AnalyzeIndexBySearch(request));
    const auto parsed_analysis = nlohmann::json::parse(analysis);
    const auto avg_distance = parsed_analysis["avg_distance_query"].get<double>();
    REQUIRE(std::abs(avg_distance - expected_avg_distance) < 1e-4);
    REQUIRE(parsed_analysis["recall_query"].get<double>() >= 0.9);
    REQUIRE(parsed_analysis["quantization_bias_ratio_query"].get<double>() == 0.0);
}
