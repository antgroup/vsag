
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

#include "uhg.h"

#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstring>
#include <future>
#include <random>
#include <sstream>

#include "framework/test_reader.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace vsag {
namespace {
struct HybridData {
    static constexpr uint64_t count = 40;
    static constexpr uint64_t dim = 8;
    std::vector<int64_t> ids;
    std::vector<float> dense;
    std::vector<uint32_t> terms;
    std::vector<float> values;
    std::vector<SparseVector> sparse;
    HybridData()
        : ids(count), dense(count * dim), terms(count * 3), values(count * 3), sparse(count) {
        std::mt19937 random(42);
        std::uniform_real_distribution<float> distribution(0.01F, 1.0F);
        for (uint64_t i = 0; i < count; ++i) {
            ids[i] = 100 + 7 * i;
            float norm = 0;
            for (uint64_t d = 0; d < dim; ++d) {
                dense[i * dim + d] = distribution(random);
                norm += dense[i * dim + d] * dense[i * dim + d];
            }
            for (uint64_t d = 0; d < dim; ++d) {
                dense[i * dim + d] /= std::sqrt(norm);
            }
            for (uint64_t t = 0; t < 3; ++t) {
                terms[i * 3 + t] = t * 10 + i % 5;
                values[i * 3 + t] = distribution(random);
            }
            sparse[i] = {3, terms.data() + i * 3, values.data() + i * 3};
        }
    }
    DatasetPtr
    Base() {
        return Dataset::Make()
            ->Owner(false)
            ->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float32Vectors(dense.data())
            ->SparseVectors(sparse.data());
    }
    DatasetPtr
    Query(uint64_t i = 3) {
        return Dataset::Make()
            ->Owner(false)
            ->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(dense.data() + i * dim)
            ->SparseVectors(sparse.data() + i);
    }
    std::vector<std::pair<double, int64_t>>
    Truth(const DatasetPtr& query, double alpha) {
        std::vector<std::pair<double, int64_t>> result;
        for (uint64_t i = 0; i < count; ++i) {
            double dd = 1, ds = 1;
            for (uint64_t d = 0; d < dim; ++d) {
                dd -= query->GetFloat32Vectors()[d] * dense[i * dim + d];
            }
            const auto& q = *query->GetSparseVectors();
            for (uint32_t a = 0; a < q.len_; ++a) {
                for (uint32_t b = 0; b < sparse[i].len_; ++b) {
                    if (q.ids_[a] == sparse[i].ids_[b]) {
                        ds -= q.vals_[a] * sparse[i].vals_[b];
                    }
                }
            }
            result.emplace_back(alpha * dd + (1 - alpha) * ds, ids[i]);
        }
        std::sort(result.begin(), result.end());
        return result;
    }
};

std::string
build_parameters(const std::string& index_param =
                     R"({"max_degree":40,"neighbors_per_alpha":40,"construction_candidates":40,
        "sindi":{"term_id_limit":100}})") {
    return R"({"dtype":"float32","metric_type":"ip","dim":8,"index_param":)" + index_param + "}";
}

IndexPtr
make_index(const std::string& parameters = build_parameters()) {
    auto result = Factory::CreateIndex("uhg", parameters);
    REQUIRE(result.has_value());
    return result.value();
}

std::string
search_parameters(float alpha, uint64_t ef = 40, bool pruning = true) {
    JsonType json;
    json["uhg"]["alpha"].SetFloat(alpha);
    json["uhg"]["ef_search"].SetUint64(ef);
    json["uhg"]["entry_points"].SetInt(1);
    json["uhg"]["use_pruning"].SetBool(pruning);
    return json.Dump();
}
}  // namespace

TEST_CASE("UHG dynamic weights match exact hybrid ground truth", "[uhg]") {
    HybridData data;
    auto index = make_index();
    REQUIRE(index->GetIndexType() == IndexType::UHG);
    REQUIRE(index->Build(data.Base()).value().empty());
    REQUIRE(index->GetNumElements() == HybridData::count);
    REQUIRE(index->GetMemoryUsage() > 0);
    REQUIRE(index->CheckIdExist(data.ids[0]));
    REQUIRE_FALSE(index->CheckIdExist(99999));
    REQUIRE(index->ExportIDs().value()->GetNumElements() == HybridData::count);
    for (float alpha : {0.0F, 0.17F, 0.5F, 0.73F, 1.0F}) {
        for (bool pruning : {false, true}) {
            auto result =
                index->KnnSearch(data.Query(), 5, search_parameters(alpha, 40, pruning)).value();
            auto truth = data.Truth(data.Query(), alpha);
            REQUIRE(result->GetDim() == 5);
            for (uint64_t j = 0; j < 5; ++j) {
                REQUIRE(result->GetIds()[j] == truth[j].second);
                REQUIRE(result->GetDistances()[j] == Catch::Approx(truth[j].first).margin(1e-5));
            }
        }
    }
    // Full seeds fill a smaller beam and exercise safe pruning/beam termination.
    auto params = R"({"uhg":{"alpha":0.7,"ef_search":8,"entry_points":8}})";
    auto conservative = index->KnnSearch(data.Query(), 5, params).value();
    auto unpruned =
        index
            ->KnnSearch(
                data.Query(),
                5,
                R"({"uhg":{"alpha":0.7,"ef_search":8,"entry_points":8,"use_pruning":false}})")
            .value();
    for (uint64_t j = 0; j < 5; ++j) {
        REQUIRE(conservative->GetIds()[j] == unpruned->GetIds()[j]);
    }
    auto future = std::async(std::launch::async, [&]() {
        return index->KnnSearch(data.Query(), 5, search_parameters(0.3F));
    });
    REQUIRE(index->KnnSearch(data.Query(), 5, search_parameters(0.8F)).has_value());
    REQUIRE(future.get().has_value());
}

TEST_CASE("UHG recovers a hybrid winner outside both guide top ones", "[uhg]") {
    int64_t ids[]{10, 20, 30};
    float dense[]{1, 0, 0.1F, 0, 0.7F, 0};
    uint32_t term = 3;
    float values[]{0.1F, 1, 0.7F};
    SparseVector sparse[]{{1, &term, values}, {1, &term, values + 1}, {1, &term, values + 2}};
    float qdense[]{1, 0};
    float qvalue = 1;
    SparseVector qsparse{1, &term, &qvalue};
    auto base = Dataset::Make()
                    ->Owner(false)
                    ->NumElements(3)
                    ->Dim(2)
                    ->Ids(ids)
                    ->Float32Vectors(dense)
                    ->SparseVectors(sparse);
    auto query = Dataset::Make()
                     ->Owner(false)
                     ->NumElements(1)
                     ->Dim(2)
                     ->Float32Vectors(qdense)
                     ->SparseVectors(&qsparse);
    auto index = make_index(R"({"dtype":"float32","dim":2,"metric_type":"ip",
        "index_param":{"neighbors_per_alpha":3,"construction_candidates":3}})");
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->KnnSearch(query, 1, search_parameters(0)).value()->GetIds()[0] == 20);
    REQUIRE(index->KnnSearch(query, 1, search_parameters(1)).value()->GetIds()[0] == 10);
    auto hybrid = index->KnnSearch(query, 1, search_parameters(0.5F)).value();
    REQUIRE(hybrid->GetIds()[0] == 30);
    REQUIRE(hybrid->GetDistances()[0] == Catch::Approx(0.3F));
}

TEST_CASE("UHG serialization restores dynamic queries", "[uhg]") {
    HybridData data;
    auto original = make_index();
    REQUIRE(original->Build(data.Base()).has_value());
    SECTION("binary set") {
        auto binary = original->Serialize().value();
        auto loaded = make_index();
        REQUIRE(loaded->Deserialize(binary).has_value());
        for (float alpha : {0.0F, 0.3F, 0.7F, 1.0F}) {
            auto before = original->KnnSearch(data.Query(), 5, search_parameters(alpha)).value();
            auto after = loaded->KnnSearch(data.Query(), 5, search_parameters(alpha)).value();
            for (uint64_t j = 0; j < 5; ++j) {
                REQUIRE(before->GetIds()[j] == after->GetIds()[j]);
                REQUIRE(before->GetDistances()[j] == after->GetDistances()[j]);
            }
        }
        REQUIRE_FALSE(loaded->Deserialize(binary).has_value());
    }
    SECTION("stream") {
        std::stringstream stream;
        REQUIRE(original->Serialize(stream).has_value());
        auto loaded = make_index();
        REQUIRE(loaded->Deserialize(stream).has_value());
        REQUIRE(loaded->GetNumElements() == HybridData::count);
    }
    SECTION("reader set") {
        auto binary = original->Serialize().value();
        ReaderSet readers;
        for (const auto& key : binary.GetKeys()) {
            readers.Set(key, std::make_shared<fixtures::TestReader>(binary.Get(key)));
        }
        auto loaded = make_index();
        REQUIRE(loaded->Deserialize(readers).has_value());
        auto result = loaded->KnnSearch(data.Query(), 5, search_parameters(0.5F)).value();
        auto truth = data.Truth(data.Query(), 0.5);
        for (uint64_t j = 0; j < 5; ++j) {
            REQUIRE(result->GetIds()[j] == truth[j].second);
        }
    }
    SECTION("write callback") {
        std::vector<char> bytes;
        auto written = original->Serialize([&](uint64_t offset, uint64_t length, const void* data) {
            bytes.resize(offset + length);
            std::memcpy(bytes.data() + offset, data, length);
        });
        REQUIRE(written.has_value());
        auto loaded = make_index();
        std::stringstream stream(std::string(bytes.begin(), bytes.end()));
        REQUIRE(loaded->Deserialize(stream).has_value());
        REQUIRE(loaded->GetNumElements() == HybridData::count);
    }
    SECTION("incompatible parameters") {
        auto binary = original->Serialize().value();
        auto loaded = make_index(build_parameters(R"({"max_degree":8})"));
        REQUIRE_FALSE(loaded->Deserialize(binary).has_value());
        REQUIRE(loaded->GetNumElements() == 0);
    }
}

TEST_CASE("UHG validates data and parameters", "[uhg]") {
    HybridData data;
    auto index = make_index();
    SECTION("duplicate IDs do not partially build") {
        data.ids[1] = data.ids[0];
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
        REQUIRE(index->GetNumElements() == 0);
        data.ids[1] = 12345;
        REQUIRE(index->Build(data.Base()).has_value());
    }
    SECTION("missing dense or sparse") {
        auto base = data.Base();
        base->SparseVectors(nullptr);
        REQUIRE_FALSE(index->Build(base).has_value());
        base->SparseVectors(data.sparse.data())->Float32Vectors(nullptr);
        REQUIRE_FALSE(index->Build(base).has_value());
    }
    SECTION("invalid sparse vector") {
        data.terms[1] = data.terms[0];
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
    }
    SECTION("invalid values") {
        data.values[0] = -1;
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
        data.values[0] = 1;
        data.dense[0] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
    }
    SECTION("empty base sparse vector") {
        data.sparse[0].len_ = 0;
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
    }
    SECTION("invalid query parameters and unsupported operations") {
        REQUIRE(index->Build(data.Base()).has_value());
        for (const auto* p : {R"({"alpha":-0.1})",
                              R"({"alpha":1.1})",
                              R"({"ef_search":1})",
                              R"({"entry_points":0})",
                              R"({"pruning_scale":2})",
                              R"({"ef_search":1.5})",
                              R"({"use_pruning":1})",
                              "[]"}) {
            REQUIRE_FALSE(index->KnnSearch(data.Query(), 5, p).has_value());
        }
        REQUIRE_FALSE(index->Build(data.Base()).has_value());
        REQUIRE_FALSE(index->Add(data.Base()).has_value());
        REQUIRE_FALSE(index->RangeSearch(data.Query(), 1, "{}").has_value());
        REQUIRE_FALSE(index->Remove(data.ids[0]).has_value());
        REQUIRE_FALSE(index->CheckFeature(SUPPORT_ADD_AFTER_BUILD));
        REQUIRE_FALSE(index->CheckFeature(SUPPORT_DELETE_BY_ID));
        auto thresholded =
            index
                ->KnnSearch(
                    data.Query(), 5, R"({"uhg":{"alpha":0.5,"ef_search":40},"threshold":-100})")
                .value();
        REQUIRE(thresholded->GetDim() == 0);
    }
    SECTION("empty query sparse vector and no overlapping terms") {
        REQUIRE(index->Build(data.Base()).has_value());
        SparseVector empty;
        auto query = data.Query()->SparseVectors(&empty);
        auto result = index->KnnSearch(query, 5, search_parameters(0.5F)).value();
        auto truth = data.Truth(query, 0.5);
        for (uint64_t j = 0; j < 5; ++j) {
            REQUIRE(result->GetIds()[j] == truth[j].second);
        }
        uint32_t term = 90;
        float value = 1;
        SparseVector unrelated{1, &term, &value};
        query->SparseVectors(&unrelated);
        REQUIRE(index->KnnSearch(query, 5, search_parameters(0.2F)).has_value());
    }
}

TEST_CASE("UHG validates creation parameters", "[uhg]") {
    REQUIRE(Factory::CreateIndex("uhg", R"({"dim":8,"dtype":"float32","metric_type":"ip"})")
                .has_value());
    for (const auto* p : {R"({"alpha_intervals":0})",
                          R"({"alpha_intervals":1001})",
                          R"({"max_degree":-1})",
                          R"({"max_degree":1.5})",
                          R"({"neighbors_per_alpha":100,"construction_candidates":10})",
                          R"({"hgraph":{"base_quantization_type":"sq8"}})",
                          R"({"sindi":{"use_quantization":true}})",
                          R"({"sindi":{"doc_prune_ratio":0.1}})",
                          R"({"sindi":[]})"}) {
        REQUIRE_FALSE(Factory::CreateIndex("uhg", build_parameters(p)).has_value());
    }
    REQUIRE_FALSE(Factory::CreateIndex("uhg", R"({"dim":8,"dtype":"float32","metric_type":"l2"})")
                      .has_value());
}

TEST_CASE("UHG ID filters keep bridge nodes traversable", "[uhg]") {
    HybridData data;
    auto index = make_index();
    REQUIRE(index->Build(data.Base()).has_value());
    auto params = search_parameters(0.5F);
    // The function overload returns true to exclude an ID.
    auto filtered =
        index
            ->KnnSearch(data.Query(), 5, params, std::function<bool(int64_t)>([&](int64_t id) {
                            return id == data.ids[3];
                        }))
            .value();
    auto truth = data.Truth(data.Query(), 0.5);
    truth.erase(
        std::remove_if(
            truth.begin(), truth.end(), [&](const auto& p) { return p.second == data.ids[3]; }),
        truth.end());
    for (uint64_t j = 0; j < 5; ++j) {
        REQUIRE(filtered->GetIds()[j] == truth[j].second);
    }
    auto none =
        index
            ->KnnSearch(
                data.Query(), 5, params, std::function<bool(int64_t)>([](int64_t) { return true; }))
            .value();
    REQUIRE(none->GetDim() == 0);
}

TEST_CASE("UHG alpha-cover selects frequent neighbors within degree budget", "[uhg]") {
    HybridData data;
    auto index = make_index(build_parameters(R"({"max_degree":4,"neighbors_per_alpha":3,
        "construction_candidates":40,"alpha_intervals":2,"sindi":{"term_id_limit":100}})"));
    REQUIRE(index->Build(data.Base()).has_value());
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    IOStreamReader reader(stream);
    uint32_t magic;
    StreamReader::ReadObj(reader, magic);
    REQUIRE(magic == 0x55484702);
    auto parameters = StreamReader::ReadString(reader);
    int64_t dim;
    StreamReader::ReadObj(reader, dim);
    std::vector<int64_t> ids;
    std::vector<double> norms;
    StreamReader::ReadVector(reader, ids);
    StreamReader::ReadVector(reader, norms);
    for (uint64_t i = 0; i < HybridData::count; ++i) {
        std::unordered_map<int64_t, uint64_t> frequency;
        for (double alpha : {0.0, 0.5, 1.0}) {
            auto candidates = data.Truth(data.Query(i), alpha);
            candidates.erase(std::remove_if(candidates.begin(),
                                            candidates.end(),
                                            [&](auto& pair) { return pair.second == data.ids[i]; }),
                             candidates.end());
            for (uint64_t j = 0; j < 3; ++j) {
                ++frequency[candidates[j].second];
            }
        }
        std::vector<std::pair<uint64_t, int64_t>> expected;
        for (auto [id, count] : frequency) {
            expected.emplace_back(count, id);
        }
        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return a.first == b.first ? a.second < b.second : a.first > b.first;
        });
        std::vector<uint32_t> edges;
        StreamReader::ReadVector(reader, edges);
        REQUIRE(edges.size() == std::min(uint64_t{4}, uint64_t(expected.size())));
        for (uint64_t j = 0; j < edges.size(); ++j) {
            REQUIRE(ids[edges[j]] == expected[j].second);
        }
    }
}

TEST_CASE("UHG rejects damaged payloads without publishing partial state", "[uhg]") {
    HybridData data;
    auto original = make_index();
    REQUIRE(original->Build(data.Base()).has_value());
    std::stringstream stream;
    REQUIRE(original->Serialize(stream).has_value());
    auto bytes = stream.str();
    SECTION("invalid version") {
        bytes[0] = 0;
    }
    SECTION("truncated payload") {
        bytes.resize(bytes.size() / 2);
    }
    SECTION("oversized header length") {
        uint64_t length = UINT64_MAX;
        std::memcpy(bytes.data() + sizeof(uint32_t), &length, sizeof(length));
    }
    auto loaded = make_index();
    std::stringstream damaged(bytes);
    REQUIRE_FALSE(loaded->Deserialize(damaged).has_value());
    REQUIRE(loaded->GetNumElements() == 0);
    REQUIRE(loaded->Build(data.Base()).has_value());
}

TEST_CASE("UHG handles a singleton and requests larger than the dataset", "[uhg]") {
    HybridData data;
    auto index = make_index();
    auto base = data.Base()->NumElements(1);
    REQUIRE(index->Build(base).has_value());
    for (float alpha : {0.0F, 0.5F, 1.0F}) {
        auto result = index->KnnSearch(data.Query(), 5, search_parameters(alpha)).value();
        REQUIRE(result->GetDim() == 1);
        REQUIRE(result->GetIds()[0] == data.ids[0]);
    }
    REQUIRE_FALSE(index->UpdateId(data.ids[0], 12345).has_value());
    REQUIRE(index->CheckIdExist(data.ids[0]));
    REQUIRE_FALSE(index->CheckIdExist(12345));
    auto empty = make_index();
    auto binary = empty->Serialize().value();
    auto loaded = make_index();
    REQUIRE(loaded->Deserialize(binary).has_value());
    REQUIRE(loaded->GetNumElements() == 0);
    REQUIRE(loaded->Build(base).has_value());
}
TEST_CASE("UHG keeps legacy graph archives readable and validates forward counts", "[uhg]") {
    HybridData data;
    auto original = make_index();
    REQUIRE(original->Build(data.Base()).has_value());
    std::stringstream stream;
    REQUIRE(original->Serialize(stream).has_value());
    auto bytes = stream.str();
    IOStreamReader reader(stream);
    uint32_t magic;
    StreamReader::ReadObj(reader, magic);
    auto parameters = StreamReader::ReadString(reader);
    int64_t dim;
    StreamReader::ReadObj(reader, dim);
    std::vector<int64_t> ids;
    std::vector<double> norms;
    StreamReader::ReadVector(reader, ids);
    StreamReader::ReadVector(reader, norms);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        std::vector<uint32_t> edges;
        StreamReader::ReadVector(reader, edges);
    }
    for (int i = 0; i < 2; ++i) {
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        reader.Seek(reader.GetCursor() + size);
    }
    auto forward_offset = reader.GetCursor();
    SECTION("legacy archive") {
        bytes.resize(forward_offset);
        magic = 0x55484701;
        std::memcpy(bytes.data(), &magic, sizeof(magic));
        auto legacy = make_index();
        std::stringstream input(bytes);
        REQUIRE(legacy->Deserialize(input).has_value());
        for (float alpha : {0.0F, 0.3F, 0.7F, 1.0F}) {
            auto result = legacy->KnnSearch(data.Query(), 5, search_parameters(alpha)).value();
            auto truth = data.Truth(data.Query(), alpha);
            for (uint64_t j = 0; j < 5; ++j) {
                REQUIRE(result->GetIds()[j] == truth[j].second);
            }
        }
        std::stringstream upgraded;
        REQUIRE(legacy->Serialize(upgraded).has_value());
        auto again = make_index();
        REQUIRE(again->Deserialize(upgraded).has_value());
    }
    SECTION("forward vector count mismatch") {
        uint32_t wrong_count = HybridData::count - 1;
        std::memcpy(
            bytes.data() + forward_offset + sizeof(bool), &wrong_count, sizeof(wrong_count));
        auto loaded = make_index();
        std::stringstream input(bytes);
        REQUIRE_FALSE(loaded->Deserialize(input).has_value());
        REQUIRE(loaded->GetNumElements() == 0);
        REQUIRE(loaded->Build(data.Base()).has_value());
    }
}
}  // namespace vsag
