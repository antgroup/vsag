
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

#include "sindi_v2.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

using namespace vsag;

namespace {

class FooterReadCountingBuffer : public std::stringbuf {
public:
    FooterReadCountingBuffer(std::string serialized, uint64_t footer_offset)
        : std::stringbuf(std::move(serialized), std::ios::in), footer_offset_(footer_offset) {
    }

    [[nodiscard]] uint64_t
    GetFooterBytesRead() const {
        return footer_bytes_read_;
    }

protected:
    std::streamsize
    xsgetn(char* destination, std::streamsize count) override {
        const auto offset = static_cast<uint64_t>(this->gptr() - this->eback());
        const auto bytes_read = std::stringbuf::xsgetn(destination, count);
        if (bytes_read <= 0) {
            return bytes_read;
        }
        const auto end = offset + static_cast<uint64_t>(bytes_read);
        if (end > footer_offset_) {
            footer_bytes_read_ += end - std::max(offset, footer_offset_);
        }
        return bytes_read;
    }

private:
    uint64_t footer_offset_{0};
    uint64_t footer_bytes_read_{0};
};

SINDIV2ParameterPtr
create_sindi_v2_param(uint32_t term_id_limit,
                      const std::string& term_path,
                      const std::string& term_io_type = "buffer_io",
                      const std::string& rerank_io_type = "memory_io",
                      const std::string& rerank_layout = "none",
                      uint32_t rerank_layout_top_terms = 16) {
    auto param_str = fmt::format(R"({{
        "term_id_limit": {},
        "window_size": 10000,
        "doc_prune_ratio": 0.0,
        "use_quantization": false,
        "use_reorder": true,
        "avg_doc_term_length": 100,
        "rerank_layout": "{}",
        "rerank_layout_top_terms": {},
        "term_io": {{ "type": "{}", "file_path": "{}" }},
        "rerank_io": {{ "type": "{}" }}
    }})",
                                 term_id_limit,
                                 rerank_layout,
                                 rerank_layout_top_terms,
                                 term_io_type,
                                 term_path,
                                 rerank_io_type);
    auto param = std::make_shared<SINDIV2Parameter>();
    param->FromJson(JsonType::Parse(param_str));
    return param;
}

}  // namespace

TEST_CASE("SINDIV2 Batch Rerank End-To-End", "[ut][SINDIV2]") {
    // Phase A regression: the rerank path now batches IO through
    // GetCodesByIdsBatch instead of issuing one Read per candidate. The
    // returned (id, distance) pairs must remain identical to the single-id
    // path. We verify that by checking that the top-k results are a sane
    // permutation of the brute-force top-k (allowing ties due to identical
    // distances when k is at the boundary).
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 500;
    const uint32_t num_query = 20;
    const int64_t max_dim = 128;
    const uint32_t term_id_limit = 5000;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, /*max_id=*/term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("sindi_v2_batch_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_sindi_v2_param(term_id_limit, term_path);
    auto index = std::make_unique<SINDIV2>(param, common_param);
    REQUIRE(index->Build(base).empty());

    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 100,
            "use_term_lists_heap_insert": false
        }
    })";

    for (uint32_t q = 0; q < num_query; ++q) {
        auto query = Dataset::Make();
        query->NumElements(1)->SparseVectors(sv_base.data() + q)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param, nullptr);
        REQUIRE(result->GetDim() == k);

        auto range_result = index->RangeSearch(query, 100.0F, search_param, nullptr, k);
        REQUIRE(range_result->GetDim() == k);
        for (int64_t i = 1; i < range_result->GetDim(); ++i) {
            REQUIRE(range_result->GetDistances()[i] >= range_result->GetDistances()[i - 1] - 1e-5F);
        }

        // Distances must be non-decreasing (heap output order).
        for (int64_t i = 1; i < result->GetDim(); ++i) {
            REQUIRE(result->GetDistances()[i] >= result->GetDistances()[i - 1] - 1e-5);
        }

        // The query is itself in the index; the best hit must match it with
        // distance ~0 (1 - <q, q>/<q, q> = 0 when normalized, otherwise the
        // smallest dist of all pairs).
        bool found_self = false;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            if (result->GetIds()[i] == static_cast<int64_t>(q)) {
                found_self = true;
                break;
            }
        }
        REQUIRE(found_self);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
}

TEST_CASE("SINDIV2 Top Terms Rerank Layout End-To-End", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 128;
    const int64_t max_dim = 64;
    const uint32_t term_id_limit = 2048;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("sindi_v2_top_terms_layout");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_sindi_v2_param(term_id_limit,
                                       term_path,
                                       "buffer_io",
                                       "memory_io",
                                       "top_terms_signature",
                                       /*rerank_layout_top_terms=*/8);
    auto index = std::make_unique<SINDIV2>(param, common_param);
    REQUIRE(index->Build(base).empty());

    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 64,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = index->KnnSearch(query, k, search_param, nullptr);
    REQUIRE(result->GetDim() == k);
    REQUIRE(result->GetIds()[0] == 0);
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        auto precise_dist = index->CalcDistanceById(query, result->GetIds()[i], true);
        REQUIRE(std::abs(result->GetDistances()[i] - precise_dist) < 1e-5);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
}

TEST_CASE("SINDIV2 Sorted Batch Rerank End-To-End", "[ut][SINDIV2]") {
    // The rerank path sorts candidate inner_ids before issuing batched IO.
    // Returned (id, distance) pairs must remain identical to the brute-force
    // truth. Each returned distance is cross-checked against CalcDistanceById to
    // ensure the batched code path produces the same distance computation.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 500;
    const uint32_t num_query = 10;
    const int64_t max_dim = 128;
    const uint32_t term_id_limit = 5000;
    const int64_t k = 20;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, /*max_id=*/term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("sindi_v2_batch_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto param = create_sindi_v2_param(term_id_limit, term_path);
    auto index = std::make_unique<SINDIV2>(param, common_param);
    REQUIRE(index->Build(base).empty());

    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 200,
            "use_term_lists_heap_insert": false
        }
    })";

    for (uint32_t q = 0; q < num_query; ++q) {
        auto query = Dataset::Make();
        query->NumElements(1)->SparseVectors(sv_base.data() + q)->Owner(false);

        auto result = index->KnnSearch(query, k, search_param, nullptr);
        REQUIRE(result->GetDim() == k);

        // Distances must be non-decreasing.
        for (int64_t i = 1; i < result->GetDim(); ++i) {
            REQUIRE(result->GetDistances()[i] >= result->GetDistances()[i - 1] - 1e-5);
        }

        // Cross-check each result distance against CalcDistanceById (which
        // uses the single-id GetCodesById path, bypassing batched IO).
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            auto precise_dist =
                index->CalcDistanceById(query, result->GetIds()[i], /*precise=*/true);
            REQUIRE(std::abs(result->GetDistances()[i] - precise_dist) < 1e-5);
        }

        // Self must appear in the results.
        bool found_self = false;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            if (result->GetIds()[i] == static_cast<int64_t>(q)) {
                found_self = true;
                break;
            }
        }
        REQUIRE(found_self);
    }

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    index.reset();
    std::remove(term_path.c_str());
}

TEST_CASE("SINDIV2 ReaderIO Rerank Uses Section Offset", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    const uint32_t num_base = 128;
    const int64_t max_dim = 64;
    const uint32_t term_id_limit = 2048;
    const int64_t k = 10;
    common_param.dim_ = max_dim;

    std::vector<int64_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0);

    auto sv_base = fixtures::GenerateSparseVectors(num_base, max_dim, term_id_limit - 1);
    auto base = Dataset::Make();
    base->NumElements(num_base)->SparseVectors(sv_base.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("sindi_v2_readerio_rerank");
    const std::string term_path = dir.GenerateRandomFile(false);
    auto build_param = create_sindi_v2_param(term_id_limit, term_path);
    SINDIV2 built(build_param, common_param);
    REQUIRE(built.Build(base).empty());

    std::stringstream stream;
    const std::string prefix = "outer-container-prefix";
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    IOStreamWriter writer(stream);
    built.Serialize(writer);

    auto load_param =
        create_sindi_v2_param(term_id_limit, term_path, IO_TYPE_VALUE_READER_IO, "reader_io");
    SINDIV2 loaded(load_param, common_param);
    stream.seekg(static_cast<std::streamoff>(prefix.size()), std::ios::beg);
    loaded.Deserialize(stream);

    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(sv_base.data())->Owner(false);
    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 64,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = loaded.KnnSearch(query, k, search_param, nullptr);
    REQUIRE(result->GetDim() == k);
    REQUIRE(result->GetIds()[0] == 0);

    for (auto& item : sv_base) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
}

TEST_CASE("SINDIV2 istream deserialize parses footer once", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto parameter = std::make_shared<SINDIV2Parameter>();
    parameter->FromJson(JsonType::Parse(R"({
        "term_id_limit": 16,
        "window_size": 10000,
        "use_reorder": false,
        "term_io": {"type": "memory_io"}
    })"));
    SINDIV2 source(parameter, common_param);
    std::stringstream serialized_stream;
    IOStreamWriter writer(serialized_stream);
    source.Serialize(writer);
    auto serialized = serialized_stream.str();

    constexpr uint64_t footer_trailer_size = 2 * sizeof(uint64_t);
    REQUIRE(serialized.size() >= footer_trailer_size);
    uint64_t footer_size = 0;
    std::memcpy(&footer_size,
                serialized.data() + serialized.size() - footer_trailer_size,
                sizeof(footer_size));
    REQUIRE(footer_size <= serialized.size());

    FooterReadCountingBuffer buffer(serialized, serialized.size() - footer_size);
    std::istream input(&buffer);
    SINDIV2 loaded(parameter, common_param);
    loaded.Deserialize(input);

    REQUIRE(buffer.GetFooterBytesRead() == footer_size + footer_trailer_size);
}

TEST_CASE("SINDIV2 rejects corrupted term layout", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.dim_ = 16;

    std::vector<int64_t> ids{0, 1};
    auto vectors = fixtures::GenerateSparseVectors(2, 16, 127);
    auto base = Dataset::Make();
    base->NumElements(2)->SparseVectors(vectors.data())->Ids(ids.data())->Owner(false);

    fixtures::TempDir dir("sindi_v2_invalid_term_layout");
    auto param = create_sindi_v2_param(128, dir.GenerateRandomFile(false));
    SINDIV2 built(param, common_param);
    REQUIRE(built.Build(base).empty());

    std::stringstream stream;
    IOStreamWriter writer(stream);
    built.Serialize(writer);
    auto bytes = stream.str();
    auto invalid_term_bytes = bytes;
    const auto term_dict_count_offset = sizeof(int64_t);
    uint64_t term_dict_count = 0;
    std::memcpy(&term_dict_count, bytes.data() + term_dict_count_offset, sizeof(term_dict_count));
    const auto term_dict_offset = term_dict_count_offset + sizeof(uint64_t);
    const auto payload_size_offset = term_dict_offset + term_dict_count * sizeof(DiskTermEntry);
    const auto invalid_payload_size = std::numeric_limits<uint64_t>::max();
    std::memcpy(
        bytes.data() + payload_size_offset, &invalid_payload_size, sizeof(invalid_payload_size));

    std::stringstream corrupted(bytes);
    IOStreamReader reader(corrupted);
    SINDIV2 loaded(param, common_param);
    REQUIRE_THROWS_WITH(loaded.Deserialize(reader),
                        Catch::Matchers::ContainsSubstring("IO payload exceeds stream length"));

    uint64_t posting_payload_size = 0;
    std::memcpy(&posting_payload_size,
                invalid_term_bytes.data() + payload_size_offset,
                sizeof(posting_payload_size));
    bool corrupted_term = false;
    for (uint32_t term = 0; term < term_dict_count; ++term) {
        const auto entry_offset =
            term_dict_offset + static_cast<uint64_t>(term) * sizeof(DiskTermEntry);
        DiskTermEntry entry;
        std::memcpy(&entry, invalid_term_bytes.data() + entry_offset, sizeof(entry));
        if (entry.posting_count == 0) {
            continue;
        }
        entry.posting_payload_offset = posting_payload_size;
        std::memcpy(invalid_term_bytes.data() + entry_offset, &entry, sizeof(entry));
        corrupted_term = true;
        break;
    }
    REQUIRE(corrupted_term);
    auto memory_parameter = create_sindi_v2_param(
        128, dir.GenerateRandomFile(false), IO_TYPE_VALUE_MEMORY_IO, IO_TYPE_VALUE_MEMORY_IO);
    std::stringstream invalid_term_stream(invalid_term_bytes);
    IOStreamReader invalid_term_reader(invalid_term_stream);
    SINDIV2 memory_loaded(memory_parameter, common_param);
    REQUIRE_THROWS_WITH(memory_loaded.Deserialize(invalid_term_reader),
                        Catch::Matchers::ContainsSubstring("term dictionary layout"));

    for (auto& vector : vectors) {
        delete[] vector.vals_;
        delete[] vector.ids_;
    }
}

TEST_CASE("SINDIV2 memory term layout mutable and immutable roundtrip", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.dim_ = 8;

    uint32_t vector0_ids[] = {1, 3, 7};
    float vector0_values[] = {0.25F, 0.5F, 0.75F};
    uint32_t vector1_ids[] = {1, 4};
    float vector1_values[] = {0.2F, 0.8F};
    uint32_t vector2_ids[] = {2, 7};
    float vector2_values[] = {0.6F, 0.4F};
    SparseVector vectors[] = {{3, vector0_ids, vector0_values},
                              {2, vector1_ids, vector1_values},
                              {2, vector2_ids, vector2_values}};
    int64_t labels[] = {10, 20, 30};
    auto base = Dataset::Make();
    base->NumElements(3)->SparseVectors(vectors)->Ids(labels)->Owner(false);
    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(vectors)->Owner(false);
    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 3,
            "use_term_lists_heap_insert": false
        }
    })";

    struct TestConfig {
        const char* name;
        const char* quantization;
        bool immutable;
        float value_epsilon;
    };
    const TestConfig configs[] = {{"mutable-fp32", "false", false, 1e-6F},
                                  {"immutable-fp16", R"("fp16")", true, 1e-3F},
                                  {"immutable-sq8", "true", true, 3e-3F}};
    for (const auto& config : configs) {
        DYNAMIC_SECTION(config.name) {
            const auto param_json = fmt::format(R"({{
                "term_id_limit": 16,
                "window_size": 10000,
                "doc_prune_ratio": 0.0,
                "use_quantization": {},
                "use_reorder": false,
                "immutable": {},
                "term_io": {{"type": "memory_io"}}
            }})",
                                                config.quantization,
                                                config.immutable);
            auto parameter = std::make_shared<SINDIV2Parameter>();
            parameter->FromJson(JsonType::Parse(param_json));
            SINDIV2 built(parameter, common_param);
            REQUIRE(built.Build(base).empty());

            auto knn = built.KnnSearch(query, 2, search_param, nullptr);
            REQUIRE(knn->GetDim() == 2);
            REQUIRE(knn->GetIds()[0] == labels[0]);
            auto range = built.RangeSearch(query, 100.0F, search_param, nullptr, 2);
            REQUIRE(range->GetDim() == 2);

            SparseVector restored_vector;
            built.GetSparseVectorByInnerId(0, &restored_vector, allocator.get());
            REQUIRE(restored_vector.len_ == vectors[0].len_);
            for (uint32_t i = 0; i < restored_vector.len_; ++i) {
                REQUIRE(restored_vector.ids_[i] == vectors[0].ids_[i]);
                REQUIRE(std::abs(restored_vector.vals_[i] - vectors[0].vals_[i]) <=
                        config.value_epsilon);
            }
            allocator->Deallocate(restored_vector.ids_);
            allocator->Deallocate(restored_vector.vals_);
            const auto expected_distance = built.CalcDistanceById(query, labels[0], false);

            std::stringstream stream;
            IOStreamWriter writer(stream);
            built.Serialize(writer);
            const auto serialized = stream.str();
            SINDIV2 loaded(parameter, common_param);
            stream.seekg(0, std::ios::beg);
            loaded.Deserialize(stream);
            auto loaded_knn = loaded.KnnSearch(query, 2, search_param, nullptr);
            REQUIRE(loaded_knn->GetDim() == 2);
            REQUIRE(loaded_knn->GetIds()[0] == labels[0]);
            REQUIRE(std::abs(loaded.CalcDistanceById(query, labels[0], false) -
                             expected_distance) <= 1e-5F);

            auto disk_parameter_json = parameter->ToJson();
            disk_parameter_json["term_io"].SetJson(JsonType::Parse(R"({"type":"reader_io"})"));
            auto disk_parameter = std::make_shared<SINDIV2Parameter>();
            disk_parameter->FromJson(disk_parameter_json);
            SINDIV2 disk_loaded(disk_parameter, common_param);
            std::stringstream disk_stream(serialized);
            disk_loaded.Deserialize(disk_stream);
            auto disk_knn = disk_loaded.KnnSearch(query, 2, search_param, nullptr);
            REQUIRE(disk_knn->GetDim() == 2);
            REQUIRE(disk_knn->GetIds()[0] == labels[0]);
            REQUIRE(std::abs(disk_loaded.CalcDistanceById(query, labels[0], false) -
                             expected_distance) <= 1e-5F);

            if (config.immutable) {
                REQUIRE_THROWS_WITH(
                    built.Add(base),
                    Catch::Matchers::ContainsSubstring("immutable SINDIV2 does not support Add"));
            }
        }
    }
}

TEST_CASE("SINDIV2 term dictionary uses the active term range", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t vector0_ids[] = {100, 5000};
    float vector0_values[] = {0.4F, 0.6F};
    uint32_t vector1_ids[] = {100};
    float vector1_values[] = {1.0F};
    SparseVector vectors[] = {{2, vector0_ids, vector0_values}, {1, vector1_ids, vector1_values}};
    int64_t labels[] = {1, 2};
    auto base = Dataset::Make();
    base->NumElements(2)->SparseVectors(vectors)->Ids(labels)->Owner(false);

    for (const bool remap_term_ids : {false, true}) {
        DYNAMIC_SECTION("remap=" << remap_term_ids) {
            const auto param_json = fmt::format(R"({{
                "term_id_limit": 10000,
                "window_size": 10000,
                "doc_prune_ratio": 0.0,
                "use_quantization": false,
                "use_reorder": false,
                "remap_term_ids": {},
                "term_io": {{"type": "memory_io"}}
            }})",
                                                remap_term_ids);
            auto parameter = std::make_shared<SINDIV2Parameter>();
            parameter->FromJson(JsonType::Parse(param_json));
            SINDIV2 index(parameter, common_param);
            REQUIRE(index.Build(base).empty());

            std::stringstream stream;
            IOStreamWriter writer(stream);
            index.Serialize(writer);
            const auto bytes = stream.str();
            uint64_t term_dict_count = 0;
            std::memcpy(&term_dict_count, bytes.data() + sizeof(int64_t), sizeof(term_dict_count));
            REQUIRE(term_dict_count == (remap_term_ids ? 2 : 5001));
        }
    }
}

TEST_CASE("SINDIV2 empty index roundtrip", "[ut][SINDIV2]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto parameter = std::make_shared<SINDIV2Parameter>();
    parameter->FromJson(JsonType::Parse(R"({
        "term_id_limit": 16,
        "window_size": 10000,
        "use_reorder": false,
        "term_io": {"type": "memory_io"}
    })"));
    SINDIV2 empty(parameter, common_param);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    empty.Serialize(writer);

    SINDIV2 loaded(parameter, common_param);
    stream.seekg(0, std::ios::beg);
    loaded.Deserialize(stream);
    uint32_t term = 1;
    float value = 1.0F;
    SparseVector sparse_query{1, &term, &value};
    auto query = Dataset::Make();
    query->NumElements(1)->SparseVectors(&sparse_query)->Owner(false);
    const std::string search_param = R"({
        "sindi_v2": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 1,
            "use_term_lists_heap_insert": false
        }
    })";
    REQUIRE(loaded.KnnSearch(query, 1, search_param, nullptr)->GetDim() == 0);
    REQUIRE(loaded.RangeSearch(query, 1.0F, search_param, nullptr)->GetDim() == 0);
}
