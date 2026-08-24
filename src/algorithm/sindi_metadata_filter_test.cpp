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

#include "algorithm/sindi_metadata_filter.h"

#include <array>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "algorithm/sindi/sindi.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "storage/serialization_tags.h"
#include "storage/serialization_template_test.h"
#include "storage/streaming_serialization_test_utils.h"
#include "unittest.h"

using namespace vsag;

namespace {

using vsag::test::EraseStreamingBlock;
using vsag::test::SetStreamingBlockVersion;

struct SmallSindiDataset {
    explicit SmallSindiDataset(uint32_t term_id_offset) {
        ids0 = {term_id_offset + 1, term_id_offset + 4, term_id_offset + 9};
        ids1 = {term_id_offset + 1, term_id_offset + 2, term_id_offset + 4};
        ids2 = {term_id_offset + 5, term_id_offset + 9};

        sparse_vectors[0].len_ = ids0.size();
        sparse_vectors[0].ids_ = ids0.data();
        sparse_vectors[0].vals_ = vals0.data();
        sparse_vectors[2].len_ = ids1.size();
        sparse_vectors[2].ids_ = ids1.data();
        sparse_vectors[2].vals_ = vals1.data();
        sparse_vectors[3].len_ = ids2.size();
        sparse_vectors[3].ids_ = ids2.data();
        sparse_vectors[3].vals_ = vals2.data();
    }

    DatasetPtr
    Base() {
        return Dataset::Make()
            ->NumElements(sparse_vectors.size())
            ->SparseVectors(sparse_vectors.data())
            ->Ids(labels.data())
            ->Owner(false);
    }

    DatasetPtr
    Query() {
        return Dataset::Make()->NumElements(1)->SparseVectors(sparse_vectors.data())->Owner(false);
    }

    std::array<int64_t, 4> labels{10, 40, 20, 30};
    std::array<uint32_t, 3> ids0{};
    std::array<float, 3> vals0{0.0F, 0.5F, 1.0F};
    std::array<uint32_t, 3> ids1{};
    std::array<float, 3> vals1{1.0F, 0.25F, 0.5F};
    std::array<uint32_t, 2> ids2{};
    std::array<float, 2> vals2{0.25F, 1.0F};
    std::array<SparseVector, 4> sparse_vectors{};
};

std::shared_ptr<SINDIParameter>
CreateSindiParameter(bool immutable, bool remap_term_ids) {
    auto param_json = JsonType::Parse(R"({
        "use_reorder": false,
        "rerank_type": "fp32",
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 16,
        "avg_doc_term_length": 3,
        "remap_term_ids": false,
        "immutable": false
    })");
    param_json["immutable"].SetBool(immutable);
    param_json["remap_term_ids"].SetBool(remap_term_ids);
    auto index_param = std::make_shared<SINDIParameter>();
    index_param->FromJson(param_json);
    return index_param;
}

constexpr auto kSindiSearchParameters = R"({
    "sindi": {
        "query_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "n_candidate": 3,
        "use_term_lists_heap_insert": false
    }
})";

void
RequireSameResults(const DatasetPtr& expected, const DatasetPtr& actual) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    for (int64_t i = 0; i < expected->GetDim(); ++i) {
        REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
        REQUIRE(std::abs(actual->GetDistances()[i] - expected->GetDistances()[i]) < 1e-6F);
    }
}

class AllowLabelFilter : public Filter {
public:
    explicit AllowLabelFilter(int64_t label) : label_(label) {
    }

    bool
    CheckValid(int64_t label) const override {
        return label == label_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return 0.25F;
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return Distribution::RELATED_TO_VECTOR;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = &label_;
        count = 1;
        ++get_valid_ids_calls_;
    }

    [[nodiscard]] uint64_t
    GetValidIdsCalls() const {
        return get_valid_ids_calls_;
    }

private:
    int64_t label_;
    mutable uint64_t get_valid_ids_calls_{0};
};

}  // namespace

TEST_CASE("SINDI date bucket and host filters route and serialize",
          "[ut][SINDI][metadata_filter][date_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SmallSindiDataset data(0);
    std::array<uint32_t, 4> host_ids = {2, 1, 2, 1};
    std::array<std::string, 4> date_buckets = {"2026", "2026/05", "2026/05/01", "2026/08"};
    auto base = data.Base()
                    ->UInt32Metadata(SINDI_HOST_ID_METADATA_NAME, host_ids.data())
                    ->Paths(SINDI_DATE_PATH_NAME, date_buckets.data());
    const bool immutable = GENERATE(false, true);
    auto parameter = CreateSindiParameter(immutable, false);
    parameter->use_reorder = GENERATE(false, true);
    auto index = std::make_unique<SINDI>(parameter, common_param);
    REQUIRE(index->Build(base) == std::vector<int64_t>{40});

    int64_t added_label = 50;
    uint32_t added_host = 2;
    std::string added_date = "2026/09";
    auto added_vector = data.sparse_vectors[0];
    auto dated_add = Dataset::Make()
                         ->NumElements(1)
                         ->SparseVectors(&added_vector)
                         ->Ids(&added_label)
                         ->UInt32Metadata(SINDI_HOST_ID_METADATA_NAME, &added_host)
                         ->Paths(SINDI_DATE_PATH_NAME, &added_date)
                         ->Owner(false);
    auto undated_add = Dataset::Make()
                           ->NumElements(1)
                           ->SparseVectors(&added_vector)
                           ->Ids(&added_label)
                           ->UInt32Metadata(SINDI_HOST_ID_METADATA_NAME, &added_host)
                           ->Owner(false);
    if (not immutable) {
        REQUIRE_THROWS_WITH(index->Add(dated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI date-aware index does not support incremental Add"));
        REQUIRE_THROWS_WITH(index->Add(undated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI date-aware index does not support incremental Add"));

        auto first_add_parameter = CreateSindiParameter(false, false);
        first_add_parameter->use_reorder = parameter->use_reorder;
        SINDI first_add_index(first_add_parameter, common_param);
        REQUIRE(first_add_index.Add(base) == std::vector<int64_t>{40});
        REQUIRE_THROWS_WITH(first_add_index.Add(undated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI date-aware index does not support incremental Add"));

        SINDI date_unaware_index(first_add_parameter, common_param);
        REQUIRE(date_unaware_index.Build(data.Base()) == std::vector<int64_t>{40});
        REQUIRE_THROWS_WITH(date_unaware_index.Add(dated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI cannot add date metadata after existing documents"));
    }

    REQUIRE(index->KnnSearch(data.Query(), 3, kSindiSearchParameters, nullptr)->GetDim() == 3);

    std::string query_date = "2026/05";
    auto query = data.Query()->Paths(SINDI_DATE_PATH_NAME, &query_date);
    auto month = index->KnnSearch(query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(month->GetDim() == 1);
    REQUIRE(month->GetIds()[0] == 20);

    query_date = "2026/05/01";
    auto day = index->KnnSearch(query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(day->GetDim() == 1);
    REQUIRE(day->GetIds()[0] == 20);

    query_date = "2026/08/01";
    REQUIRE(index->KnnSearch(query, 3, kSindiSearchParameters, nullptr)->GetDim() == 0);
    query_date = "2026/08";
    auto coarser_base = index->KnnSearch(query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(coarser_base->GetDim() == 1);
    REQUIRE(coarser_base->GetIds()[0] == 30);

    query_date = "2027";
    REQUIRE(index->KnnSearch(query, 3, kSindiSearchParameters, nullptr)->GetDim() == 0);
    REQUIRE(index->RangeSearch(query, 2.0F, kSindiSearchParameters, nullptr, -1)->GetDim() == 3);
    query_date = "2026";
    REQUIRE(index->KnnSearch(query, 3, kSindiSearchParameters, nullptr)->GetDim() == 3);

    std::string query_date_begin = "2026/05/01";
    std::string query_date_end = "2026/08";
    auto range_query = data.Query()
                           ->Paths(SINDI_DATE_BEGIN_PATH_NAME, &query_date_begin)
                           ->Paths(SINDI_DATE_END_PATH_NAME, &query_date_end);
    auto range = index->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(range->GetDim() == 2);
    REQUIRE((std::set<int64_t>(range->GetIds(), range->GetIds() + range->GetDim()) ==
             std::set<int64_t>{20, 30}));

    query_date_begin = "2026/05";
    query_date_end = "2026/08/01";
    auto partial_bucket_range = index->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(partial_bucket_range->GetDim() == 1);
    REQUIRE(partial_bucket_range->GetIds()[0] == 20);

    query_date_begin = "2026/05/01";
    query_date_end = "2026/08";
    uint32_t range_host_id = 2;
    range_query->UInt32Metadata(SINDI_HOST_ID_METADATA_NAME, &range_host_id);
    auto host_range = index->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(host_range->GetDim() == 1);
    REQUIRE(host_range->GetIds()[0] == 20);
    REQUIRE(index
                ->KnnSearch(
                    range_query, 3, kSindiSearchParameters, std::make_shared<AllowLabelFilter>(30))
                ->GetDim() == 0);

    uint32_t host_id = 2;
    query->UInt32Metadata(SINDI_HOST_ID_METADATA_NAME, &host_id);
    auto combined = index->KnnSearch(query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(combined->GetDim() == 2);
    REQUIRE(combined->GetIds()[0] == 10);
    REQUIRE(combined->GetIds()[1] == 20);

    SearchRequest request;
    request.query_ = query;
    request.topk_ = 3;
    request.params_str_ = kSindiSearchParameters;
    RequireSameResults(combined, index->SearchWithRequest(request));

    auto filtered =
        index->KnnSearch(query, 3, kSindiSearchParameters, std::make_shared<AllowLabelFilter>(20));
    REQUIRE(filtered->GetDim() == 1);
    REQUIRE(filtered->GetIds()[0] == 20);

    auto restored = std::make_unique<SINDI>(parameter, common_param);
    test_serializion(*index, *restored);
    RequireSameResults(combined, restored->KnnSearch(query, 3, kSindiSearchParameters, nullptr));
    RequireSameResults(host_range,
                       restored->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr));
    if (not immutable) {
        REQUIRE_THROWS_WITH(restored->Add(dated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI date-aware index does not support incremental Add"));
    }

    std::stringstream stream;
    REQUIRE_NOTHROW(index->SerializeStreaming(stream));
    const auto bytes = stream.str();
    auto streaming_restored = std::make_unique<SINDI>(parameter, common_param);
    REQUIRE_NOTHROW(streaming_restored->DeserializeStreaming(stream));
    RequireSameResults(combined,
                       streaming_restored->KnnSearch(query, 3, kSindiSearchParameters, nullptr));
    RequireSameResults(
        host_range, streaming_restored->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr));
    if (not immutable) {
        REQUIRE_THROWS_WITH(streaming_restored->Add(undated_add),
                            Catch::Matchers::ContainsSubstring(
                                "SINDI date-aware index does not support incremental Add"));
    }

    auto missing_date_block =
        EraseStreamingBlock(bytes, StreamSerializationTag::SINDI_DATE_METADATA);
    auto invalid_restored = std::make_unique<SINDI>(parameter, common_param);
    std::stringstream invalid_stream(missing_date_block);
    REQUIRE_THROWS(invalid_restored->DeserializeStreaming(invalid_stream));

    query_date = "2026/02/29";
    REQUIRE_THROWS(index->KnnSearch(query, 3, kSindiSearchParameters, nullptr));

    auto missing_range_end = data.Query()->Paths(SINDI_DATE_BEGIN_PATH_NAME, &query_date_begin);
    REQUIRE_THROWS(index->KnnSearch(missing_range_end, 3, kSindiSearchParameters, nullptr));

    query_date_begin = "2026/09";
    query_date_end = "2026/08";
    REQUIRE_THROWS(index->KnnSearch(range_query, 3, kSindiSearchParameters, nullptr));

    auto conflicting_query = data.Query()
                                 ->Paths(SINDI_DATE_PATH_NAME, &query_date)
                                 ->Paths(SINDI_DATE_BEGIN_PATH_NAME, &query_date_begin)
                                 ->Paths(SINDI_DATE_END_PATH_NAME, &query_date_end);
    REQUIRE_THROWS(index->KnnSearch(conflicting_query, 3, kSindiSearchParameters, nullptr));

    std::array<std::string, 4> invalid_buckets = {"2026", "2026/13", "2026/05/01", "2026/08"};
    SINDI invalid_bucket_index(parameter, common_param);
    REQUIRE_THROWS(invalid_bucket_index.Build(
        data.Base()->Paths(SINDI_DATE_PATH_NAME, invalid_buckets.data())));
}

TEST_CASE("SINDI date ranges preserve coarse bucket containment across years",
          "[ut][SINDI][metadata_filter][date_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t term_id = 1;
    std::array<float, 6> values{6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    std::array<int64_t, 6> labels{10, 11, 12, 20, 21, 22};
    std::array<std::string, 6> date_buckets{
        "2024/02", "2024/02/29", "2024", "2025/01", "2025/01/15", "2025"};
    std::array<SparseVector, 6> vectors{};
    for (uint64_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = SparseVector{1, &term_id, &values[i]};
    }
    auto base = Dataset::Make()
                    ->NumElements(vectors.size())
                    ->SparseVectors(vectors.data())
                    ->Ids(labels.data())
                    ->Paths(SINDI_DATE_PATH_NAME, date_buckets.data())
                    ->Owner(false);
    auto parameter = CreateSindiParameter(true, false);
    SINDI index(parameter, common_param);
    REQUIRE(index.Build(base).empty());

    float query_value = 1.0F;
    SparseVector query_vector{1, &term_id, &query_value};
    std::string query_date_begin = "2024/02/29";
    std::string query_date_end = "2025/01";
    auto query = Dataset::Make()
                     ->NumElements(1)
                     ->SparseVectors(&query_vector)
                     ->Paths(SINDI_DATE_BEGIN_PATH_NAME, &query_date_begin)
                     ->Paths(SINDI_DATE_END_PATH_NAME, &query_date_end)
                     ->Owner(false);
    constexpr auto search_parameters = R"({"sindi": {"n_candidate": 6}})";

    auto partial_range = index.KnnSearch(query, 6, search_parameters, nullptr);
    REQUIRE(partial_range->GetDim() == 3);
    REQUIRE((std::set<int64_t>(partial_range->GetIds(),
                               partial_range->GetIds() + partial_range->GetDim()) ==
             std::set<int64_t>{11, 20, 21}));

    query_date_begin = "2024/02";
    query_date_end = "2024/02";
    auto leap_month = index.KnnSearch(query, 6, search_parameters, nullptr);
    REQUIRE(leap_month->GetDim() == 2);
    REQUIRE((std::set<int64_t>(leap_month->GetIds(), leap_month->GetIds() + leap_month->GetDim()) ==
             std::set<int64_t>{10, 11}));

    query_date_begin = "2024";
    query_date_end = "2025";
    auto whole_years = index.KnnSearch(query, 6, search_parameters, nullptr);
    REQUIRE(whole_years->GetDim() == 6);
}

TEST_CASE("SINDI immutable host filter routes", "[ut][SINDI][host_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SmallSindiDataset data(0);
    constexpr uint32_t sparse_host_id = std::numeric_limits<uint32_t>::max();
    std::array<uint32_t, 4> host_ids = {sparse_host_id, 1, sparse_host_id, 1};
    auto base = data.Base()->UInt32Metadata("host_id", host_ids.data());
    auto parameter = CreateSindiParameter(true, false);
    auto index = std::make_unique<SINDI>(parameter, common_param);
    REQUIRE(index->Build(base) == std::vector<int64_t>{40});

    uint32_t host_id = 1;
    auto query = data.Query()->UInt32Metadata("host_id", &host_id);
    auto host_result = index->KnnSearch(query, 2, kSindiSearchParameters, nullptr);
    REQUIRE(host_result->GetDim() == 1);
    REQUIRE(host_result->GetIds()[0] == 30);
    SearchRequest request;
    request.query_ = query;
    request.topk_ = 2;
    request.params_str_ = kSindiSearchParameters;
    auto request_result = index->SearchWithRequest(request);
    REQUIRE(request_result->GetDim() == 1);
    REQUIRE(request_result->GetIds()[0] == 30);
    auto allow_twenty = std::make_shared<AllowLabelFilter>(20);
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, allow_twenty)->GetDim() == 0);
    REQUIRE(index->Remove({30}, RemoveMode::MARK_REMOVE) == 1);
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, nullptr)->GetDim() == 0);

    host_id = sparse_host_id;
    auto window_result = index->KnnSearch(query, 2, kSindiSearchParameters, nullptr);
    REQUIRE(window_result->GetDim() == 2);
    REQUIRE(window_result->GetIds()[0] == 10);
    for (int64_t i = 0; i < window_result->GetDim(); ++i) {
        REQUIRE((window_result->GetIds()[i] == 10 or window_result->GetIds()[i] == 20));
    }

    auto filtered_result = index->KnnSearch(query, 2, kSindiSearchParameters, allow_twenty);
    REQUIRE(filtered_result->GetDim() == 1);
    REQUIRE(filtered_result->GetIds()[0] == 20);
    REQUIRE(allow_twenty->GetValidIdsCalls() > 0);

    REQUIRE(index->Remove({10}, RemoveMode::MARK_REMOVE) == 1);
    auto deleted_result = index->KnnSearch(query, 2, kSindiSearchParameters, nullptr);
    REQUIRE(deleted_result->GetDim() == 1);
    REQUIRE(deleted_result->GetIds()[0] == 20);

    host_id = 0;
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, nullptr)->GetDim() == 0);
    host_id = 2;
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, nullptr)->GetDim() == 0);

    auto no_host_query = data.Query();
    REQUIRE(index->KnnSearch(no_host_query, 2, kSindiSearchParameters, nullptr)->GetDim() == 1);
}

TEST_CASE("SINDI host filter spans immutable windows", "[ut][SINDI][host_filter]") {
    constexpr uint32_t num_elements = 10002;
    constexpr uint32_t host_two_count = 5001;

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    std::vector<int64_t> labels(num_elements);
    std::vector<uint32_t> host_ids(num_elements);
    std::vector<float> values(num_elements, 1.0F);
    std::vector<SparseVector> sparse_vectors(num_elements);
    uint32_t term_id = 1;
    for (uint32_t i = 0; i < num_elements; ++i) {
        labels[i] = static_cast<int64_t>(i + 1);
        host_ids[i] = i < host_two_count ? 2 : 1;
        sparse_vectors[i].len_ = 1;
        sparse_vectors[i].ids_ = &term_id;
        sparse_vectors[i].vals_ = &values[i];
    }
    values[0] = 5.0F;

    auto base = Dataset::Make()
                    ->NumElements(num_elements)
                    ->SparseVectors(sparse_vectors.data())
                    ->Ids(labels.data())
                    ->UInt32Metadata("host_id", host_ids.data())
                    ->Owner(false);
    auto parameter = CreateSindiParameter(true, false);
    auto index = std::make_unique<SINDI>(parameter, common_param);
    REQUIRE(index->Build(base).empty());

    SparseVector query_vector{1, &term_id, values.data()};
    uint32_t host_id = 2;
    auto query = Dataset::Make()
                     ->NumElements(1)
                     ->SparseVectors(&query_vector)
                     ->UInt32Metadata("host_id", &host_id)
                     ->Owner(false);
    auto result = index->KnnSearch(query, 1, kSindiSearchParameters, nullptr);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 1);
}

TEST_CASE("SINDI host filter preserves term prune candidates at window boundaries",
          "[ut][SINDI][host_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    uint32_t term_id = 1;
    std::array<float, 4> values{10.0F, 9.0F, 2.0F, 1.0F};
    std::array<int64_t, 4> labels{10, 11, 20, 21};
    std::array<uint32_t, 4> host_ids{1, 1, 2, 2};
    std::array<SparseVector, 4> vectors;
    for (uint32_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = SparseVector{1, &term_id, &values[i]};
    }
    auto base = Dataset::Make()
                    ->NumElements(vectors.size())
                    ->SparseVectors(vectors.data())
                    ->Ids(labels.data())
                    ->UInt32Metadata("host_id", host_ids.data())
                    ->Owner(false);
    auto parameter = CreateSindiParameter(true, false);
    parameter->window_size = 4;
    SINDI index(parameter, common_param);
    REQUIRE(index.Build(base).empty());

    float query_value = 1.0F;
    SparseVector query_vector{1, &term_id, &query_value};
    uint32_t query_host_id = 2;
    auto query = Dataset::Make()
                     ->NumElements(1)
                     ->SparseVectors(&query_vector)
                     ->UInt32Metadata("host_id", &query_host_id)
                     ->Owner(false);
    const auto query_prune_ratio = GENERATE(0.0F, 0.2F);
    auto search_parameters = JsonType::Parse(kSindiSearchParameters);
    search_parameters["sindi"]["query_prune_ratio"].SetFloat(query_prune_ratio);
    search_parameters["sindi"]["term_prune_ratio"].SetFloat(0.5F);
    search_parameters["sindi"]["n_candidate"].SetInt(2);
    auto result = index.KnnSearch(query, 2, search_parameters.Dump(), nullptr);
    REQUIRE(result->GetDim() == 2);
    REQUIRE(result->GetIds()[0] == 20);
    REQUIRE(result->GetIds()[1] == 21);
}

TEST_CASE("SINDI host metadata accepts missing host ID zero", "[ut][SINDI][host_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SmallSindiDataset data(0);
    std::array<uint32_t, 4> host_ids = {0, 1, 2, 0};
    auto base = data.Base()->UInt32Metadata("host_id", host_ids.data());

    auto parameter = CreateSindiParameter(true, false);
    SINDI index(parameter, common_param);
    REQUIRE(index.Build(base) == std::vector<int64_t>{40});

    uint32_t host_id = 0;
    auto query = data.Query()->UInt32Metadata("host_id", &host_id);
    auto result = index.KnnSearch(query, 2, kSindiSearchParameters, nullptr);
    REQUIRE(result->GetDim() == 2);
    REQUIRE(result->GetIds()[0] == 10);
    REQUIRE(result->GetIds()[1] == 30);
}

TEST_CASE("SINDI host metadata rejects invalid serialized ranges",
          "[ut][SINDI][host_filter][streaming]") {
    std::stringstream stream;
    IOStreamWriter writer(stream);
    const uint64_t host_count = 1;
    const uint32_t host_id = 0;
    const uint64_t offset_count = 2;
    const std::array<uint32_t, 2> offsets{0, 1};
    const uint64_t range_count = 1;
    const SindiHostRange invalid_range{0, 2};
    StreamWriter::WriteObj(writer, host_count);
    StreamWriter::WriteObj(writer, host_id);
    StreamWriter::WriteObj(writer, offset_count);
    for (const auto offset : offsets) {
        StreamWriter::WriteObj(writer, offset);
    }
    StreamWriter::WriteObj(writer, range_count);
    StreamWriter::WriteObj(writer, invalid_range.begin);
    StreamWriter::WriteObj(writer, invalid_range.end);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SindiHostFilter host_filter(allocator.get());
    IOStreamReader reader(stream);
    REQUIRE_THROWS(host_filter.Deserialize(reader, 1));
}

TEST_CASE("SINDI date metadata rejects invalid element counts",
          "[ut][SINDI][metadata_filter][date_filter][serialization]") {
    const uint64_t element_count =
        GENERATE(uint64_t{0}, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1);
    std::stringstream stream;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SindiDateFilter date_filter(allocator.get());
    IOStreamReader reader(stream);
    REQUIRE_THROWS_WITH(date_filter.Deserialize(reader, element_count),
                        Catch::Matchers::ContainsSubstring(
                            "serialized SINDI date metadata element count must be in [1,"));
}

TEST_CASE("SINDI host route skips windows between disjoint ranges", "[ut][SINDI][host_filter]") {
    std::stringstream stream;
    IOStreamWriter writer(stream);
    const std::array<uint32_t, 2> host_ids{7, 8};
    const std::array<uint32_t, 3> offsets{0, 2, 3};
    const std::array<SindiHostRange, 3> ranges{
        SindiHostRange{0, 2}, SindiHostRange{10, 12}, SindiHostRange{2, 10}};
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(host_ids.size()));
    for (const auto host_id : host_ids) {
        StreamWriter::WriteObj(writer, host_id);
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(offsets.size()));
    for (const auto offset : offsets) {
        StreamWriter::WriteObj(writer, offset);
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(ranges.size()));
    for (const auto& range : ranges) {
        StreamWriter::WriteObj(writer, range.begin);
        StreamWriter::WriteObj(writer, range.end);
    }

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SindiHostFilter host_filter(allocator.get());
    IOStreamReader reader(stream);
    host_filter.Deserialize(reader, 12);

    uint32_t query_host_id = 7;
    auto query =
        Dataset::Make()->NumElements(1)->UInt32Metadata("host_id", &query_host_id)->Owner(false);
    const auto route = host_filter.Classify(query);
    REQUIRE(route.kind == SindiHostRouteKind::WINDOW);

    int64_t min_window_id = 0;
    int64_t max_window_id = 5;
    SindiHostFilter::ApplyWindowRoute(route, 2, min_window_id, max_window_id);
    REQUIRE(min_window_id == 0);
    REQUIRE(max_window_id == 5);
    REQUIRE(host_filter.NextMatchingWindow(route, 2, 0, max_window_id) == 0);
    REQUIRE(host_filter.NextMatchingWindow(route, 2, 1, max_window_id) == 5);
    REQUIRE_FALSE(host_filter.RequiresFullTermScan(route, 0, 2));
    REQUIRE_FALSE(host_filter.RequiresFullTermScan(route, 5, 2));

    auto inner_filter = std::make_shared<AllowLabelFilter>(10);
    FilterPtr filter = inner_filter;
    host_filter.ApplyFilter(route, filter);
    REQUIRE(filter->ValidRatio() == inner_filter->ValidRatio());
    REQUIRE(filter->FilterDistribution() == inner_filter->FilterDistribution());
}

TEST_CASE("SINDI host filter supports mutable immutable and reorder modes",
          "[ut][SINDI][host_filter]") {
    const bool immutable = GENERATE(false, true);
    const bool use_reorder = GENERATE(false, true);
    DYNAMIC_SECTION("immutable=" << immutable << ", use_reorder=" << use_reorder) {
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.metric_ = MetricType::METRIC_TYPE_IP;

        SmallSindiDataset data(0);
        std::array<uint32_t, 4> host_ids = {2, 0, 2, 0};
        auto base = data.Base()->UInt32Metadata("host_id", host_ids.data());
        auto parameter = CreateSindiParameter(immutable, false);
        parameter->use_reorder = use_reorder;
        parameter->rerank_type = SPARSE_RERANK_TYPE_FP32;
        SINDI index(parameter, common_param);
        REQUIRE(index.Build(base) == std::vector<int64_t>{40});

        uint32_t host_id = 0;
        auto query = data.Query()->UInt32Metadata("host_id", &host_id);
        auto result = index.KnnSearch(query, 2, kSindiSearchParameters, nullptr);
        REQUIRE(result->GetDim() == 1);
        REQUIRE(result->GetIds()[0] == 30);

        host_id = 2;
        result = index.KnnSearch(query, 2, kSindiSearchParameters, nullptr);
        REQUIRE(result->GetDim() == 2);
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            REQUIRE((result->GetIds()[i] == 10 or result->GetIds()[i] == 20));
        }

        if (not immutable) {
            std::array<int64_t, 2> added_labels{50, 60};
            std::array<uint32_t, 2> added_hosts{0, 2};
            std::array<SparseVector, 2> added_vectors{data.sparse_vectors[0],
                                                      data.sparse_vectors[0]};
            auto missing_host_metadata = Dataset::Make()
                                             ->NumElements(added_vectors.size())
                                             ->SparseVectors(added_vectors.data())
                                             ->Ids(added_labels.data())
                                             ->Owner(false);
            REQUIRE_THROWS(index.Add(missing_host_metadata));
            auto added = Dataset::Make()
                             ->NumElements(added_vectors.size())
                             ->SparseVectors(added_vectors.data())
                             ->Ids(added_labels.data())
                             ->UInt32Metadata("host_id", added_hosts.data())
                             ->Owner(false);
            REQUIRE(index.Add(added).empty());
            host_id = 0;
            result = index.KnnSearch(query, 2, kSindiSearchParameters, nullptr);
            REQUIRE(result->GetDim() == 2);
            REQUIRE(result->GetIds()[0] == 50);
            REQUIRE(result->GetIds()[1] == 30);

            std::array<int64_t, 2> second_added_labels{70, 80};
            std::array<uint32_t, 2> second_added_hosts{2, 0};
            auto second_added = Dataset::Make()
                                    ->NumElements(added_vectors.size())
                                    ->SparseVectors(added_vectors.data())
                                    ->Ids(second_added_labels.data())
                                    ->UInt32Metadata("host_id", second_added_hosts.data())
                                    ->Owner(false);
            REQUIRE(index.Add(second_added).empty());
            result = index.KnnSearch(query, 3, kSindiSearchParameters, nullptr);
            REQUIRE(result->GetDim() == 3);
            std::set<int64_t> result_ids(result->GetIds(), result->GetIds() + result->GetDim());
            REQUIRE((result_ids == std::set<int64_t>{30, 50, 80}));
        }
    }
}

TEST_CASE("SINDI small host uses posting scan", "[ut][SINDI][host_filter]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SmallSindiDataset data(0);
    std::array<uint32_t, 4> host_ids = {2, 1, 2, 1};
    auto base = data.Base()->UInt32Metadata("host_id", host_ids.data());
    auto parameter = CreateSindiParameter(true, false);
    parameter->rerank_type = SPARSE_RERANK_TYPE_FP32;
    auto index = std::make_unique<SINDI>(parameter, common_param);
    REQUIRE(index->Build(base) == std::vector<int64_t>{40});

    uint32_t host_id = 1;
    auto query = data.Query()->UInt32Metadata("host_id", &host_id);
    auto result = index->KnnSearch(query, 2, kSindiSearchParameters, nullptr);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 30);

    auto allow_twenty = std::make_shared<AllowLabelFilter>(20);
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, allow_twenty)->GetDim() == 0);
    REQUIRE(index->Remove({30}, RemoveMode::MARK_REMOVE) == 1);
    REQUIRE(index->KnnSearch(query, 2, kSindiSearchParameters, nullptr)->GetDim() == 0);
}

TEST_CASE("SINDI legacy deserialize clears host metadata",
          "[ut][SINDI][host_filter][serialization]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    SmallSindiDataset data(0);
    auto parameter = CreateSindiParameter(false, false);
    SINDI legacy_source(parameter, common_param);
    REQUIRE(legacy_source.Build(data.Base()) == std::vector<int64_t>{40});

    uint32_t query_host_id = 99;
    auto query = data.Query()->UInt32Metadata("host_id", &query_host_id);
    auto expected = legacy_source.KnnSearch(query, 3, kSindiSearchParameters, nullptr);
    REQUIRE(expected->GetDim() == 3);

    std::stringstream legacy_stream;
    IOStreamWriter legacy_writer(legacy_stream);
    legacy_source.Serialize(legacy_writer);

    std::array<uint32_t, 4> host_ids{2, 0, 2, 0};
    SINDI restored(parameter, common_param);
    REQUIRE(restored.Build(data.Base()->UInt32Metadata("host_id", host_ids.data())) ==
            std::vector<int64_t>{40});
    REQUIRE(restored.KnnSearch(query, 3, kSindiSearchParameters, nullptr)->GetDim() == 0);

    legacy_stream.seekg(0, std::ios::beg);
    IOStreamReader legacy_reader(legacy_stream);
    REQUIRE_NOTHROW(restored.Deserialize(legacy_reader));
    RequireSameResults(expected, restored.KnnSearch(query, 3, kSindiSearchParameters, nullptr));
}

TEST_CASE("SINDI host serialization supports mutable and immutable",
          "[ut][SINDI][host_filter][serialization][streaming]") {
    const bool immutable = GENERATE(false, true);
    DYNAMIC_SECTION("immutable=" << immutable) {
        SmallSindiDataset data(0);
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.metric_ = MetricType::METRIC_TYPE_IP;

        std::array<uint32_t, 4> host_ids{2, 0, 2, 0};
        auto base = data.Base()->UInt32Metadata("host_id", host_ids.data());
        auto parameter = CreateSindiParameter(immutable, false);
        parameter->use_reorder = false;
        parameter->rerank_type = SPARSE_RERANK_TYPE_FP32;
        SINDI index(parameter, common_param);
        REQUIRE(index.Build(base) == std::vector<int64_t>{40});

        if (!immutable) {
            std::array<int64_t, 2> added_labels{50, 60};
            std::array<uint32_t, 2> added_hosts{0, 2};
            std::array<SparseVector, 2> added_vectors{data.sparse_vectors[0],
                                                      data.sparse_vectors[0]};
            auto added = Dataset::Make()
                             ->NumElements(added_vectors.size())
                             ->SparseVectors(added_vectors.data())
                             ->Ids(added_labels.data())
                             ->UInt32Metadata("host_id", added_hosts.data())
                             ->Owner(false);
            REQUIRE(index.Add(added).empty());
        }

        uint32_t host_id = 0;
        auto query = data.Query()->UInt32Metadata("host_id", &host_id);
        auto expected = index.KnnSearch(query, 3, kSindiSearchParameters, nullptr);

        std::stringstream legacy_stream;
        IOStreamWriter legacy_writer(legacy_stream);
        REQUIRE_NOTHROW(index.Serialize(legacy_writer));
        legacy_stream.seekg(0, std::ios::beg);
        SINDI legacy_restored(parameter, common_param);
        IOStreamReader legacy_reader(legacy_stream);
        REQUIRE_NOTHROW(legacy_restored.Deserialize(legacy_reader));
        RequireSameResults(expected,
                           legacy_restored.KnnSearch(query, 3, kSindiSearchParameters, nullptr));

        std::stringstream stream;
        REQUIRE_NOTHROW(index.SerializeStreaming(stream));
        const auto bytes = stream.str();

        SINDI restored(parameter, common_param);
        std::stringstream deserialize_stream(bytes);
        REQUIRE_NOTHROW(restored.DeserializeStreaming(deserialize_stream));
        RequireSameResults(expected, restored.KnnSearch(query, 3, kSindiSearchParameters, nullptr));

        std::stringstream load_stream(bytes);
        auto loaded = Index::Load(load_stream, "{}");
        REQUIRE(loaded.has_value());
        auto loaded_result = loaded.value()->KnnSearch(query, 3, kSindiSearchParameters).value();
        RequireSameResults(expected, loaded_result);

        if (!immutable) {
            int64_t added_label = 70;
            uint32_t added_host = 0;
            auto added = Dataset::Make()
                             ->NumElements(1)
                             ->SparseVectors(&data.sparse_vectors[0])
                             ->Ids(&added_label)
                             ->UInt32Metadata("host_id", &added_host)
                             ->Owner(false);
            REQUIRE(restored.Add(added).empty());
            REQUIRE(restored.KnnSearch(query, 4, kSindiSearchParameters, nullptr)->GetDim() == 3);
        }

        auto missing_host = EraseStreamingBlock(bytes, StreamSerializationTag::SINDI_HOST_METADATA);
        SINDI missing_restored(parameter, common_param);
        std::stringstream missing_stream(missing_host);
        REQUIRE_THROWS(missing_restored.DeserializeStreaming(missing_stream));

        auto unsupported_host =
            SetStreamingBlockVersion(bytes, StreamSerializationTag::SINDI_HOST_METADATA, 2);
        SINDI unsupported_restored(parameter, common_param);
        std::stringstream unsupported_stream(unsupported_host);
        REQUIRE_THROWS(unsupported_restored.DeserializeStreaming(unsupported_stream));
    }
}
