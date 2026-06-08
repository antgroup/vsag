
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

#include "flatten_datacell.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include "flatten_interface_test.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "unittest.h"

using namespace vsag;

void
TestFlattenDataCell(FlattenDataCellParamPtr& param,
                    IndexCommonParam& common_param,
                    float error = 1e-3) {
    auto count = GENERATE(100, 1000);
    auto flatten = FlattenInterface::MakeInstance(param, common_param);

    FlattenInterfaceTest test(flatten, common_param.metric_);
    test.BasicTest(common_param.dim_, count, error);
    auto other = FlattenInterface::MakeInstance(param, common_param);
    test.TestSerializeAndDeserialize(common_param.dim_, other, error);
}

TEST_CASE("FlattenDataCell Basic Test", "[ut][FlattenDataCell] ") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64, 512);
    std::string io_type = GENERATE("memory_io", "block_memory_io");
    std::vector<std::pair<std::string, float>> quantizer_errors = {{"sq8", 2e-2f}, {"fp32", 1e-5}};
    MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    constexpr const char* param_temp =
        R"(
        {{
            "io_params": {{
                "type": "{}"
            }},
            "quantization_params": {{
                "type": "{}"
            }}
        }}
        )";
    for (auto& quantizer_error : quantizer_errors) {
        for (auto& metric : metrics) {
            auto param_str = fmt::format(param_temp, io_type, quantizer_error.first);
            auto param_json = JsonType::Parse(param_str);
            auto param = std::make_shared<FlattenDataCellParameter>();
            param->FromJson(param_json);
            IndexCommonParam common_param;
            common_param.allocator_ = allocator;
            common_param.dim_ = dim;
            common_param.metric_ = metric;

            TestFlattenDataCell(param, common_param, quantizer_error.second);
        }
    }
}

TEST_CASE("RaBitQSplitDataCell direct split compute", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 32;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto queries = fixtures::generate_vectors(4, dim, 17);

    constexpr const char* param_temp = R"(
        {{
            "codes_type": "rabitq_split",
            "io_params": {{
                "type": "memory_io"
            }},
            "quantization_params": {{
                "type": "rabitq",
                "rabitq_version": "split_1bit_7bit",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": {}
            }}
        }}
        )";

    for (uint64_t base_bits = 1; base_bits <= 8; ++base_bits) {
        auto param_json = JsonType::Parse(fmt::format(param_temp, base_bits));
        auto param = std::make_shared<FlattenDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

        auto flatten = FlattenInterface::MakeInstance(param, common_param);
        flatten->Train(vectors.data(), count);
        flatten->BatchInsertVector(vectors.data(), count);

        std::vector<InnerIdType> idx(count);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> dists(count);
        std::vector<float> lower_bounds(count);
        for (uint64_t query_id = 0; query_id < 4; ++query_id) {
            auto* query = queries.data() + query_id * dim;
            auto computer = flatten->FactoryComputer(query);
            auto* rabitq_computer =
                static_cast<Computer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>>*>(
                    computer.get());

            flatten->Query(dists.data(), computer, idx.data(), count);
            for (InnerIdType id = 0; id < count; ++id) {
                bool need_release = false;
                const auto* full_code = flatten->GetCodesById(id, need_release);
                float merged_dist = 0.0F;
                rabitq_computer->ComputeDist(full_code, &merged_dist);
                if (need_release) {
                    flatten->Release(full_code);
                }
                REQUIRE(std::abs(dists[id] - merged_dist) <= 1e-6F);
            }

            flatten->QueryWithDistanceLowerBound(
                dists.data(), lower_bounds.data(), computer, idx.data(), count);
            for (InnerIdType id = 0; id < count; ++id) {
                REQUIRE(std::isfinite(dists[id]));
                REQUIRE(std::isfinite(lower_bounds[id]));
                REQUIRE(lower_bounds[id] <= dists[id] + 1e-5F);
            }
        }
    }
}
TEST_CASE("RaBitQSplitDataCell serialize and methods", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 32;
    auto vectors = fixtures::generate_vectors(count, dim);

    constexpr const char* param_str = R"(
        {
            "codes_type": "rabitq_split",
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "rabitq",
                "rabitq_version": "split_1bit_7bit",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 4
            }
        }
        )";

    auto param_json = JsonType::Parse(param_str);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);

    SECTION("InsertVector and UpdateVector") {
        for (InnerIdType i = 0; i < count; ++i) {
            flatten->InsertVector(vectors.data() + i * dim);
        }
        REQUIRE(flatten->TotalCount() == count);

        REQUIRE(flatten->UpdateVector(vectors.data(), 0) == true);
        REQUIRE(flatten->UpdateVector(vectors.data(), count + 10) == false);
    }

    SECTION("BatchInsertVector with explicit ids") {
        std::vector<InnerIdType> ids(count);
        std::iota(ids.begin(), ids.end(), 0);
        flatten->BatchInsertVector(vectors.data(), count, ids.data());
        REQUIRE(flatten->TotalCount() == count);
    }

    SECTION("Serialize and Deserialize") {
        flatten->BatchInsertVector(vectors.data(), count);

        std::stringstream ss;
        IOStreamWriter writer(ss);
        flatten->Serialize(writer);
        ss.seekg(0, std::ios::beg);
        IOStreamReader reader(ss);

        auto other = FlattenInterface::MakeInstance(param, common_param);
        other->Train(vectors.data(), count);
        other->Deserialize(reader);
        REQUIRE(other->TotalCount() == flatten->TotalCount());

        auto query = fixtures::generate_vectors(1, dim, 99);
        auto computer = flatten->FactoryComputer(query.data());
        std::vector<InnerIdType> idx(count);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> dists1(count), dists2(count);
        flatten->Query(dists1.data(), computer, idx.data(), count);
        other->Query(dists2.data(), computer, idx.data(), count);
        for (InnerIdType i = 0; i < count; ++i) {
            REQUIRE(dists1[i] == dists2[i]);
        }
    }

    SECTION("GetCodesById") {
        flatten->BatchInsertVector(vectors.data(), count);
        bool need_release = false;
        const auto* code0 = flatten->GetCodesById(0, need_release);
        REQUIRE(code0 != nullptr);
        if (need_release) {
            flatten->Release(code0);
        }
    }

    SECTION("Encode and Decode") {
        flatten->BatchInsertVector(vectors.data(), count);
        auto code_size = flatten->code_size_;
        std::vector<uint8_t> codes(code_size);
        REQUIRE(flatten->Encode(vectors.data(), codes.data()) == true);
        std::vector<float> decoded(dim);
        flatten->Decode(codes.data(), decoded.data());
    }

    SECTION("Resize and ShrinkToFit") {
        flatten->BatchInsertVector(vectors.data(), count);
        flatten->Resize(count * 2);
        flatten->ShrinkToFit(count);
    }

    SECTION("Move") {
        flatten->BatchInsertVector(vectors.data(), count);
        flatten->Move(0, count);
    }

    SECTION("GetCodesById variants") {
        flatten->BatchInsertVector(vectors.data(), count);
        bool need_release = false;
        const auto* codes = flatten->GetCodesById(0, need_release);
        REQUIRE(codes != nullptr);
        if (need_release) {
            flatten->Release(codes);
        }

        auto code_size = flatten->code_size_;
        std::vector<uint8_t> buf(code_size);
        REQUIRE(flatten->GetCodesById(0, buf.data()) == true);
    }

    SECTION("ExportModel") {
        flatten->BatchInsertVector(vectors.data(), count);
        auto other = FlattenInterface::MakeInstance(param, common_param);
        other->Train(vectors.data(), count);
        flatten->ExportModel(other);
    }

    SECTION("MergeOther") {
        flatten->BatchInsertVector(vectors.data(), count / 2);
        auto other_param = std::make_shared<FlattenDataCellParameter>();
        other_param->FromJson(param_json);
        auto other = FlattenInterface::MakeInstance(other_param, common_param);
        other->Train(vectors.data(), count);
        other->BatchInsertVector(vectors.data() + (count / 2) * dim, count / 2);
        flatten->MergeOther(other, count / 2);
        REQUIRE(flatten->TotalCount() == count);
    }

    SECTION("Metadata methods") {
        REQUIRE_FALSE(flatten->GetQuantizerName().empty());
        REQUIRE(flatten->GetMetricType() == MetricType::METRIC_TYPE_L2SQR);
        REQUIRE(flatten->InMemory() == true);
        auto memory = flatten->GetMemoryUsage();
        REQUIRE(memory > 0);
    }

    SECTION("QueryWithDistanceFilter") {
        flatten->BatchInsertVector(vectors.data(), count);
        auto query = fixtures::generate_vectors(1, dim, 42);
        auto computer = flatten->FactoryComputer(query.data());
        std::vector<InnerIdType> idx(count);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> dists(count);
        flatten->QueryWithDistanceFilter(
            dists.data(), computer, idx.data(), count, std::numeric_limits<float>::max());
        for (InnerIdType i = 0; i < count; ++i) {
            REQUIRE(std::isfinite(dists[i]));
        }
    }
}

TEST_CASE("RaBitQSplitDataCell IP metric", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 16;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto queries = fixtures::generate_vectors(2, dim, 42);

    constexpr const char* param_str = R"(
        {
            "codes_type": "rabitq_split",
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "rabitq",
                "rabitq_version": "split_1bit_7bit",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 4
            }
        }
        )";

    auto param_json = JsonType::Parse(param_str);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);

    std::vector<InnerIdType> idx(count);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<float> dists(count);
    std::vector<float> lower_bounds(count);

    auto computer = flatten->FactoryComputer(queries.data());
    flatten->Query(dists.data(), computer, idx.data(), count);
    for (InnerIdType i = 0; i < count; ++i) {
        REQUIRE(std::isfinite(dists[i]));
    }

    flatten->QueryWithDistanceLowerBound(
        dists.data(), lower_bounds.data(), computer, idx.data(), count);
    for (InnerIdType i = 0; i < count; ++i) {
        REQUIRE(std::isfinite(dists[i]));
    }
}
