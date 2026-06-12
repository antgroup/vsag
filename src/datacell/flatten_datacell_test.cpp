
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

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
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
                "rabitq_version": "split",
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
                "rabitq_version": "split",
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
                "rabitq_version": "split",
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

TEST_CASE("RaBitQSplitDataCell hybrid IO (1bit in memory, supplement on disk)",
          "[ut][RaBitQSplitDataCell]") {
    // Verifies the mixed IO mode: one-bit traversal codes stay in memory
    // (block_memory_io) while xbit supplement codes are backed by file IO
    // (async_io, which transparently falls back to buffer_io when libaio
    // is unavailable). Behaviour must be numerically identical to the
    // memory-only baseline. Because RaBitQ training uses a random
    // orthogonal projection (std::random_device-seeded), the two cells
    // cannot be trained independently and still produce identical codes.
    // Instead we train + populate the memory-only cell first and then
    // Serialize -> Deserialize into the hybrid cell so both share the
    // same projection matrix and bytes.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 50;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto queries = fixtures::generate_vectors(4, dim, 31);

    const std::string tmp_prefix = "/tmp/vsag_rabitq_split_hybrid_ut_" + std::to_string(::getpid());
    struct TempFileCleanup {
        explicit TempFileCleanup(std::string prefix) : prefix_(std::move(prefix)) {
            cleanup();
        }

        ~TempFileCleanup() {
            cleanup();
        }

        void
        cleanup() const {
            for (const auto& suffix : {"_base", "_base_onebit", "_base_supplement"}) {
                std::remove((prefix_ + suffix).c_str());
            }
        }

        std::string prefix_;
    } cleanup(tmp_prefix);

    struct SplitCase {
        uint64_t base_bits;
        uint64_t filter_bits;
    };
    const SplitCase split_cases[] = {{1, 1}, {4, 1}, {7, 1}, {8, 3}};
    for (const auto split_case : split_cases) {
        auto memory_param_str = fmt::format(R"({{
            "codes_type": "rabitq_split",
            "io_params": {{ "type": "block_memory_io" }},
            "quantization_params": {{
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": {},
                "rabitq_bits_per_dim_filter": {}
            }}
        }})",
                                            split_case.base_bits,
                                            split_case.filter_bits);
        auto hybrid_param_str = fmt::format(R"({{
            "codes_type": "rabitq_split",
            "io_params": {{
                "type": "block_memory_io",
                "file_path": "{}_base"
            }},
            "supplement_io_params": {{
                "type": "async_io"
            }},
            "quantization_params": {{
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": {},
                "rabitq_bits_per_dim_filter": {}
            }}
        }})",
                                            tmp_prefix,
                                            split_case.base_bits,
                                            split_case.filter_bits);

        auto mem_param = std::make_shared<FlattenDataCellParameter>();
        mem_param->FromJson(JsonType::Parse(memory_param_str));
        auto hyb_param = std::make_shared<FlattenDataCellParameter>();
        hyb_param->FromJson(JsonType::Parse(hybrid_param_str));
        REQUIRE(hyb_param->supplement_io_parameter != nullptr);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

        auto mem_cell = FlattenInterface::MakeInstance(mem_param, common_param);
        auto hyb_cell = FlattenInterface::MakeInstance(hyb_param, common_param);

        REQUIRE(mem_cell->InMemory());
        REQUIRE_FALSE(hyb_cell->InMemory());

        mem_cell->Train(vectors.data(), count);
        mem_cell->BatchInsertVector(vectors.data(), count);

        // Sync trained model + codes from mem_cell into hyb_cell so both
        // share the same RaBitQ projection / codes.
        std::stringstream ss;
        IOStreamWriter writer(ss);
        mem_cell->Serialize(writer);
        IOStreamReader reader(ss);
        hyb_cell->Deserialize(reader);
        REQUIRE(hyb_cell->TotalCount() == mem_cell->TotalCount());

        std::vector<InnerIdType> idx(count);
        std::iota(idx.begin(), idx.end(), 0);

        for (uint64_t query_id = 0; query_id < 4; ++query_id) {
            auto* query = queries.data() + query_id * dim;
            auto mem_computer = mem_cell->FactoryComputer(query);
            auto hyb_computer = hyb_cell->FactoryComputer(query);

            std::vector<float> mem_dists(count);
            std::vector<float> hyb_dists(count);
            mem_cell->Query(mem_dists.data(), mem_computer, idx.data(), count);
            hyb_cell->Query(hyb_dists.data(), hyb_computer, idx.data(), count);
            for (InnerIdType id = 0; id < count; ++id) {
                REQUIRE(mem_dists[id] == hyb_dists[id]);
            }

            std::vector<float> mem_lb(count), hyb_lb(count);
            mem_cell->QueryWithDistanceLowerBound(
                mem_dists.data(), mem_lb.data(), mem_computer, idx.data(), count);
            hyb_cell->QueryWithDistanceLowerBound(
                hyb_dists.data(), hyb_lb.data(), hyb_computer, idx.data(), count);
            for (InnerIdType id = 0; id < count; ++id) {
                REQUIRE(mem_dists[id] == hyb_dists[id]);
                REQUIRE(mem_lb[id] == hyb_lb[id]);
            }

            std::vector<float> mem_hint_dists(count);
            std::vector<float> hyb_hint_dists(count);
            mem_cell->QueryWithDistanceHint(
                mem_hint_dists.data(), mem_dists.data(), mem_computer, idx.data(), count);
            hyb_cell->QueryWithDistanceHint(
                hyb_hint_dists.data(), hyb_dists.data(), hyb_computer, idx.data(), count);
            for (InnerIdType id = 0; id < count; ++id) {
                REQUIRE(mem_hint_dists[id] == hyb_hint_dists[id]);
            }
        }
    }
}
