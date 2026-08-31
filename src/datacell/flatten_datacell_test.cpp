
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <numeric>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>

#include "flatten_interface_test.h"
#include "flatten_optimized_build_interface.h"
#include "hgraph_rabitq_fused_datacell.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer.h"
#include "rabitq_split_datacell.h"
#include "unittest.h"

using namespace vsag;

namespace {

bool
IsNaNBitPattern(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7FFFFFFFU) > 0x7F800000U;
}

class RejectSecondThreadPool final : public ThreadPool {
public:
    ~RejectSecondThreadPool() override {
        this->WaitUntilEmpty();
    }

    void
    WaitUntilEmpty() override {
        release_.store(true, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void
    SetQueueSizeLimit(uint64_t) override {
    }

    void
    SetPoolSize(uint64_t) override {
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        const uint64_t submission = submissions_.fetch_add(1, std::memory_order_relaxed);
        if (submission > 0) {
            release_.store(true, std::memory_order_release);
            throw std::runtime_error("injected enqueue failure");
        }

        worker_ = std::thread([this, task = std::move(task)]() mutable {
            while (not release_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            task_started_.store(true, std::memory_order_release);
            task();
        });
        return {};
    }

    [[nodiscard]] bool
    TaskStarted() const {
        return task_started_.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> submissions_{0};
    std::atomic<bool> release_{false};
    std::atomic<bool> task_started_{false};
    std::thread worker_{};
};

}  // namespace

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
TEST_CASE("RaBitQSplitDataCell fused residual clusters", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr uint64_t cluster_count = 16;
    constexpr uint64_t vectors_per_cluster = 20;
    constexpr uint64_t count = cluster_count * vectors_per_cluster;
    Vector<float> vectors(count * dim, 0.0F, allocator.get());
    for (uint64_t cluster = 0; cluster < cluster_count; ++cluster) {
        for (uint64_t row = 0; row < vectors_per_cluster; ++row) {
            auto* vector = vectors.data() + (cluster * vectors_per_cluster + row) * dim;
            vector[cluster] = 100.0F;
            vector[(cluster + 17) % dim] = static_cast<float>(row) * 0.001F;
        }
    }

    auto param_json = JsonType::Parse(R"({
        "codes_type": "rabitq_split",
        "io_params": {"type": "memory_io"},
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "use_fht": true
        }
    })");
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    REQUIRE(split->FusedFilterBits() == 1);
    REQUIRE(split->FusedSupplementBits() == 7);
    REQUIRE(split->UsesLegacyHnswFusedCodec());
    split->TrainFusedCodec(vectors.data(), count, cluster_count);
    const auto codec_model = split->ExportFusedCodec();
    REQUIRE_FALSE(codec_model.empty());

    auto invalid_centroid_count = codec_model;
    const uint64_t oversized_count = std::numeric_limits<uint64_t>::max();
    std::memcpy(invalid_centroid_count.data() + sizeof(uint32_t),
                &oversized_count,
                sizeof(oversized_count));
    REQUIRE_THROWS(split->ImportFusedCodec(invalid_centroid_count));

    auto trailing_codec_model = codec_model;
    trailing_codec_model.push_back('\0');
    REQUIRE_THROWS(split->ImportFusedCodec(trailing_codec_model));

    Vector<uint8_t> one_bit(split->OneBitCodeSize(), allocator.get());
    Vector<uint8_t> supplement(split->SupplementCodeSize(), allocator.get());
    const float invalid_values[] = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    constexpr uint8_t output_sentinel = 0xA5;
    for (const float invalid_value : invalid_values) {
        CAPTURE(invalid_value);
        std::vector<float> invalid_training(vectors.data(), vectors.data() + vectors.size());
        invalid_training[0] = invalid_value;
        REQUIRE_THROWS(split->TrainFusedCodec(invalid_training.data(), count, cluster_count));
        REQUIRE(split->ExportFusedCodec() == codec_model);

        std::vector<float> invalid_vector(vectors.data(), vectors.data() + dim);
        invalid_vector[0] = invalid_value;
        std::fill(one_bit.begin(), one_bit.end(), output_sentinel);
        std::fill(supplement.begin(), supplement.end(), output_sentinel);
        uint32_t invalid_cluster_id = cluster_count;
        REQUIRE_FALSE(split->EncodeFused(
            invalid_vector.data(), one_bit.data(), supplement.data(), &invalid_cluster_id));
        REQUIRE(invalid_cluster_id == cluster_count);
        REQUIRE(std::all_of(one_bit.begin(), one_bit.end(), [](uint8_t value) {
            return value == output_sentinel;
        }));
        REQUIRE(std::all_of(supplement.begin(), supplement.end(), [](uint8_t value) {
            return value == output_sentinel;
        }));
    }

    UnorderedSet<uint32_t> assigned_clusters(allocator.get());
    for (uint64_t cluster = 0; cluster < cluster_count; ++cluster) {
        uint32_t cluster_id = 0;
        REQUIRE(split->EncodeFused(vectors.data() + cluster * vectors_per_cluster * dim,
                                   one_bit.data(),
                                   supplement.data(),
                                   &cluster_id));
        assigned_clusters.insert(cluster_id);
    }
    REQUIRE(assigned_clusters.size() == cluster_count);

    auto computer = split->FactoryFusedComputer(vectors.data());
    uint32_t cluster_id = 0;
    REQUIRE(split->EncodeFused(vectors.data(), one_bit.data(), supplement.data(), &cluster_id));
    float coarse_distance = 0.0F;
    float lower_bound = 0.0F;
    float filter_inner_product = 0.0F;
    REQUIRE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                  cluster_id,
                                                  one_bit.data(),
                                                  supplement.data(),
                                                  &coarse_distance,
                                                  &lower_bound,
                                                  &filter_inner_product,
                                                  nullptr));
    REQUIRE(IsNaNBitPattern(filter_inner_product));
    float full_distance = 0.0F;
    REQUIRE_FALSE(split->ComputeFusedFullWithFilterIP(
        computer, cluster_id, one_bit.data(), supplement.data(), 0.0F, &full_distance, nullptr));
    REQUIRE(split->ComputeFusedFull(
        computer, cluster_id, one_bit.data(), supplement.data(), &full_distance, nullptr));
    REQUIRE(std::isfinite(coarse_distance));
    REQUIRE(std::isfinite(lower_bound));
    REQUIRE(std::isfinite(full_distance));

    QueryContext narrow_context;
    narrow_context.rabitq_error_rate = 0.95F;
    QueryContext wide_context;
    wide_context.rabitq_error_rate = 3.8F;
    float narrow_distance = 0.0F;
    float narrow_lower_bound = 0.0F;
    float wide_distance = 0.0F;
    float wide_lower_bound = 0.0F;
    REQUIRE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                  cluster_id,
                                                  one_bit.data(),
                                                  supplement.data(),
                                                  &narrow_distance,
                                                  &narrow_lower_bound,
                                                  nullptr,
                                                  &narrow_context));
    REQUIRE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                  cluster_id,
                                                  one_bit.data(),
                                                  supplement.data(),
                                                  &wide_distance,
                                                  &wide_lower_bound,
                                                  nullptr,
                                                  &wide_context));
    REQUIRE(std::abs(coarse_distance - narrow_distance) <= 1e-6F);
    REQUIRE(std::abs(coarse_distance - wide_distance) <= 1e-6F);
    const float default_gap = coarse_distance - lower_bound;
    const float narrow_gap = narrow_distance - narrow_lower_bound;
    const float wide_gap = wide_distance - wide_lower_bound;
    REQUIRE(default_gap > 1e-6F);
    REQUIRE(std::abs(narrow_gap - 0.5F * default_gap) <= 1e-4F * default_gap + 1e-6F);
    REQUIRE(std::abs(wide_gap - 2.0F * default_gap) <= 1e-4F * default_gap + 1e-6F);
}

TEST_CASE("RaBitQSplitDataCell native fused bit splits", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint32_t cluster_count = 16;
    constexpr InnerIdType count = 64;
    struct FusedSplitCase {
        uint32_t filter_bits;
        uint32_t supplement_bits;
        uint64_t dim;
    };
    const FusedSplitCase cases[] = {
        {1, 3, 65},
        {2, 6, 960},
        {3, 5, 65},
        {4, 4, 960},
    };

    constexpr const char* param_template = R"(
        {{
            "codes_type": "rabitq_split",
            "io_params": {{
                "type": "memory_io"
            }},
            "quantization_params": {{
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": {},
                "rabitq_bits_per_dim_filter": {},
                "use_fht": true
            }}
        }}
    )";

    for (const auto& split_case : cases) {
        CAPTURE(split_case.filter_bits, split_case.supplement_bits, split_case.dim);
        const uint32_t base_bits = split_case.filter_bits + split_case.supplement_bits;
        auto param_json =
            JsonType::Parse(fmt::format(param_template, base_bits, split_case.filter_bits));
        auto param = std::make_shared<FlattenDataCellParameter>();
        param->FromJson(param_json);

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = split_case.dim;
        common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

        auto vectors = fixtures::generate_vectors(
            count, split_case.dim, false, 31 + static_cast<int>(split_case.filter_bits));
        auto query = fixtures::generate_vectors(
            1, split_case.dim, false, 71 + static_cast<int>(split_case.filter_bits));
        auto encoded_vectors = fixtures::generate_vectors(
            3, split_case.dim, false, 51 + static_cast<int>(split_case.filter_bits));
        auto flatten = FlattenInterface::MakeInstance(param, common_param);
        flatten->Train(vectors.data(), count);
        auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
        REQUIRE(split != nullptr);
        REQUIRE(split->FusedFilterBits() == split_case.filter_bits);
        REQUIRE(split->FusedSupplementBits() == split_case.supplement_bits);
        REQUIRE_FALSE(split->UsesLegacyHnswFusedCodec());

        split->TrainFusedCodec(vectors.data(), count, cluster_count);
        auto graph_param = std::make_shared<GraphDataCellParameter>();
        graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        graph_param->max_degree_ = 8;
        graph_param->init_max_capacity_ = 4;
        auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
            graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
        split->AttachFusedCodeStorage(graph.get());
        auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
        REQUIRE(optimized_build != nullptr);
        auto build_pool = SafeThreadPool::FactoryDefaultThreadPool();
        build_pool->SetPoolSize(4);
        flatten->Resize(4);
        REQUIRE(optimized_build->BeginOptimizedBuild({build_pool, 4}));
        Vector<uint8_t> filter_code(split->OneBitCodeSize(), allocator.get());
        for (InnerIdType id = 0; id < 4; ++id) {
            flatten->InsertVector(
                encoded_vectors.data() + static_cast<uint64_t>(id % 3) * split_case.dim, id);
        }
        REQUIRE(std::isfinite(flatten->ComputePairVectors(0, 3)));
        optimized_build->FinalizeOptimizedBuild();
        REQUIRE_FALSE(optimized_build->IsOptimizedBuildActive());
        Vector<uint8_t> supplement_code(split->SupplementCodeSize(), allocator.get());
        auto computer = split->FactoryFusedComputer(query.data());
        REQUIRE(computer != nullptr);

        for (InnerIdType id = 0; id < 3; ++id) {
            const auto* encoded_vector =
                encoded_vectors.data() + static_cast<uint64_t>(id) * split_case.dim;
            flatten->InsertVector(encoded_vector, id);
            uint32_t cluster_id = cluster_count;
            REQUIRE(split->EncodeFused(
                encoded_vector, filter_code.data(), supplement_code.data(), &cluster_id));
            REQUIRE(cluster_id < cluster_count);
            graph->SetNodeCodes(id,
                                static_cast<LabelType>(id),
                                cluster_id,
                                filter_code.data(),
                                supplement_code.data());

            float coarse_distance = 0.0F;
            float lower_bound = 0.0F;
            float filter_inner_product = 0.0F;
            REQUIRE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                          cluster_id,
                                                          filter_code.data(),
                                                          supplement_code.data(),
                                                          &coarse_distance,
                                                          &lower_bound,
                                                          &filter_inner_product,
                                                          nullptr));
            REQUIRE(std::isfinite(coarse_distance));
            REQUIRE(std::isfinite(lower_bound));
            if (split_case.filter_bits == 1) {
                REQUIRE(IsNaNBitPattern(filter_inner_product));
            } else {
                REQUIRE(std::isfinite(filter_inner_product));
            }

            float direct_full_distance = 0.0F;
            REQUIRE(split->ComputeFusedFull(computer,
                                            cluster_id,
                                            filter_code.data(),
                                            supplement_code.data(),
                                            &direct_full_distance,
                                            nullptr));
            REQUIRE(std::isfinite(direct_full_distance));

            if (split_case.filter_bits >= 2) {
                float hinted_full_distance = 0.0F;
                REQUIRE(split->ComputeFusedFullWithFilterIP(computer,
                                                            cluster_id,
                                                            filter_code.data(),
                                                            supplement_code.data(),
                                                            filter_inner_product,
                                                            &hinted_full_distance,
                                                            nullptr));
                REQUIRE(std::isfinite(hinted_full_distance));
                const float tolerance = 2e-4F * std::max({1.0F,
                                                          std::abs(direct_full_distance),
                                                          std::abs(hinted_full_distance)});
                REQUIRE(std::abs(direct_full_distance - hinted_full_distance) <= tolerance);
                REQUIRE_FALSE(
                    split->ComputeFusedFullWithFilterIP(computer,
                                                        cluster_id,
                                                        filter_code.data(),
                                                        supplement_code.data(),
                                                        std::numeric_limits<float>::quiet_NaN(),
                                                        &hinted_full_distance,
                                                        nullptr));

                RaBitQFusedTraversalQuery traversal_query;
                REQUIRE(split->GetFusedTraversalQuery(computer, &traversal_query));
                std::vector<uint8_t> invalid_filter_code(filter_code.begin(), filter_code.end());
                const float invalid_metadata = std::numeric_limits<float>::quiet_NaN();
                std::memcpy(invalid_filter_code.data() + traversal_query.one_bit_metadata_offset,
                            &invalid_metadata,
                            sizeof(invalid_metadata));
                REQUIRE_FALSE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                                    cluster_id,
                                                                    invalid_filter_code.data(),
                                                                    supplement_code.data(),
                                                                    &coarse_distance,
                                                                    &lower_bound,
                                                                    &filter_inner_product,
                                                                    nullptr));
                REQUIRE(split->ComputeFusedFull(computer,
                                                cluster_id,
                                                invalid_filter_code.data(),
                                                supplement_code.data(),
                                                &direct_full_distance,
                                                nullptr));

                auto invalid_bound_code =
                    std::vector<uint8_t>(filter_code.begin(), filter_code.end());
                const float overflowing_error_unit = std::numeric_limits<float>::max();
                std::memcpy(invalid_bound_code.data() + traversal_query.one_bit_metadata_offset +
                                2U * sizeof(float),
                            &overflowing_error_unit,
                            sizeof(overflowing_error_unit));
                coarse_distance = std::numeric_limits<float>::max();
                REQUIRE_FALSE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                                    cluster_id,
                                                                    invalid_bound_code.data(),
                                                                    supplement_code.data(),
                                                                    &coarse_distance,
                                                                    &lower_bound,
                                                                    &filter_inner_product,
                                                                    nullptr));
                REQUIRE(std::isfinite(coarse_distance));
                REQUIRE(coarse_distance < std::numeric_limits<float>::max());

                graph->SetNodeCodes(id,
                                    static_cast<LabelType>(id),
                                    cluster_id,
                                    invalid_bound_code.data(),
                                    supplement_code.data());
                SearchStatistics no_reorder_stats;
                QueryContext no_reorder_context;
                no_reorder_context.stats = &no_reorder_stats;
                no_reorder_context.enable_rabitq_reorder = false;
                const InnerIdType query_id = id;
                float filtered_distance = std::numeric_limits<float>::max();
                flatten->QueryWithDistanceFilter(&filtered_distance,
                                                 computer,
                                                 &query_id,
                                                 1,
                                                 std::numeric_limits<float>::max(),
                                                 &no_reorder_context);
                REQUIRE(filtered_distance == coarse_distance);
                REQUIRE(no_reorder_stats.rabitq_filter_count.load() == 1);
                REQUIRE(no_reorder_stats.rabitq_full_count.load() == 0);
                REQUIRE(no_reorder_stats.rabitq_filter_fallback_full_count.load() == 0);
                graph->SetNodeCodes(id,
                                    static_cast<LabelType>(id),
                                    cluster_id,
                                    filter_code.data(),
                                    supplement_code.data());
            } else {
                float hinted_full_distance = 0.0F;
                REQUIRE_FALSE(split->ComputeFusedFullWithFilterIP(computer,
                                                                  cluster_id,
                                                                  filter_code.data(),
                                                                  supplement_code.data(),
                                                                  0.0F,
                                                                  &hinted_full_distance,
                                                                  nullptr));
            }
        }

        constexpr InnerIdType alias_id = 3;
        flatten->InsertVector(encoded_vectors.data(), alias_id);
        uint32_t alias_cluster_id = cluster_count;
        REQUIRE(split->EncodeFused(
            encoded_vectors.data(), filter_code.data(), supplement_code.data(), &alias_cluster_id));
        graph->SetNodeCodes(alias_id,
                            static_cast<LabelType>(alias_id),
                            alias_cluster_id,
                            filter_code.data(),
                            supplement_code.data());

        Vector<float> decoded(split_case.dim, 0.0F, allocator.get());
        Vector<float> decoded_alias(split_case.dim, 0.0F, allocator.get());
        REQUIRE(split->DecodeFusedById(0, decoded.data()));
        REQUIRE(split->DecodeFusedById(alias_id, decoded_alias.data()));
        float source_norm_sqr = 0.0F;
        float decode_error_sqr = 0.0F;
        for (uint64_t d = 0; d < split_case.dim; ++d) {
            REQUIRE(std::isfinite(decoded[d]));
            REQUIRE(std::abs(decoded[d] - decoded_alias[d]) <= 1e-6F);
            source_norm_sqr += encoded_vectors[d] * encoded_vectors[d];
            const float error = decoded[d] - encoded_vectors[d];
            decode_error_sqr += error * error;
        }
        REQUIRE(decode_error_sqr < source_norm_sqr);
        REQUIRE_FALSE(split->DecodeFusedById(alias_id + 1, decoded.data()));
        REQUIRE_FALSE(split->DecodeFusedById(0, nullptr));
    }
}

TEST_CASE("RaBitQSplitDataCell fused zero residual metadata",
          "[ut][RaBitQSplitDataCell][fused_zero_residual]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 32;
    constexpr uint32_t cluster_count = 16;
    constexpr const char* param_template = R"(
        {{
            "codes_type": "rabitq_split",
            "io_params": {{
                "type": "memory_io"
            }},
            "quantization_params": {{
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": {},
                "rabitq_bits_per_dim_filter": {},
                "use_fht": true
            }}
        }}
    )";

    Vector<float> vectors(static_cast<uint64_t>(count) * dim, 0.0F, allocator.get());
    for (uint64_t d = 0; d < dim; ++d) {
        const float value = static_cast<float>(static_cast<int64_t>(d % 13) - 6) * 0.125F;
        for (InnerIdType row = 0; row < count; ++row) {
            vectors[static_cast<uint64_t>(row) * dim + d] = value;
        }
    }
    Vector<float> queries(2 * dim, 0.0F, allocator.get());
    std::copy_n(vectors.data(), dim, queries.data());
    for (uint64_t d = 0; d < dim; ++d) {
        queries[dim + d] = vectors[d] + static_cast<float>(static_cast<int64_t>(d % 7) - 3) * 0.05F;
    }

    for (const auto metric : {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP}) {
        for (uint32_t filter_bits = 1; filter_bits <= 4; ++filter_bits) {
            CAPTURE(static_cast<int>(metric), filter_bits);
            const uint32_t supplement_bits = filter_bits == 1 ? 3 : 8 - filter_bits;
            const uint32_t base_bits = filter_bits + supplement_bits;
            auto param_json = JsonType::Parse(fmt::format(param_template, base_bits, filter_bits));
            auto param = std::make_shared<FlattenDataCellParameter>();
            param->FromJson(param_json);

            IndexCommonParam common_param;
            common_param.allocator_ = allocator;
            common_param.dim_ = dim;
            common_param.metric_ = metric;
            auto flatten = FlattenInterface::MakeInstance(param, common_param);
            flatten->Train(vectors.data(), count);
            auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
            REQUIRE(split != nullptr);
            REQUIRE_FALSE(split->UsesLegacyHnswFusedCodec());
            split->TrainFusedCodec(vectors.data(), count, cluster_count);

            Vector<uint8_t> one_bit(split->OneBitCodeSize(), allocator.get());
            Vector<uint8_t> supplement(split->SupplementCodeSize(), allocator.get());
            uint32_t cluster_id = cluster_count;
            REQUIRE(
                split->EncodeFused(vectors.data(), one_bit.data(), supplement.data(), &cluster_id));
            REQUIRE(cluster_id < cluster_count);

            auto first_computer = split->FactoryFusedComputer(queries.data());
            RaBitQFusedTraversalQuery traversal_query;
            REQUIRE(split->GetFusedTraversalQuery(first_computer, &traversal_query));
            float filter_add = std::numeric_limits<float>::quiet_NaN();
            float filter_rescale = std::numeric_limits<float>::quiet_NaN();
            float filter_error_unit = std::numeric_limits<float>::quiet_NaN();
            const auto* metadata = one_bit.data() + traversal_query.one_bit_metadata_offset;
            std::memcpy(&filter_add, metadata, sizeof(filter_add));
            std::memcpy(&filter_rescale, metadata + sizeof(float), sizeof(filter_rescale));
            std::memcpy(
                &filter_error_unit, metadata + 2U * sizeof(float), sizeof(filter_error_unit));
            REQUIRE(std::isfinite(filter_add));
            REQUIRE(filter_rescale == 0.0F);
            REQUIRE(filter_error_unit == 0.0F);

            for (uint64_t query_id = 0; query_id < 2; ++query_id) {
                auto computer = split->FactoryFusedComputer(queries.data() + query_id * dim);
                float coarse_distance = std::numeric_limits<float>::max();
                float lower_bound = std::numeric_limits<float>::max();
                float filter_inner_product = std::numeric_limits<float>::quiet_NaN();
                REQUIRE(split->ComputeFusedOneBitWithFilterIP(computer,
                                                              cluster_id,
                                                              one_bit.data(),
                                                              supplement.data(),
                                                              &coarse_distance,
                                                              &lower_bound,
                                                              &filter_inner_product,
                                                              nullptr));
                REQUIRE(std::isfinite(coarse_distance));
                REQUIRE(coarse_distance < std::numeric_limits<float>::max());
                REQUIRE(std::isfinite(lower_bound));
                REQUIRE(lower_bound <= coarse_distance + 1e-5F);
                double expected_distance = metric == MetricType::METRIC_TYPE_IP ? 1.0 : 0.0;
                for (uint64_t d = 0; d < dim; ++d) {
                    const double base = vectors[d];
                    const double query = queries[query_id * dim + d];
                    if (metric == MetricType::METRIC_TYPE_IP) {
                        expected_distance -= base * query;
                    } else {
                        const double difference = base - query;
                        expected_distance += difference * difference;
                    }
                }
                const float expected = static_cast<float>(expected_distance);
                const float expected_tolerance = 5e-4F * std::max(1.0F, std::fabs(expected));
                REQUIRE(std::fabs(coarse_distance - expected) <= expected_tolerance);

                float full_distance = std::numeric_limits<float>::max();
                REQUIRE(split->ComputeFusedFull(computer,
                                                cluster_id,
                                                one_bit.data(),
                                                supplement.data(),
                                                &full_distance,
                                                nullptr));
                REQUIRE(std::isfinite(full_distance));
                REQUIRE(full_distance < std::numeric_limits<float>::max());
                const float tolerance =
                    1e-5F * std::max({1.0F, std::fabs(coarse_distance), std::fabs(full_distance)});
                REQUIRE(std::fabs(full_distance - coarse_distance) <= tolerance);
            }
        }
    }
}

TEST_CASE("RaBitQSplitDataCell fused model-only serialization",
          "[ut][RaBitQSplitDataCell][serialize]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr uint32_t cluster_count = 16;
    constexpr InnerIdType count = 128;
    constexpr uint64_t vectors_per_cluster = count / cluster_count;
    Vector<float> vectors(static_cast<uint64_t>(count) * dim, 0.0F, allocator.get());
    for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        for (uint64_t row = 0; row < vectors_per_cluster; ++row) {
            auto* vector =
                vectors.data() + (static_cast<uint64_t>(cluster) * vectors_per_cluster + row) * dim;
            vector[cluster] = 100.0F;
            vector[(cluster + 17) % dim] = static_cast<float>(row) * 0.01F;
        }
    }

    auto param_json = JsonType::Parse(R"({
        "codes_type": "rabitq_split",
        "io_params": {"type": "memory_io"},
        "quantization_params": {
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 1,
            "use_fht": true
        }
    })");
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 32;
    graph_param->init_max_capacity_ = count;

    auto serialize_flatten = [](const FlattenInterfacePtr& value) {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        value->Serialize(writer);
        return stream.str();
    };
    auto serialize_graph = [](const HGraphRaBitQFusedDataCellPtr& value) {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        value->Serialize(writer);
        return stream.str();
    };
    auto deserialize_flatten = [](const FlattenInterfacePtr& value, const std::string& payload) {
        std::stringstream stream(payload);
        IOStreamReader reader(stream);
        value->Deserialize(reader);
    };
    auto deserialize_graph = [](const HGraphRaBitQFusedDataCellPtr& value,
                                const std::string& payload) {
        std::stringstream stream(payload);
        IOStreamReader reader(stream);
        value->Deserialize(reader);
    };

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    split->TrainFusedCodec(vectors.data(), count, cluster_count);

    const auto legacy_payload = serialize_flatten(flatten);
    const uint64_t memory_with_split_codes = flatten->GetMemoryUsage();
    const uint64_t code_payload_size =
        static_cast<uint64_t>(count) * (split->OneBitCodeSize() + split->SupplementCodeSize());

    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
    Vector<uint8_t> one_bit(split->OneBitCodeSize(), allocator.get());
    Vector<uint8_t> supplement(split->SupplementCodeSize(), allocator.get());
    Vector<InnerIdType> empty_neighbors(allocator.get());
    for (InnerIdType id = 0; id < count; ++id) {
        uint32_t cluster_id = 0;
        REQUIRE(split->EncodeFused(vectors.data() + static_cast<uint64_t>(id) * dim,
                                   one_bit.data(),
                                   supplement.data(),
                                   &cluster_id));
        graph->SetNodeCodes(
            id, static_cast<LabelType>(id), cluster_id, one_bit.data(), supplement.data());
        graph->InsertNeighborsById(id, empty_neighbors);
    }
    graph->SetCodecModel(split->ExportFusedCodec());
    split->AttachFusedCodeStorage(graph.get());

    Vector<float> decoded(dim, 0.0F, allocator.get());
    REQUIRE(split->DecodeFusedById(0, decoded.data()));
    for (const float value : decoded) {
        REQUIRE(std::isfinite(value));
    }
    REQUIRE_FALSE(split->DecodeFusedById(count, decoded.data()));
    REQUIRE_FALSE(split->DecodeFusedById(0, nullptr));
    const auto expected_decoded = decoded;

    const auto model_payload = serialize_flatten(flatten);
    const auto graph_payload = serialize_graph(graph);
    REQUIRE(memory_with_split_codes == flatten->GetMemoryUsage() + code_payload_size);
    REQUIRE(static_cast<uint64_t>(legacy_payload.size()) + sizeof(uint32_t) ==
            static_cast<uint64_t>(model_payload.size()) + code_payload_size);

    auto make_attached_pair = [&]() {
        auto restored_flatten = FlattenInterface::MakeInstance(param, common_param);
        auto restored_split =
            std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(restored_flatten);
        REQUIRE(restored_split != nullptr);
        auto restored_graph =
            std::make_shared<HGraphRaBitQFusedDataCell>(graph_param,
                                                        restored_split->OneBitCodeSize(),
                                                        restored_split->SupplementCodeSize(),
                                                        common_param);
        restored_split->AttachFusedCodeStorage(restored_graph.get());
        return std::make_tuple(restored_flatten, restored_split, restored_graph);
    };

    auto [model_flatten, model_split, model_graph] = make_attached_pair();
    deserialize_flatten(model_flatten, model_payload);
    deserialize_graph(model_graph, graph_payload);
    model_split->ImportFusedCodec(model_graph->CodecModel());
    REQUIRE(model_split->DecodeFusedById(0, decoded.data()));
    for (uint64_t d = 0; d < dim; ++d) {
        REQUIRE(std::abs(decoded[d] - expected_decoded[d]) <= 1e-6F);
    }

    auto query = fixtures::generate_vectors(1, dim, 97);
    std::vector<InnerIdType> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    auto query_all = [&](const FlattenInterfacePtr& value) {
        auto computer = value->FactoryComputer(query.data());
        std::vector<float> distances(count);
        value->Query(distances.data(), computer, ids.data(), count);
        return distances;
    };
    const auto expected_distances = query_all(flatten);
    const auto model_distances = query_all(model_flatten);
    for (InnerIdType id = 0; id < count; ++id) {
        REQUIRE(std::abs(expected_distances[id] - model_distances[id]) <= 1e-6F);
    }

    auto [legacy_flatten, legacy_split, legacy_graph] = make_attached_pair();
    deserialize_flatten(legacy_flatten, legacy_payload);
    deserialize_graph(legacy_graph, graph_payload);
    legacy_split->ImportFusedCodec(legacy_graph->CodecModel());
    REQUIRE(legacy_split->DecodeFusedById(0, decoded.data()));
    for (uint64_t d = 0; d < dim; ++d) {
        REQUIRE(std::abs(decoded[d] - expected_decoded[d]) <= 1e-6F);
    }
    REQUIRE(legacy_split->UsesExternalFusedCodeStorage());
    using MemorySplitDataCell =
        RaBitQSplitDataCell<MetricType::METRIC_TYPE_L2SQR, MemoryIO, MemoryIO>;
    auto legacy_memory_split = std::dynamic_pointer_cast<MemorySplitDataCell>(legacy_flatten);
    REQUIRE(legacy_memory_split != nullptr);
    REQUIRE(legacy_memory_split->x_bit_cell_->GetMemoryUsage() == 0);
    REQUIRE(legacy_memory_split->supplement_cell_->GetMemoryUsage() == 0);
    REQUIRE(legacy_flatten->GetMemoryUsage() == model_flatten->GetMemoryUsage());
    const auto legacy_distances = query_all(legacy_flatten);
    for (InnerIdType id = 0; id < count; ++id) {
        REQUIRE(std::abs(expected_distances[id] - legacy_distances[id]) <= 1e-6F);
    }
}

TEST_CASE("RaBitQSplitDataCell supports MRLE transform quantizer",
          "[ut][RaBitQSplitDataCell][MRLE]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr uint64_t mrle_dim = 32;
    constexpr InnerIdType count = 24;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto query = fixtures::generate_vectors(1, dim, 71);

    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(R"({
        "codes_type": "rabitq_split",
        "io_params": { "type": "memory_io" },
        "quantization_params": {
            "type": "tq",
            "tq_chain": "mrle, rabitq",
            "mrle_dim": 32,
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 3,
            "fast_encode_rabitq": true
        }
    })"));

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    REQUIRE(flatten->GetQuantizerName() == std::string("tq"));
    flatten->Train(vectors.data(), count);
    flatten->Resize(count);

    auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
    REQUIRE(optimized_build != nullptr);
    auto finalize_pool = SafeThreadPool::FactoryDefaultThreadPool();
    finalize_pool->SetPoolSize(2);
    FlattenOptimizedBuildContext build_context{finalize_pool, 2};
    REQUIRE(optimized_build->BeginOptimizedBuild(build_context));
    flatten->BatchInsertVector(vectors.data(), count);

    std::vector<InnerIdType> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> build_dists(count);
    auto computer = flatten->FactoryComputer(query.data());
    flatten->Query(build_dists.data(), computer, ids.data(), count);
    optimized_build->FinalizeOptimizedBuild();

    std::vector<float> split_dists(count);
    flatten->Query(split_dists.data(), computer, ids.data(), count);
    using QuantizerT = TransformQuantizer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                          MetricType::METRIC_TYPE_L2SQR>;
    auto* transform_computer = static_cast<Computer<QuantizerT>*>(computer.get());
    for (InnerIdType id = 0; id < count; ++id) {
        bool need_release = false;
        const auto* full_code = flatten->GetCodesById(id, need_release);
        float merged_dist = 0.0F;
        transform_computer->ComputeDist(full_code, &merged_dist);
        if (need_release) {
            flatten->Release(full_code);
        }
        REQUIRE(std::abs(build_dists[id] - split_dists[id]) <= 1e-4F);
        REQUIRE(std::abs(split_dists[id] - merged_dist) <= 1e-5F);
    }

    auto invalid_json = param->ToJson();
    invalid_json["quantization_params"]["tq_chain"].SetString("pca, rabitq");
    invalid_json["quantization_params"]["pca_dim"].SetInt(mrle_dim);
    auto invalid_param = std::make_shared<FlattenDataCellParameter>();
    invalid_param->FromJson(invalid_json);
    REQUIRE_THROWS(FlattenInterface::MakeInstance(invalid_param, common_param));
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

TEST_CASE("RaBitQSplitDataCell optimized scalar-code build",
          "[ut][RaBitQSplitDataCell][optimized_build]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 24;
    auto vectors = fixtures::generate_vectors(count, dim);
    auto query = fixtures::generate_vectors(1, dim, 29);
    auto metric = GENERATE(
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE);

    auto bits = GENERATE(std::make_tuple(8U, 3U), std::make_tuple(7U, 2U), std::make_tuple(5U, 4U));
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
                "rabitq_bits_per_dim_base": 8,
                "rabitq_bits_per_dim_filter": 3,
                "fast_encode_rabitq": true
            }
        }
        )";

    auto param_json = JsonType::Parse(param_str);
    param_json["quantization_params"]["rabitq_bits_per_dim_base"].SetInt(std::get<0>(bits));
    param_json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(std::get<1>(bits));
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = metric;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
    REQUIRE(optimized_build != nullptr);
    flatten->Train(vectors.data(), count);
    std::vector<uint8_t> expected_codes(flatten->code_size_ * count);
    for (InnerIdType id = 0; id < count; ++id) {
        REQUIRE(flatten->Encode(vectors.data() + id * dim,
                                expected_codes.data() + id * flatten->code_size_));
    }

    const uint64_t memory_before_build = flatten->GetMemoryUsage();
    auto finalize_pool = SafeThreadPool::FactoryDefaultThreadPool();
    finalize_pool->SetPoolSize(4);
    FlattenOptimizedBuildContext build_context{finalize_pool, 4};
    REQUIRE(optimized_build->BeginOptimizedBuild(build_context));
    REQUIRE(optimized_build->IsOptimizedBuildActive());
    flatten->ShrinkToFit(0);
    REQUIRE_THROWS(flatten->InsertVector(vectors.data()));
    REQUIRE(flatten->TotalCount() == 0);
    flatten->Resize(count);
    flatten->BatchInsertVector(vectors.data(), count);
    REQUIRE(std::isfinite(flatten->ComputePairVectors(0, 1)));
    auto build_computer = optimized_build->FactoryComputerForBuild(vectors.data(), 0);
    std::vector<InnerIdType> build_ids(count);
    std::iota(build_ids.begin(), build_ids.end(), 0);
    std::vector<float> pair_dists(count);
    flatten->Query(pair_dists.data(), build_computer, build_ids.data(), count);
    for (InnerIdType id = 0; id < count; ++id) {
        REQUIRE(std::abs(pair_dists[id] - flatten->ComputePairVectors(0, id)) <= 1e-6F);
    }
    flatten->Resize(count + 8);
    flatten->Move(1, 2);
    std::memcpy(expected_codes.data() + 2 * flatten->code_size_,
                expected_codes.data() + flatten->code_size_,
                flatten->code_size_);
    REQUIRE(std::abs(flatten->ComputePairVectors(0, 2) - flatten->ComputePairVectors(0, 1)) <=
            1e-6F);
    flatten->ShrinkToFit(count);
    flatten->Resize(count + 8);
    REQUIRE(flatten->GetMemoryUsage() > memory_before_build);
    REQUIRE_THROWS(flatten->CalcSerializeSize());

    std::vector<InnerIdType> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> build_dists(count);
    auto computer = flatten->FactoryComputer(query.data());
    flatten->Query(build_dists.data(), computer, ids.data(), count);
    for (InnerIdType id = 0; id < count; ++id) {
        bool need_release = false;
        const auto* full_code = flatten->GetCodesById(id, need_release);
        REQUIRE(need_release);
        REQUIRE(std::memcmp(full_code,
                            expected_codes.data() + id * flatten->code_size_,
                            flatten->code_size_) == 0);
        flatten->Release(full_code);
    }

    const float build_pair_distance = flatten->ComputePairVectors(0, 2);

    optimized_build->FinalizeOptimizedBuild();
    REQUIRE_FALSE(optimized_build->IsOptimizedBuildActive());
    REQUIRE(std::abs(flatten->ComputePairVectors(0, 2) - build_pair_distance) <= 1e-5F);
    std::vector<float> split_dists(count);
    flatten->Query(split_dists.data(), computer, ids.data(), count);
    for (InnerIdType id = 0; id < count; ++id) {
        std::vector<uint8_t> merged_code(flatten->code_size_);
        REQUIRE(flatten->GetCodesById(id, merged_code.data()));
        REQUIRE(std::memcmp(merged_code.data(),
                            expected_codes.data() + id * flatten->code_size_,
                            flatten->code_size_) == 0);
        REQUIRE(std::abs(split_dists[id] - build_dists[id]) <= 1e-4F);
    }

    const uint64_t memory_after_finalize = flatten->GetMemoryUsage();
    REQUIRE(memory_after_finalize >=
            static_cast<uint64_t>(flatten->max_capacity_) * flatten->code_size_);
    REQUIRE(optimized_build->BeginOptimizedBuild(build_context));
    REQUIRE(flatten->GetMemoryUsage() > memory_after_finalize);
    optimized_build->AbortOptimizedBuild();
    REQUIRE_FALSE(optimized_build->IsOptimizedBuildActive());
    REQUIRE(flatten->GetMemoryUsage() == memory_after_finalize);
}

TEST_CASE("RaBitQSplitDataCell waits submitted finalize tasks after enqueue failure",
          "[ut][RaBitQSplitDataCell][optimized_build]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 16;
    auto vectors = fixtures::generate_vectors(count, dim);
    constexpr const char* param_str = R"(
        {
            "codes_type": "rabitq_split",
            "io_params": { "type": "memory_io" },
            "quantization_params": {
                "type": "rabitq",
                "rabitq_version": "split",
                "rabitq_bits_per_dim_query": 32,
                "rabitq_bits_per_dim_base": 8,
                "rabitq_bits_per_dim_filter": 3,
                "fast_encode_rabitq": true
            }
        }
        )";

    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
    REQUIRE(optimized_build != nullptr);
    flatten->Train(vectors.data(), count);
    auto rejecting_pool = std::make_shared<RejectSecondThreadPool>();
    auto safe_pool = std::make_shared<SafeThreadPool>(rejecting_pool);
    FlattenOptimizedBuildContext build_context{safe_pool, 4};
    REQUIRE(optimized_build->BeginOptimizedBuild(build_context));
    flatten->Resize(count);
    flatten->BatchInsertVector(vectors.data(), count);

    const auto begin = std::chrono::steady_clock::now();
    REQUIRE_THROWS(optimized_build->FinalizeOptimizedBuild());
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    REQUIRE(elapsed >= std::chrono::milliseconds(75));
    REQUIRE(rejecting_pool->TaskStarted());
    rejecting_pool->WaitUntilEmpty();

    REQUIRE(optimized_build->IsOptimizedBuildActive());
    optimized_build->AbortOptimizedBuild();
    REQUIRE_FALSE(optimized_build->IsOptimizedBuildActive());
}

TEST_CASE("RaBitQSplitDataCell finalizes mmap storage serially",
          "[ut][RaBitQSplitDataCell][optimized_build]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 64;
    constexpr InnerIdType count = 16;
    auto vectors = fixtures::generate_vectors(count, dim);
    const std::string tmp_prefix =
        "/tmp/vsag_rabitq_split_mmap_finalize_ut_" + std::to_string(::getpid());
    struct TempFileCleanup {
        explicit TempFileCleanup(std::string prefix) : prefix_(std::move(prefix)) {
            cleanup();
        }
        ~TempFileCleanup() {
            cleanup();
        }
        void
        cleanup() const {
            std::remove((prefix_ + "_onebit").c_str());
            std::remove((prefix_ + "_supplement").c_str());
        }
        std::string prefix_;
    } cleanup(tmp_prefix);

    const auto param_str = fmt::format(R"({{
        "codes_type": "rabitq_split",
        "io_params": {{ "type": "mmap_io", "file_path": "{}" }},
        "quantization_params": {{
            "type": "rabitq",
            "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32,
            "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 3,
            "fast_encode_rabitq": true
        }}
    }})",
                                       tmp_prefix);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(param_str));
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(flatten);
    REQUIRE(optimized_build != nullptr);
    flatten->Train(vectors.data(), count);
    auto rejecting_pool = std::make_shared<RejectSecondThreadPool>();
    auto safe_pool = std::make_shared<SafeThreadPool>(rejecting_pool);
    FlattenOptimizedBuildContext build_context{safe_pool, 4};
    REQUIRE(optimized_build->BeginOptimizedBuild(build_context));
    flatten->Resize(count);
    flatten->BatchInsertVector(vectors.data(), count);

    REQUIRE_NOTHROW(optimized_build->FinalizeOptimizedBuild());
    REQUIRE_FALSE(optimized_build->IsOptimizedBuildActive());
    REQUIRE_FALSE(rejecting_pool->TaskStarted());
    REQUIRE(std::isfinite(flatten->ComputePairVectors(0, 1)));
}
