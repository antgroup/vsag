// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rabitq_split_datacell.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "flatten_datacell_parameter.h"
#include "flatten_interface.h"
#include "hgraph_rabitq_fused_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io_parameter.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("RaBitQ split interface queries with filter IP hints", "[ut][RaBitQSplitDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr InnerIdType train_count = 64;
    constexpr InnerIdType query_count = 3;
    constexpr uint64_t dim = 64;
    constexpr uint32_t cluster_count = 16;

    auto param_json = JsonType::Parse(R"(
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
                "rabitq_bits_per_dim_filter": 2,
                "use_fht": true
            }
        }
    )");
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);

    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto vectors = fixtures::generate_vectors(train_count, dim, false, 31);
    auto query = fixtures::generate_vectors(1, dim, false, 71);
    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), train_count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    split->TrainFusedCodec(vectors.data(), train_count, cluster_count);

    auto graph_param = std::make_shared<GraphDataCellParameter>();
    graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
    graph_param->max_degree_ = 8;
    graph_param->init_max_capacity_ = query_count;
    auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
        graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
    split->AttachFusedCodeStorage(graph.get());
    flatten->Resize(query_count);

    Vector<uint8_t> filter_code(split->OneBitCodeSize(), allocator.get());
    Vector<uint8_t> supplement_code(split->SupplementCodeSize(), allocator.get());
    std::array<InnerIdType, query_count> ids{};
    for (InnerIdType id = 0; id < query_count; ++id) {
        ids[id] = id;
        const auto* vector = vectors.data() + static_cast<uint64_t>(id) * dim;
        flatten->InsertVector(vector, id);
        uint32_t cluster_id = cluster_count;
        REQUIRE(
            split->EncodeFused(vector, filter_code.data(), supplement_code.data(), &cluster_id));
        graph->SetNodeCodes(
            id, static_cast<LabelType>(id), cluster_id, filter_code.data(), supplement_code.data());
    }

    auto computer = split->FactoryFusedComputer(query.data());
    REQUIRE(computer != nullptr);
    Vector<float> coarse_distances(query_count, 0.0F, allocator.get());
    Vector<float> lower_bounds(query_count, 0.0F, allocator.get());
    Vector<float> filter_inner_products(query_count, 0.0F, allocator.get());
    SearchStatistics filter_statistics;
    QueryContext filter_context;
    filter_context.stats = &filter_statistics;
    split->QueryWithDistanceLowerBoundAndFilterIP(coarse_distances.data(),
                                                  lower_bounds.data(),
                                                  filter_inner_products.data(),
                                                  computer,
                                                  ids.data(),
                                                  query_count,
                                                  &filter_context);
    for (InnerIdType i = 0; i < query_count; ++i) {
        REQUIRE(std::isfinite(coarse_distances[i]));
        REQUIRE(std::isfinite(lower_bounds[i]));
        REQUIRE(std::isfinite(filter_inner_products[i]));
    }
    REQUIRE(filter_statistics.rabitq_filter_count.load() == query_count);

    Vector<float> expected_distances(query_count, 0.0F, allocator.get());
    flatten->Query(expected_distances.data(), computer, ids.data(), query_count, nullptr);

    Vector<float> hinted_distances(query_count, 0.0F, allocator.get());
    SearchStatistics hint_statistics;
    QueryContext hint_context;
    hint_context.stats = &hint_statistics;
    split->QueryWithFilterIPHint(hinted_distances.data(),
                                 filter_inner_products.data(),
                                 computer,
                                 ids.data(),
                                 query_count,
                                 &hint_context);
    for (InnerIdType i = 0; i < query_count; ++i) {
        const float tolerance =
            2e-4F *
            std::max({1.0F, std::abs(expected_distances[i]), std::abs(hinted_distances[i])});
        REQUIRE(std::abs(expected_distances[i] - hinted_distances[i]) <= tolerance);
    }
    REQUIRE(hint_statistics.rabitq_full_count.load() == query_count);
    REQUIRE(hint_statistics.rabitq_reorder_hint_full_count.load() == query_count);

    filter_inner_products[1] = std::numeric_limits<float>::quiet_NaN();
    Vector<float> fallback_distances(query_count, 0.0F, allocator.get());
    SearchStatistics fallback_statistics;
    QueryContext fallback_context;
    fallback_context.stats = &fallback_statistics;
    split->QueryWithFilterIPHint(fallback_distances.data(),
                                 filter_inner_products.data(),
                                 computer,
                                 ids.data(),
                                 query_count,
                                 &fallback_context);
    for (InnerIdType i = 0; i < query_count; ++i) {
        const float tolerance =
            2e-4F *
            std::max({1.0F, std::abs(expected_distances[i]), std::abs(fallback_distances[i])});
        REQUIRE(std::abs(expected_distances[i] - fallback_distances[i]) <= tolerance);
    }
    REQUIRE(fallback_statistics.rabitq_full_count.load() == query_count);
    REQUIRE(fallback_statistics.rabitq_reorder_hint_full_count.load() == query_count - 1);
}

TEST_CASE("RaBitQ fused pair distances match reconstructed vectors",
          "[ut][RaBitQSplitDataCell][FusedPairDistance]") {
    const auto metric = GENERATE(MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP);
    const auto use_fht = GENERATE(true, false);
    // 64/960 exercise code-domain kernels; 96 also checks the reconstruction fallback.
    const uint64_t dim = GENERATE(64, 96, 960);
    CAPTURE(metric, use_fht, dim);
    constexpr InnerIdType count = 64;
    constexpr uint32_t cluster_count = 16;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto param_json = JsonType::Parse(R"(
        {
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
        }
    )");
    param_json["quantization_params"]["use_fht"].SetBool(use_fht);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = metric;
    auto vectors = fixtures::generate_vectors(count, dim, false, 31);
    auto other_vectors = fixtures::generate_vectors(count, dim, false, 71);
    for (uint64_t i = 0; i < other_vectors.size(); ++i) {
        other_vectors[i] += 0.25F;
    }
    std::vector<std::shared_ptr<HGraphRaBitQFusedDataCell>> storage;
    auto check_pairs = [&](const FlattenInterfacePtr& flatten, const float* data) {
        auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
        REQUIRE(split != nullptr);
        auto graph_param = std::make_shared<GraphDataCellParameter>();
        graph_param->io_parameter_ = std::make_shared<MemoryIOParameter>();
        graph_param->max_degree_ = 8;
        graph_param->init_max_capacity_ = count;
        auto graph = std::make_shared<HGraphRaBitQFusedDataCell>(
            graph_param, split->OneBitCodeSize(), split->SupplementCodeSize(), common_param);
        storage.push_back(graph);
        split->AttachFusedCodeStorage(graph.get());
        flatten->Resize(count);
        Vector<uint8_t> filter_code(split->OneBitCodeSize(), allocator.get());
        Vector<uint8_t> supplement_code(split->SupplementCodeSize(), allocator.get());
        Vector<float> decoded(count * dim, 0.0F, allocator.get());
        std::array<uint32_t, count> clusters{};
        for (InnerIdType id = 0; id < count; ++id) {
            const auto* vector = data + static_cast<uint64_t>(id) * dim;
            flatten->InsertVector(vector, id);
            REQUIRE(split->EncodeFused(
                vector, filter_code.data(), supplement_code.data(), &clusters[id]));
            graph->SetNodeCodes(id, id, clusters[id], filter_code.data(), supplement_code.data());
            REQUIRE(split->DecodeFusedById(id, decoded.data() + static_cast<uint64_t>(id) * dim));
        }
        uint64_t same_cluster_pairs = 0;
        uint64_t cross_cluster_pairs = 0;
        for (InnerIdType i = 0; i < count; ++i) {
            for (InnerIdType j = i; j < count; ++j) {
                CAPTURE(i, j, clusters[i], clusters[j]);
                double expected = metric == MetricType::METRIC_TYPE_IP ? 1.0 : 0.0;
                for (uint64_t d = 0; d < dim; ++d) {
                    const double x = decoded[static_cast<uint64_t>(i) * dim + d];
                    const double y = decoded[static_cast<uint64_t>(j) * dim + d];
                    expected += metric == MetricType::METRIC_TYPE_IP ? -x * y : (x - y) * (x - y);
                }
                const double tolerance = 2e-4 * std::max(1.0, std::abs(expected));
                REQUIRE(std::abs(flatten->ComputePairVectors(i, j) - expected) <= tolerance);
                REQUIRE(std::abs(flatten->ComputePairVectors(j, i) - expected) <= tolerance);
                if (i != j) {
                    same_cluster_pairs += clusters[i] == clusters[j];
                    cross_cluster_pairs += clusters[i] != clusters[j];
                }
            }
        }
        REQUIRE(same_cluster_pairs > 0);
        REQUIRE(cross_cluster_pairs > 0);
    };

    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    split->TrainFusedCodec(vectors.data(), count, cluster_count);
    check_pairs(flatten, vectors.data());
    const auto codec = split->ExportFusedCodec();

    // Restore over a different codec, so a stale derived centroid cache cannot pass.
    auto restored = FlattenInterface::MakeInstance(param, common_param);
    flatten->ExportModel(restored);
    auto restored_split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(restored);
    REQUIRE(restored_split != nullptr);
    restored_split->TrainFusedCodec(other_vectors.data(), count, cluster_count);
    REQUIRE(restored_split->ExportFusedCodec() != codec);
    restored_split->ImportFusedCodec(codec);
    REQUIRE(restored_split->ExportFusedCodec() == codec);
    check_pairs(restored, vectors.data());

    // Retraining must refresh the same cache too.
    split->TrainFusedCodec(other_vectors.data(), count, cluster_count);
    check_pairs(flatten, other_vectors.data());
}

}  // namespace vsag
