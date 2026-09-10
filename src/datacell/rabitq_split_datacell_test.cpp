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
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    REQUIRE(split != nullptr);
    REQUIRE_THROWS_AS(split->TrainFusedCodec(vectors.data(), train_count, cluster_count),
                      VsagException);
    REQUIRE(split->FusedClusterCount() == 0);
    if (GENERATE(false, true)) {
        REQUIRE_NOTHROW(split->TrainFusedTransform());
    } else {
        flatten->Train(vectors.data(), train_count);
    }
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

TEST_CASE("Fused dynamic centers share a lazy query cache and serialize the model",
          "[ut][RaBitQSplitDataCell][fused_full]") {
    const auto metric = GENERATE(MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP);
    const auto filter_bits = GENERATE(1, 2, 3, 4);
    const auto k = GENERATE(1U, 7U, 33U);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t dim = 32;
    constexpr uint64_t count = 70;
    auto param_json = JsonType::Parse(R"({
        "codes_type": "rabitq_split", "io_params": {"type": "memory_io"},
        "quantization_params": {"type": "rabitq", "rabitq_version": "split",
            "rabitq_bits_per_dim_query": 32, "rabitq_bits_per_dim_base": 8,
            "rabitq_bits_per_dim_filter": 2, "use_fht": true}
    })");
    param_json["quantization_params"]["rabitq_bits_per_dim_filter"].SetInt(filter_bits);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(param_json);
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = dim;
    common_param.metric_ = metric;
    auto vectors = fixtures::generate_vectors(count, dim, false, 71);
    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    flatten->Train(vectors.data(), count);
    auto split = std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(flatten);
    split->TrainFusedCodec(vectors.data(), count, k, 2);
    REQUIRE(split->FusedClusterCount() == k);
    const auto payload = split->ExportFusedCodec();
    REQUIRE(payload.size() == FusedCodecSize(dim, k));
    auto old_version = payload;
    const uint32_t version = 1;
    std::memcpy(old_version.data(), &version, sizeof(version));
    REQUIRE_THROWS(split->ImportFusedCodec(old_version));
    REQUIRE_THROWS(split->ImportFusedCodec(payload + "x"));
    REQUIRE_THROWS(split->ImportFusedCodec(payload.substr(0, payload.size() - 1)));
    auto bad_count = payload;
    const uint32_t oversized_count = std::numeric_limits<uint32_t>::max();
    std::memcpy(bad_count.data() + sizeof(uint32_t), &oversized_count, sizeof(oversized_count));
    REQUIRE_THROWS(split->ImportFusedCodec(bad_count));
    auto bad_center = payload;
    const float invalid = std::numeric_limits<float>::infinity();
    std::memcpy(bad_center.data() + 16, &invalid, sizeof(invalid));
    REQUIRE_THROWS(split->ImportFusedCodec(bad_center));
    REQUIRE(split->ExportFusedCodec() == payload);
    auto stale_norms = payload;
    const double invalid_norm = std::numeric_limits<double>::infinity();
    const uint64_t norms_offset = 16 + 2 * uint64_t{k} * dim * sizeof(float);
    for (uint32_t id = 0; id < k; ++id) {
        std::memcpy(stale_norms.data() + norms_offset + id * sizeof(double),
                    &invalid_norm,
                    sizeof(invalid_norm));
    }
    REQUIRE_NOTHROW(split->ImportFusedCodec(payload));
    const auto canonical_payload = split->ExportFusedCodec();
    REQUIRE_NOTHROW(split->ImportFusedCodec(stale_norms));
    // Recomputed norms may differ in the last bit from training under FP contraction/reduction.
    // Compare two imports of identical centers, not a reduction performed at another call site.
    REQUIRE(static_cast<bool>(split->ExportFusedCodec() == canonical_payload));

    auto computer = split->FactoryFusedComputer(vectors.data());
    RaBitQFusedTraversalQuery traversal;
    REQUIRE(split->GetFusedTraversalQuery(computer, &traversal));
    REQUIRE(traversal.cluster_cache->computed_count == 0);
    // A high-numbered center must work even when no visited node has populated it yet.
    traversal.EnsureCluster(k - 1);
    REQUIRE(traversal.cluster_cache->computed_count == 1);
    traversal.EnsureCluster(k - 1);
    REQUIRE(traversal.cluster_cache->computed_count == 1);
    uint64_t offset = 16 + k * dim * sizeof(float);
    std::vector<float> centers(k * dim);
    std::memcpy(centers.data(), payload.data() + offset, centers.size() * sizeof(float));
    for (uint32_t id = 0; id < k; ++id) {
        traversal.EnsureCluster(id);
        double squared = 0.0, dot = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            const double q = traversal.transformed_query[d];
            const double c = centers[uint64_t{id} * dim + d];
            squared += (q - c) * (q - c);
            dot += q * c;
        }
        const float expected =
            static_cast<float>(metric == MetricType::METRIC_TYPE_IP ? -dot : squared);
        REQUIRE(std::abs(traversal.cluster_g_add[id] - expected) < 1e-4F);
        REQUIRE(std::abs(traversal.cluster_g_error[id] - std::sqrt(squared)) < 1e-4);
    }
    REQUIRE(traversal.cluster_cache->computed_count == k);
    auto other = split->FactoryFusedComputer(vectors.data() + dim);
    RaBitQFusedTraversalQuery other_traversal;
    REQUIRE(split->GetFusedTraversalQuery(other, &other_traversal));
    REQUIRE(other_traversal.cluster_cache != traversal.cluster_cache);
    REQUIRE(other_traversal.cluster_cache->computed_count == 0);

    Vector<uint8_t> filter(split->OneBitCodeSize(), allocator.get());
    Vector<uint8_t> supplement(split->SupplementCodeSize(), allocator.get());
    uint32_t id = 0;
    REQUIRE(split->EncodeFused(vectors.data() + dim, filter.data(), supplement.data(), &id));
    float before = 0.0F, after = 0.0F;
    REQUIRE(
        split->ComputeFusedFull(computer, id, filter.data(), supplement.data(), &before, nullptr));
    // Query caches borrow the model, so release them before replacing it.
    computer.reset();
    other.reset();
    split->ImportFusedCodec(payload);
    computer = split->FactoryFusedComputer(vectors.data());
    REQUIRE(
        split->ComputeFusedFull(computer, id, filter.data(), supplement.data(), &after, nullptr));
    REQUIRE(before == after);
}

TEST_CASE("Fused query center terms avoid cancellation", "[ut][RaBitQSplitDataCell][fused_full]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const float query[] = {1000000.0F, 1000000.125F};
    const float center[] = {1000000.0F, 1000000.0F};
    const double norm = 2000000000000.0;
    RaBitQFusedQueryCache cache(allocator.get());
    cache.Initialize(query, center, &norm, 2, 1, MetricType::METRIC_TYPE_L2SQR);
    cache.Ensure(0);
    REQUIRE(cache.add[0] == 0.015625F);
    REQUIRE(cache.error[0] == 0.125F);
}

TEST_CASE("Fused query center terms handle zero and tiny vectors",
          "[ut][RaBitQSplitDataCell][fused_full]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    const float magnitude = GENERATE(0.0F, std::numeric_limits<float>::min());
    const float query[] = {GENERATE(-1.0F, 1.0F) * magnitude};
    const float center[] = {magnitude};
    const double norm = static_cast<double>(magnitude) * magnitude;
    const auto metric = GENERATE(MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP);
    RaBitQFusedQueryCache cache(allocator.get());
    cache.Initialize(query, center, &norm, 1, 1, metric);
    cache.Ensure(0);
    // The squared term may round to zero in FP32, but its FP64 square root must not be lost.
    REQUIRE(cache.add[0] == 0.0F);
    REQUIRE(cache.error[0] ==
            static_cast<float>(std::abs(static_cast<double>(query[0]) - center[0])));
    cache.Ensure(0);
    REQUIRE(cache.computed_count == 1);
}

}  // namespace vsag
