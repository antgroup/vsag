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

#include "mci_searcher.h"

#include <catch2/catch_approx.hpp>

#include "impl/label_table/label_table.h"
#include "impl/reorder/flatten_reorder.h"
#include "searcher_test.h"

using namespace vsag;

TEST_CASE("MCISearcher uses the HGraph one-bit backend for seeds and clique expansion",
          "[ut][MCISearcher][mci][mci_one_bit]") {
    const auto metric = GENERATE(
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE);
    const bool complete_scan = GENERATE(false, true);
    const bool batch = GENERATE(false, true);
    const bool one_bit = GENERATE(false, true);
    CAPTURE(metric, complete_scan, batch, one_bit);
    constexpr InnerIdType count = 79;
    constexpr uint64_t dim = 64;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = dim;
    common.metric_ = metric;
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(R"({"codes_type":"rabitq_split",
        "io_params":{"type":"memory_io"},"quantization_params":{"type":"rabitq",
        "rabitq_bits_per_dim_base":8,"rabitq_bits_per_dim_filter":1}})"));
    auto flatten = FlattenInterface::MakeInstance(param, common);
    auto vectors = fixtures::generate_vectors(count, dim, 23);
    auto queries = fixtures::generate_vectors(1, dim, 17);
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);
    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> offsets({0, count}, allocator.get());
    Vector<InnerIdType> members(count, allocator.get());
    std::iota(members.begin(), members.end(), 0);
    Vector<InnerIdType> inverse_offsets(count + 1, allocator.get());
    std::iota(inverse_offsets.begin(), inverse_offsets.end(), 0);
    Vector<InnerIdType> inverse(count, 0, allocator.get());
    cliques->Assign(std::move(offsets),
                    std::move(members),
                    std::move(inverse_offsets),
                    std::move(inverse),
                    count);
    cliques->MarkAvailable(count);
    Vector<InnerIdType> seeds(allocator.get());
    for (InnerIdType id = 0; id < (complete_scan ? count : 1); ++id) {
        seeds.push_back(id);
    }
    const std::vector<uint8_t> bitmap(count, 1);
    MCISearcherParam params;
    params.seed_count = seeds.size();
    params.seed_inner_ids = &seeds;
    params.enumerated_valid_count = complete_scan ? count : 0;
    params.valid_bitmap = bitmap.data();
    params.valid_bitmap_size = bitmap.size();
    params.batch_distance = batch;
    params.use_one_bit_distance = one_bit;
    InnerSearchParam search;
    search.topk = count;
    search.ef = count;
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator.get(), .stats = &stats};
    auto result =
        MCISearcher(common).Search(cliques, flatten, queries.data(), search, params, &ctx);
    REQUIRE(result->Size() == count);
    REQUIRE(stats.dist_cmp.load() == count);
    REQUIRE(stats.io_cnt.load() == 0);
    REQUIRE(stats.rabitq_filter_count.load() == (one_bit ? count : 0));
    REQUIRE(stats.hops.load() == (complete_scan ? 0 : 1));

    std::vector<InnerIdType> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<float> expected(count);
    auto computer = flatten->FactoryComputer(queries.data());
    if (one_bit) {
        flatten->QueryWithDistanceLowerBound(expected.data(), nullptr, computer, ids.data(), count);
    } else {
        flatten->Query(expected.data(), computer, ids.data(), count);
    }
    while (not result->Empty()) {
        const auto record = result->Top();
        REQUIRE(record.first == Catch::Approx(expected[record.second]));
        result->Pop();
    }
    if (one_bit) {
        // Loose bounds force recovery beyond a one-entry approximate heap. Validate the
        // existing reorder against full scores over exactly the recovered candidate union.
        search.ef = 1;
        search.topk = 1;
        DistanceRecordVector extra(allocator.get());
        params.one_bit_lower_bound_candidates = &extra;
        SearchStatistics recovery_stats;
        QueryContext recovery_ctx{.alloc = allocator.get(), .stats = &recovery_stats};
        recovery_ctx.rabitq_error_rate = 1000.0F;
        auto candidates = MCISearcher(common).Search(
            cliques, flatten, queries.data(), search, params, &recovery_ctx);
        REQUIRE(candidates->Size() == 1);
        REQUIRE_FALSE(extra.empty());
        std::vector<uint8_t> recovered(count, 0);
        recovered[candidates->Top().second] = 1;
        for (const auto& record : extra) {
            REQUIRE(record.second < count);
            recovered[record.second] = 1;
        }
        flatten->Query(expected.data(), computer, ids.data(), count);
        auto minimum = std::numeric_limits<float>::max();
        for (InnerIdType id = 0; id < count; ++id) {
            if (recovered[id] != 0) {
                minimum = std::min(minimum, expected[id]);
            }
        }
        auto refined = FlattenReorder(flatten, allocator.get())
                           .Reorder(candidates, queries.data(), 1, recovery_ctx, nullptr, &extra);
        REQUIRE(refined->Size() == 1);
        REQUIRE(recovered[refined->Top().second] != 0);
        REQUIRE(refined->Top().first == Catch::Approx(minimum));
        REQUIRE(recovery_stats.reorder_lower_bound_probe_count.load() == 1);
        REQUIRE(recovery_stats.reorder_distance_count.load() > 1);
        REQUIRE(recovery_stats.io_cnt.load() == 0);
    }
}

TEST_CASE("MCISearcher traverses base extras delta and tombstones before and after flush",
          "[ut][MCISearcher][mci][flush]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto quantizer =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer, io, common);
    std::vector<float> vectors{0, 1, 2, 3, 4};
    std::vector<InnerIdType> ids{0, 1, 2, 3, 4};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> offsets({0, 2, 4}, allocator.get());
    Vector<InnerIdType> members({0, 1, 1, 2}, allocator.get());
    Vector<InnerIdType> inverse_offsets({0, 1, 3, 4}, allocator.get());
    Vector<InnerIdType> inverse({0, 0, 1, 1}, allocator.get());
    cliques->Assign(
        std::move(offsets), std::move(members), std::move(inverse_offsets), std::move(inverse), 3);
    REQUIRE(cliques->AppendNodeToClique(4, 0, 5, 3));
    Vector<InnerIdType> delta({2, 3, 4}, allocator.get());
    cliques->AppendNewClique(delta, 5);
    Vector<InnerIdType> removed(allocator.get());
    removed.push_back(1);
    auto snapshot = cliques->PrepareDelete(removed, 2, 3);
    cliques->CommitDelete(removed, snapshot.retired_clique_ids, 5);
    cliques->MarkAvailable(5);

    const bool flush = GENERATE(false, true);
    if (flush) {
        cliques->Flush(5);
    }
    const bool use_raw = GENERATE(false, true);
    const bool bitmap = GENERATE(false, true);
    const bool pending_delete = GENERATE(false, true);
    bool used_raw = false;
    Vector<InnerIdType> seeds(allocator.get());
    seeds.push_back(0);
    LabelTable labels(allocator.get());
    MCISearcherParam params;
    params.packed_filter = GENERATE(false, true);
    params.batch_distance = GENERATE(false, true);
    params.seed_count = 1;
    params.seed_inner_ids = &seeds;
    params.precise_vectors = use_raw ? vectors.data() : nullptr;
    params.dim = 1;
    params.precise_vector_stride = 1;
    params.metric = common.metric_;
    params.used_precise_float_csr = &used_raw;
    std::vector<uint8_t> valid_bitmap(5, 7);  // Nonzero bytes are valid, including the tombstone.
    bool used_bitmap = false;
    params.valid_bitmap = bitmap ? valid_bitmap.data() : nullptr;
    params.valid_bitmap_size = valid_bitmap.size();
    params.used_bitmap_fast_path = &used_bitmap;
    for (InnerIdType id = 0; id < 5; ++id) {
        labels.Insert(id, id);
    }
    // Model a label deletion that is visible before its MCI snapshot is updated.
    if (pending_delete) {
        REQUIRE(labels.MarkRemove(3) == 1);
    }
    params.deleted_ids_filter = labels.GetDeletedIdsReadView();
    InnerSearchParam search;
    search.ef = 8;
    search.topk = 5;
    search.is_inner_id_allowed = params.deleted_ids_filter;
    const float query = 1.0F;
    auto result = MCISearcher(common).Search(cliques, flatten, &query, search, params, nullptr);
    REQUIRE(used_raw == use_raw);
    REQUIRE(used_bitmap == bitmap);
    REQUIRE(result->Size() == (pending_delete ? 3 : 4));
    std::vector<InnerIdType> found;
    while (not result->Empty()) {
        const auto id = result->Top().second;
        REQUIRE(result->Top().first == (vectors[id] - query) * (vectors[id] - query));
        found.push_back(id);
        result->Pop();
    }
    std::sort(found.begin(), found.end());
    const std::vector<InnerIdType> expected =
        pending_delete ? std::vector<InnerIdType>{0, 2, 4} : std::vector<InnerIdType>{0, 2, 3, 4};
    REQUIRE(found == expected);
}
