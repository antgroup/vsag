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

#include "impl/label_table/label_table.h"
#include "searcher_test.h"
#include "utils/search_lock.h"

using namespace vsag;

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
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
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

TEST_CASE("MCISearcher threshold results backfill past non-finite traversal seeds",
          "[ut][MCISearcher][threshold][nonfinite]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_IP;

    constexpr const char* param_template = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_template, "fp32")));
    auto io_param = IOParameter::GetIOParameterByJson(
        JsonType::Parse(fmt::format(param_template, "memory_io")));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_IP>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, common);
    flatten->SetQuantizer(
        std::make_shared<FP32Quantizer<MetricType::METRIC_TYPE_IP>>(1, allocator.get()));
    flatten->SetIO(std::make_unique<MemoryIO>(allocator.get()));

    std::vector<float> vectors = {std::numeric_limits<float>::max(), 0.0F};
    std::vector<InnerIdType> ids = {0, 1};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());

    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> p_maxc(allocator.get());
    Vector<InnerIdType> maxcs(allocator.get());
    Vector<InnerIdType> p_node_to_cid(allocator.get());
    Vector<InnerIdType> node_to_cids(allocator.get());
    p_maxc.insert(p_maxc.end(), {0, 2});
    maxcs.insert(maxcs.end(), {0, 1});
    p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 2});
    node_to_cids.insert(node_to_cids.end(), {0, 0});
    cliques->Assign(std::move(p_maxc),
                    std::move(maxcs),
                    std::move(p_node_to_cid),
                    std::move(node_to_cids),
                    ids.size());

    InnerSearchParam search_param;
    search_param.ef = 1;
    search_param.topk = 1;
    search_param.distance_threshold = 1.0F;
    Vector<InnerIdType> seeds(allocator.get());
    seeds.push_back(0);

    const bool use_precise_csr = GENERATE(false, true);
    bool used_precise_csr = false;
    MCISearcherParam mci_param;
    mci_param.seed_count = 1;
    mci_param.seed_inner_ids = &seeds;
    mci_param.precise_vectors = use_precise_csr ? vectors.data() : nullptr;
    mci_param.dim = 1;
    mci_param.precise_vector_stride = 1;
    mci_param.metric = MetricType::METRIC_TYPE_IP;
    mci_param.used_precise_float_csr = &used_precise_csr;

    const float query = std::numeric_limits<float>::max();
    auto result =
        MCISearcher(common).Search(cliques, flatten, &query, search_param, mci_param, nullptr);

    REQUIRE(used_precise_csr == use_precise_csr);
    REQUIRE(result->Size() == 1);
    REQUIRE(result->Top().second == 1);
    REQUIRE(result->Top().first == 1.0F);
}

TEST_CASE("Static MCI reuses cached seeds and propagates depth in both storage paths",
          "[ut][mci][mci_distance_sampling][regression]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto quantizer =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer, io, common);
    std::vector<float> vectors{0, 1, 2, 3};
    std::vector<InnerIdType> ids{0, 1, 2, 3};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> offsets({0, 2, 4, 6}, allocator.get());
    Vector<InnerIdType> members({0, 1, 1, 2, 2, 3}, allocator.get());
    Vector<InnerIdType> inverse_offsets({0, 1, 3, 5, 6}, allocator.get());
    Vector<InnerIdType> inverse({0, 0, 1, 1, 2, 2}, allocator.get());
    cliques->Assign(std::move(offsets),
                    std::move(members),
                    std::move(inverse_offsets),
                    std::move(inverse),
                    ids.size());
    const bool raw = GENERATE(false, true);
    const bool bitmap = GENERATE(false, true);
    const bool exhaustive = GENERATE(false, true);
    const bool cached = GENERATE(false, true);
    const uint32_t depth = GENERATE(1U, 2U, 3U, 4U);
    Vector<InnerIdType> seeds(allocator.get());
    Vector<float> distances(allocator.get());
    for (uint64_t i = 0; i < (exhaustive ? 4 : 1); ++i) {
        seeds.push_back(i);
        distances.push_back((3.0F - vectors[i]) * (3.0F - vectors[i]));
    }
    MCISearcherParam params;
    params.seed_count = seeds.size();
    params.seed_inner_ids = &seeds;
    params.seed_distances = cached ? &distances : nullptr;
    params.enumerated_valid_count = exhaustive ? seeds.size() : 0;
    params.hops_limit = depth;
    params.expansion_idle_window = 0;
    params.precise_vectors = raw ? vectors.data() : nullptr;
    params.dim = 1;
    params.precise_vector_stride = 1;
    uint8_t mask[]{1, 1, 1, 1};
    params.valid_bitmap = bitmap ? mask : nullptr;
    params.valid_bitmap_size = 4;
    bool saturated = false;
    params.skipped_expansion = &saturated;
    InnerSearchParam search;
    search.ef = 4;
    search.topk = 4;
    SearchStatistics stats;
    QueryContext context;
    context.alloc = allocator.get();
    context.stats = &stats;
    const float query = 3.0F;
    const SearchLockScope scope(true);
    auto result = MCISearcher(common).Search(cliques, flatten, &query, search, params, &context);
    const uint64_t expected = exhaustive ? 4 : depth;
    REQUIRE(result->Size() == expected);
    CHECK(saturated == exhaustive);
    CHECK(stats.dist_cmp.load() == expected - (cached ? seeds.size() : 0));
    for (uint64_t id = 0; id < expected; ++id) {
        REQUIRE_FALSE(result->Empty());
        CHECK(result->Top().second == id);
        CHECK(result->Top().first == (query - vectors[id]) * (query - vectors[id]));
        result->Pop();
    }
}
