// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include "algorithm/hgraph/hgraph_parameter.h"
#include "algorithm/mci/mci_clique_activation.h"
#include "mci_packed_filter.h"
#include "mci_searcher.h"
#include "searcher_test.h"
#include "storage/serialization.h"
#include "utils/search_lock.h"
#include "vsag/factory.h"

using namespace vsag;

TEST_CASE("MCI packed predicate preserves CSR order and checks gather bounds",
          "[ut][mci][MCIPackedFilter]") {
    MCIPackedFilter filter;
    const uint64_t total = GENERATE(0, 1, 31, 32, 33, 1031);
    const uint64_t count = GENERATE(0, 1, 7, 8, 15, 16, 17, 63, 65, 133);
    std::vector<uint8_t> bitmap(total, 0);
    std::vector<InnerIdType> valid;
    for (InnerIdType id = 0; id < total; ++id) {
        if (id % 3 != 0) {
            bitmap[id] = 7;
            valid.push_back(id);
        }
    }
    std::vector<InnerIdType> members(count);
    for (uint64_t i = 0; i < count; ++i) {
        members[i] = i % 17 == 0 ? UINT32_MAX : (i * 193 + 7) % (total + 9);
    }
    std::vector<InnerIdType> expected;
    for (auto id : members) {
        if (id < total and bitmap[id] != 0) {
            expected.push_back(id);
        }
    }
    for (bool from_ids : {false, true}) {
        // An empty list still has an explicit non-null source.
        const InnerIdType empty = 0;
        filter.Build(total,
                     bitmap.data(),
                     from_ids ? (valid.empty() ? &empty : valid.data()) : nullptr,
                     valid.size());
        const auto scalar = filter.Filter(members.data(), members.size(), total, false);
        REQUIRE(std::vector<InnerIdType>(scalar.begin(), scalar.end()) == expected);
        const auto simd = filter.Filter(members.data(), members.size(), total, true);
        REQUIRE(std::vector<InnerIdType>(simd.begin(), simd.end()) == expected);
    }
    std::fill(bitmap.begin(), bitmap.end(), 0);
    filter.Build(total, bitmap.data(), nullptr, 0);
    REQUIRE(filter.Filter(members.data(), members.size(), total).count == 0);
}

TEST_CASE("MCI activation greedily covers near edges and persists in existing CSR format",
          "[ut][mci][MCIActivation]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr uint64_t total = 300;
    const uint64_t degree = GENERATE(3, 260);
    Vector<Vector<InnerIdType>> cliques(35, Vector<InnerIdType>(allocator.get()), allocator.get());
    cliques[0] = {0, 1};
    cliques[1] = {0, 1, 2};
    cliques[2] = {0, 2, 3};
    for (uint64_t c = 3; c < cliques.size(); ++c) {
        cliques[c] = {0, 4};
    }
    Vector<Vector<InnerIdType>> rows(total, Vector<InnerIdType>(allocator.get()), allocator.get());
    for (InnerIdType c = 0; c < cliques.size(); ++c) {
        for (auto id : cliques[c]) {
            rows[id].push_back(c);
        }
    }
    std::vector<InnerIdType> neighbors(total * degree, UINT32_MAX);
    std::iota(neighbors.begin(), neighbors.begin() + static_cast<int64_t>(degree), 1);
    MCIGraphView graph{neighbors.data(), nullptr, total, degree, degree};
    const auto original = rows;
    rank_mci_clique_activation(graph, cliques, rows, GENERATE(1, 4));
    REQUIRE(rows[0][0] == 1);
    REQUIRE(rows[0][1] == (degree == 3 ? 2 : 3));
    if (degree != 3) {
        REQUIRE(rows[0][2] == 2);
    }
    for (uint64_t id = 0; id < total; ++id) {
        auto sorted = rows[id];
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(sorted == original[id]);
    }
    Vector<InnerIdType> co(allocator.get());
    Vector<InnerIdType> no(allocator.get());
    co.push_back(0);
    no.push_back(0);
    Vector<InnerIdType> members(allocator.get());
    Vector<InnerIdType> ids(allocator.get());
    for (const auto& clique : cliques) {
        members.insert(members.end(), clique.begin(), clique.end());
        co.push_back(members.size());
    }
    for (const auto& row : rows) {
        ids.insert(ids.end(), row.begin(), row.end());
        no.push_back(ids.size());
    }
    CliqueDataCell cell(allocator.get());
    cell.Assign(std::move(co), std::move(members), std::move(no), std::move(ids), total);
    std::stringstream stream;
    IOStreamWriter writer(stream);
    cell.Serialize(writer);
    IOStreamReader reader(stream);
    CliqueDataCell loaded(allocator.get());
    loaded.Deserialize(reader, 2, total);
    CliqueDataCellSearchView view;
    REQUIRE(loaded.TryGetSearchView(total, view));
    REQUIRE(view.static_csr);
    REQUIRE(view.node_to_cids[0] == 1);
    REQUIRE(view.node_to_cids[1] == rows[0][1]);
}

TEST_CASE("MCI batching scores each legal point once and depth follows its parent",
          "[ut][mci][MCISearcher][MCIOptimization]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    const auto quantizer =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    const auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer, io, common);
    const std::vector<float> vectors{0, 1, 2, 3};
    std::vector<InnerIdType> ids{0, 1, 2, 3};
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> co({0, 2, 4, 6}, allocator.get());
    Vector<InnerIdType> members({0, 1, 1, 2, 2, 3}, allocator.get());
    Vector<InnerIdType> no({0, 1, 3, 5, 6}, allocator.get());
    Vector<InnerIdType> ci({0, 0, 1, 1, 2, 2}, allocator.get());
    cliques->Assign(std::move(co), std::move(members), std::move(no), std::move(ci), 4);
    const bool raw = GENERATE(false, true);
    const bool packed = GENERATE(false, true);
    const bool batch = GENERATE(false, true);
    const uint32_t depth = GENERATE(1, 2, 3, 4);
    const uint32_t cap = GENERATE(0, 1);
    Vector<InnerIdType> seeds({0, 0}, allocator.get());
    MCISearcherParam params;
    params.seed_count = 2;
    params.seed_inner_ids = &seeds;
    params.precise_vectors = raw ? vectors.data() : nullptr;
    params.dim = 1;
    params.precise_vector_stride = 1;
    params.packed_filter = packed;
    params.batch_distance = batch;
    params.depth_limit = depth;
    params.activation_cap = cap;
    const std::vector<uint8_t> bitmap(4, 9);
    params.valid_bitmap = bitmap.data();
    params.valid_bitmap_size = bitmap.size();
    InnerSearchParam search;
    search.topk = 4;
    search.ef = 4;
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator.get(), .stats = &stats};
    const float query = 3.25F;
    const SearchLockScope scope(true);
    auto result = MCISearcher(common).Search(cliques, flatten, &query, search, params, &ctx);
    const auto expected = std::min<uint32_t>(depth, cap == 0 ? 4 : 2);
    REQUIRE(result->Size() == expected);
    REQUIRE(stats.dist_cmp.load() == expected);
    std::vector<InnerIdType> actual;
    while (not result->Empty()) {
        actual.push_back(result->Top().second);
        result->Pop();
    }
    std::sort(actual.begin(), actual.end());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == i);
    }
}

TEST_CASE("MCI search optimization parameters reject malformed budgets",
          "[ut][mci][HGraphParameter][MCIOptimization]") {
    auto defaults = HGraphSearchParameters::FromJson(R"({"hgraph":{"ef_search":40}})");
    REQUIRE(defaults.mci_packed_filter);
    REQUIRE(defaults.mci_batch_distance);
    REQUIRE_FALSE(defaults.search_no_lock);
    REQUIRE(defaults.mci_activation_cap == 0);
    for (const std::string& field :
         {"mci_activation_cap", "mci_hops_limit", "mci_expansion_idle_window"}) {
        for (const std::string& value :
             {"-1", "1.5", "true", "4294967296", "18446744073709551615"}) {
            INFO(field << "=" << value);
            std::string parameters = R"({"hgraph":{"ef_search":40,")";
            parameters.append(field).append("\":").append(value).append("}}");
            REQUIRE_THROWS(HGraphSearchParameters::FromJson(parameters));
        }
    }
    REQUIRE_THROWS(HGraphSearchParameters::FromJson(
        R"({"hgraph":{"ef_search":40,"search_no_lock":true,"parallelism":2}})"));
    REQUIRE_FALSE(SearchLockScope::IsNoLock());
    {
        const SearchLockScope outer(true);
        REQUIRE(SearchLockScope::IsNoLock());
        auto worker = std::async(std::launch::async, [] { return SearchLockScope::IsNoLock(); });
        REQUIRE_FALSE(worker.get());
        {
            const SearchLockScope inner(false);
            REQUIRE_FALSE(SearchLockScope::IsNoLock());
        }
        REQUIRE(SearchLockScope::IsNoLock());
    }
    REQUIRE_FALSE(SearchLockScope::IsNoLock());
}

TEST_CASE("MCI batches across overlapping cliques and skips exhaustive expansion",
          "[ut][mci][MCISearcher][MCIOptimization]") {
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
    constexpr uint64_t total = 130;
    std::vector<float> vectors(total);
    std::vector<InnerIdType> ids(total);
    std::iota(vectors.begin(), vectors.end(), 0.0F);
    std::iota(ids.begin(), ids.end(), 0);
    flatten->Train(vectors.data(), total);
    flatten->BatchInsertVector(vectors.data(), total, ids.data());
    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> co({0, 96, 226}, allocator.get());
    Vector<InnerIdType> cm(allocator.get());
    cm.insert(cm.end(), ids.begin(), ids.begin() + 96);
    cm.insert(cm.end(), ids.begin(), ids.end());
    Vector<InnerIdType> no(allocator.get());
    Vector<InnerIdType> ci(allocator.get());
    no.push_back(0);
    for (uint64_t u = 0; u < total; ++u) {
        if (u < 96) {
            ci.push_back(0);
        }
        ci.push_back(1);
        no.push_back(ci.size());
    }
    cliques->Assign(std::move(co), std::move(cm), std::move(no), std::move(ci), total);
    const bool use_bitmap = GENERATE(false, true);
    const bool batch = GENERATE(false, true);
    const bool packed = GENERATE(false, true);
    std::vector<uint8_t> bitmap(total, 0);
    Vector<InnerIdType> valid(allocator.get());
    for (uint64_t u = 0; u < total; ++u) {
        if (not use_bitmap or u % 3 != 1) {
            bitmap[u] = 5;
            valid.push_back(u);
        }
    }
    Vector<InnerIdType> seeds(allocator.get());
    seeds.push_back(0);
    MCISearcherParam params;
    params.seed_inner_ids = &seeds;
    params.seed_count = 1;
    params.valid_bitmap = bitmap.data();
    params.valid_bitmap_size = total;
    params.valid_inner_ids = valid.data();
    params.valid_inner_count = valid.size();
    params.batch_distance = batch;
    params.packed_filter = packed;
    InnerSearchParam search;
    search.ef = total;
    search.topk = total;
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator.get(), .stats = &stats};
    const float query = 111.25F;
    const SearchLockScope scope(true);
    auto result = MCISearcher(common).Search(cliques, flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == valid.size());
    REQUIRE(stats.dist_cmp.load() == valid.size());
    REQUIRE(stats.hops.load() == 2);
    while (not result->Empty()) {
        const auto id = result->Top().second;
        REQUIRE(bitmap[id] != 0);
        REQUIRE(result->Top().first == (vectors[id] - query) * (vectors[id] - query));
        result->Pop();
    }
    params.seed_inner_ids = &valid;
    params.seed_count = valid.size();
    params.enumerated_valid_count = valid.size();
    params.activation_cap = 1;
    SearchStatistics exhaustive_stats;
    ctx.stats = &exhaustive_stats;
    result = MCISearcher(common).Search(cliques, flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == valid.size());
    REQUIRE(exhaustive_stats.dist_cmp.load() == valid.size());
    REQUIRE(exhaustive_stats.hops.load() == 0);
    Vector<InnerIdType> delta({0, 129}, allocator.get());
    cliques->AppendNewClique(delta, total);
    REQUIRE_THROWS_AS(MCISearcher(common).Search(cliques, flatten, &query, search, params, &ctx),
                      VsagException);
}

TEST_CASE("Ranked MCI builds reloads and searches mapped labels concurrently",
          "[ut][mci][MCIOptimization][hgraph]") {
    constexpr int64_t total = 64;
    constexpr int64_t dim = 4;
    std::vector<int64_t> labels(total);
    std::vector<float> vectors(total * dim);
    const bool identity = GENERATE(false, true);
    for (int64_t u = 0; u < total; ++u) {
        labels[u] = identity ? u : 1000 + u * 2;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[u * dim + j] = static_cast<float>((u * 7 + j * 13) % 71);
        }
    }
    const std::string config = R"({"dtype":"float32","metric_type":"l2","dim":4,
        "index_param":{"base_quantization_type":"fp32","graph_type":"nsw",
        "build_thread_count":2,"max_degree":16,"ef_construction":64,
        "use_mci":true,"mci_mcs":16,"mci_clique_max":3,"mci_alpha":1.2,
        "mci_clique_activation_order":true}})";
    auto made = Factory::CreateIndex("hgraph", config);
    REQUIRE(made.has_value());
    auto index = made.value();
    auto base = Dataset::Make()
                    ->NumElements(total)
                    ->Dim(dim)
                    ->Ids(labels.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    auto loaded = Factory::CreateIndex("hgraph", config).value();
    REQUIRE(loaded->Deserialize(stream).has_value());
    class ValidLabels : public Filter {
    public:
        std::vector<int64_t> ids;
        mutable std::function<void()> on_check;
        bool
        CheckValid(int64_t label) const override {
            if (on_check) {
                auto callback = std::exchange(on_check, nullptr);
                callback();
            }
            return std::find(ids.begin(), ids.end(), label) != ids.end();
        }
        float
        ValidRatio() const override {
            return 0.5F;
        }
        void
        GetValidIds(const int64_t** out, int64_t& count) const override {
            *out = ids.data();
            count = static_cast<int64_t>(ids.size());
        }
    };
    auto filter = std::make_shared<ValidLabels>();
    for (int64_t u = 0; u < total; u += 2) {
        filter->ids.push_back(labels[u]);
    }
    // Unknown labels must not make preparation subsample away a real legal label.
    filter->ids.insert(filter->ids.begin(), 9000);
    filter->ids.push_back(9001);
    filter->ids.push_back(-1);
    filter->ids.push_back(std::numeric_limits<int64_t>::max());
    filter->ids.push_back(labels[0]);
    std::reverse(filter->ids.begin(), filter->ids.end());
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    const std::string search = R"({"hgraph":{"ef_search":32,"use_mci":true,
        "hgraph_valid_ratio_threshold":1,"mci_seed_ratio":100,"mci_seed_coverage":0,
        "mci_activation_cap":8,"mci_hops_limit":9,"search_no_lock":true,"parallelism":1}})";
    std::vector<std::future<bool>> workers;
    for (uint64_t t = 0; t < 4; ++t) {
        workers.emplace_back(std::async(std::launch::async, [&] {
            for (uint64_t i = 0; i < 16; ++i) {
                auto result = loaded->KnnSearch(query, 4, search, filter);
                if (not result.has_value() or result.value()->GetDim() == 0 or
                    SearchLockScope::IsNoLock()) {
                    return false;
                }
                if (JsonType::Parse(result.value()->GetStatistics())["mci_hybrid_route"]
                        .GetString() != "mci") {
                    return false;
                }
                for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                    if (not filter->CheckValid(result.value()->GetIds()[j])) {
                        return false;
                    }
                }
            }
            return true;
        }));
    }
    for (auto& worker : workers) {
        REQUIRE(worker.get());
    }
    REQUIRE_FALSE(SearchLockScope::IsNoLock());

    // A filter callback may issue another MCI query after the outer predicate IDs have
    // been prepared but before they are packed. Its scratch must not overwrite the outer set.
    auto inner_filter = std::make_shared<ValidLabels>();
    for (int64_t u = 1; u < total; u += 2) {
        inner_filter->ids.push_back(labels[u]);
    }
    const std::string small_search = R"({"hgraph":{"ef_search":32,"use_mci":true,
        "hgraph_valid_ratio_threshold":1,"mci_seed_ratio":1,"mci_seed_coverage":0,
        "mci_hops_limit":9,"search_no_lock":true,"parallelism":1}})";
    auto reference = loaded->KnnSearch(query, 32, small_search, filter);
    REQUIRE(reference.has_value());
    bool nested = false;
    filter->on_check = [&] {
        auto inner_result = loaded->KnnSearch(query, 32, small_search, inner_filter);
        REQUIRE(inner_result.has_value());
        nested = true;
        REQUIRE(SearchLockScope::IsNoLock());
    };
    auto result = loaded->KnnSearch(query, 32, small_search, filter);
    REQUIRE(result.has_value());
    REQUIRE(nested);
    REQUIRE(result.value()->GetDim() == reference.value()->GetDim());
    REQUIRE(std::equal(reference.value()->GetIds(),
                       reference.value()->GetIds() + reference.value()->GetDim(),
                       result.value()->GetIds()));
    REQUIRE_FALSE(SearchLockScope::IsNoLock());
}
