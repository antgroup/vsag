// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <future>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#include "algorithm/hgraph/hgraph.h"
#include "algorithm/pyramid/pyramid.h"
#include "impl/pipnn/pipnn_incremental.h"
#include "index/index_impl.h"
#include "unittest.h"
#include "vsag/vsag.h"

namespace vsag {
struct PiPNNQuantizedAddTestAccess {
    static void
    write_graph(std::ostringstream& out,
                const PiPNNIncrementalState& state,
                const GraphInterfacePtr& graph,
                Allocator* allocator) {
        IOStreamWriter writer(out);
        state.Serialize(writer, false);
        Vector<InnerIdType> edges(allocator);
        for (auto id : state.ids) {
            graph->GetNeighbors(id, edges);
            StreamWriter::WriteVector(writer, edges);
        }
    }
    static auto
    hgraph(const IndexPtr& index) {
        auto impl = std::dynamic_pointer_cast<IndexImpl<HGraph>>(index);
        return std::dynamic_pointer_cast<HGraph>(impl->GetInnerIndex());
    }

    static auto
    lock_graph(const IndexPtr& index) {
        return std::shared_lock(hgraph(index)->global_mutex_);
    }

    static bool
    codes_ready(const IndexPtr& index, InnerIdType id) {
        auto graph = hgraph(index);
        std::shared_lock lock(graph->persistent_codes_mutex_);
        return graph->total_count_.load() > id &&
               graph->pending_code_writes_.find(id) == graph->pending_code_writes_.end();
    }

    static bool
    same_code_slot(const IndexPtr& index, InnerIdType a, InnerIdType b) {
        auto graph = hgraph(index);
        return graph->code_slot_map_->Resolve(a) == graph->code_slot_map_->Resolve(b);
    }

    static std::string
    topology(const IndexPtr& index, bool pyramid) {
        std::ostringstream buffer;
        if (pyramid) {
            auto impl = std::dynamic_pointer_cast<IndexImpl<Pyramid>>(index);
            auto graph = std::dynamic_pointer_cast<Pyramid>(impl->GetInnerIndex());
            for (const auto& node : graph->pipnn_graphs_in_path_order()) {
                auto it = graph->pipnn_graph_states_.find(node.get());
                if (it != graph->pipnn_graph_states_.end())
                    write_graph(buffer, *it->second, node, graph->allocator_);
            }
        } else {
            auto impl = std::dynamic_pointer_cast<IndexImpl<HGraph>>(index);
            auto graph = std::dynamic_pointer_cast<HGraph>(impl->GetInnerIndex());
            write_graph(buffer, *graph->pipnn_add_state_, graph->bottom_graph_, graph->allocator_);
        }
        return buffer.str();
    }
    static std::vector<FlattenInterfacePtr>
    codes(const IndexPtr& index, bool pyramid) {
        std::vector<FlattenInterfacePtr> stores;
        if (pyramid) {
            auto impl = std::dynamic_pointer_cast<IndexImpl<Pyramid>>(index);
            auto graph = std::dynamic_pointer_cast<Pyramid>(impl->GetInnerIndex());
            stores.push_back(graph->base_codes_);
            if (graph->has_precise_reorder())
                stores.push_back(graph->precise_codes_);
        } else {
            auto impl = std::dynamic_pointer_cast<IndexImpl<HGraph>>(index);
            auto graph = std::dynamic_pointer_cast<HGraph>(impl->GetInnerIndex());
            stores.push_back(graph->basic_flatten_codes_);
            if (graph->has_precise_reorder())
                stores.push_back(graph->high_precise_codes_);
        }
        return stores;
    }
};
}  // namespace vsag

TEST_CASE("Quantized incremental PiPNN keeps FP32 graph state through path promotion and reload",
          "[ut][pipnn][add][quantized_add]") {
    const bool pyramid = GENERATE(false, true);
    const std::string codec = GENERATE("sq8", "rabitq", "split", "rq3sq8");
    const bool raw = GENERATE(false, true);
    CAPTURE(pyramid, codec, raw);
    constexpr int dim = 64, n = 340;
    std::vector<float> vectors(n * dim);
    std::vector<int64_t> labels(n);
    std::iota(labels.begin(), labels.end(), 1000);
    std::mt19937 random(178);
    for (auto& v : vectors) v = std::generate_canonical<float, 24>(random);
    const std::string name = pyramid ? "pyramid" : "hgraph";
    auto parameters = vsag::JsonType::Parse(
        R"({"dim":64,"dtype":"float32","metric_type":"l2","index_param":{"graph_type":"pipnn","pipnn_incremental_add":true,"max_degree":16,"ef_construction":100,"alpha":1.2,"pipnn_max_leaf_size":64,"pipnn_min_leaf_size":12,"build_thread_count":1}})");
    auto create = [&](bool reference, bool incremental = true) {
        auto p = vsag::JsonType::Parse(parameters.Dump());
        auto setting = p["index_param"];
        setting["base_quantization_type"].SetString(reference        ? "fp32"
                                                    : codec == "sq8" ? "sq8"
                                                                     : "rabitq");
        setting["store_raw_vector"].SetBool(raw);
        setting["pipnn_incremental_add"].SetBool(incremental);
        if (!reference && codec != "sq8")
            setting["rabitq_bits_per_dim_base"].SetInt(codec == "rabitq" ? 1 : 3);
        if (!reference && (codec == "rq3sq8" || codec == "split")) {
            setting["use_reorder"].SetBool(true);
            setting["precise_quantization_type"].SetString(codec == "split" ? "rabitq" : "sq8");
            if (codec == "split")
                setting["rabitq_bits_per_dim_precise"].SetInt(5);
        }
        if (!pyramid && codec == "rq3sq8") {
            setting["build_by_base"].SetBool(raw);
        }
        if (pyramid) {
            setting["root_graph_type"].SetString("multi_layer");
            setting["index_min_size"].SetInt(24);
            setting["no_build_levels"].SetJson(vsag::JsonType::Parse("[]"));
        }
        p["index_param"].SetJson(setting);
        auto created = vsag::Factory::CreateIndex(name, p.Dump());
        REQUIRE(created.has_value());
        return created.value();
    };
    auto data = [&](int begin, int count) {
        auto ds = vsag::Dataset::Make()
                      ->Dim(dim)
                      ->NumElements(count)
                      ->Ids(labels.data() + begin)
                      ->Float32Vectors(vectors.data() + begin * dim)
                      ->Owner(false);
        if (pyramid) {
            std::vector<std::vector<std::string>> paths;
            for (int i = begin; i < begin + count; i++) {
                paths.push_back({i < 255 ? "a/x" : "b/z"});
                if (i % 11 == 0)
                    paths.back().push_back("a/shared");
            }
            ds->Paths("", std::move(paths));
        }
        return ds;
    };
    auto quantized = create(false), reference = create(true);
    std::vector<std::vector<uint8_t>> old;
    for (auto [begin, end] : {std::pair{0, 260}, std::pair{260, 300}, std::pair{300, n}}) {
        for (auto index : {reference, quantized}) {
            auto result = begin == 0 ? index->Build(data(begin, end - begin))
                                     : index->Add(data(begin, end - begin));
            CAPTURE(begin, end);
            if (!result.has_value())
                FAIL(result.error().message);
            REQUIRE(result.value().empty());
        }
        if (pyramid) {
            for (bool ref : {false, true}) {
                auto impl = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::Pyramid>>(
                    ref ? reference : quantized);
                auto memory = impl->GetInnerIndex()->GetMemoryUsageDetail();
                REQUIRE(memory.contains("pipnn_shared_fp32") == (!raw && !ref));
            }
        }
        auto expected = vsag::PiPNNQuantizedAddTestAccess::topology(reference, pyramid);
        REQUIRE_FALSE(expected.empty());
        REQUIRE(vsag::PiPNNQuantizedAddTestAccess::topology(quantized, pyramid) == expected);
        for (bool ref : {true, false}) {
            auto& index = ref ? reference : quantized;
            std::stringstream snapshot;
            REQUIRE(index->Serialize(snapshot).has_value());
            index = create(ref);
            REQUIRE(index->Deserialize(snapshot).has_value());
            REQUIRE(index->GetNumElements() == end);
            REQUIRE(vsag::PiPNNQuantizedAddTestAccess::topology(index, pyramid) == expected);
        }
        if (pyramid && raw && codec == "rabitq" && begin == 0) {
            std::stringstream same_base;
            REQUIRE(quantized->Serialize(same_base).has_value());
            auto nsw = create(false, false);
            REQUIRE(nsw->Deserialize(same_base).has_value());
            auto add = nsw->Add(data(end, 40));
            if (!add.has_value())
                FAIL(add.error().message);
            REQUIRE(add.value().empty());
            REQUIRE(nsw->GetNumElements() == end + 40);
            auto found = nsw->KnnSearch(data(end + 39, 1), 10, R"({"pyramid":{"ef_search":128}})");
            REQUIRE(found.has_value());
            REQUIRE(found.value()->GetDim() == 10);
        }
        auto stores = vsag::PiPNNQuantizedAddTestAccess::codes(quantized, pyramid);
        REQUIRE(stores.size() == (codec == "rq3sq8" ? 2 : 1));
        for (size_t j = 0; j < stores.size(); ++j) {
            auto store = stores[j];
            REQUIRE(store->SupportSplitCodeStorage() == (codec == "split"));
            const auto size = store->GetQuantizerCodeSize();
            std::vector<uint8_t> encoded(size), stored(size);
            for (auto row : {3, end - 1}) {
                REQUIRE(store->GetCodesById(row, stored.data()));
                REQUIRE(store->Encode(vectors.data() + row * dim, encoded.data()));
                REQUIRE(stored == encoded);
                if (row == 3) {
                    if (begin == 0)
                        old.push_back(stored);
                    else
                        REQUIRE(stored == old[j]);
                }
            }
        }
        for (auto row : {3, end - 1}) {
            auto query =
                quantized->KnnSearch(data(row, 1), 10, "{\"" + name + "\":{\"ef_search\":128}}");
            REQUIRE(query.has_value());
            REQUIRE(query.value()->GetDim() > 0);
        }
    }
}

TEST_CASE("PiPNN preserves quantized physical slots for cross-batch duplicates",
          "[ut][pipnn][add][quantized_add]") {
    const std::string codec = GENERATE("sq8", "rabitq", "split", "rq3sq8");
    CAPTURE(codec);
    constexpr uint64_t dim = 64, count = 160;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 100);
    std::vector<float> points(count * dim);
    std::mt19937 rng(47);
    for (auto& v : points) v = std::generate_canonical<float, 24>(rng);
    std::copy_n(points.data(), dim, points.data() + 102 * dim);
    std::copy_n(points.data() + 102 * dim, dim, points.data() + 140 * dim);
    auto p = vsag::JsonType::Parse(
        R"({"dim":64,"dtype":"float32","metric_type":"l2","index_param":{"graph_type":"pipnn","pipnn_incremental_add":true,"deduplicate_storage":true,"support_duplicate":true,"max_degree":16,"build_thread_count":1}})");
    auto cfg = p["index_param"];
    cfg["base_quantization_type"].SetString(codec == "sq8" ? "sq8" : "rabitq");
    if (codec != "sq8")
        cfg["rabitq_bits_per_dim_base"].SetInt(codec == "rabitq" ? 1 : 3);
    if (codec == "rq3sq8" || codec == "split") {
        cfg["use_reorder"].SetBool(true);
        cfg["precise_quantization_type"].SetString(codec == "split" ? "rabitq" : "sq8");
        if (codec == "split")
            cfg["rabitq_bits_per_dim_precise"].SetInt(5);
    }
    p["index_param"].SetJson(cfg);
    auto make = [&] {
        auto index = vsag::Factory::CreateIndex("hgraph", p.Dump());
        REQUIRE(index.has_value());
        return index.value();
    };
    auto index = make();
    for (auto [begin, end] : {std::pair{0, 100}, std::pair{100, 130}, std::pair{130, 160}}) {
        auto ds = vsag::Dataset::Make()
                      ->Dim(dim)
                      ->NumElements(end - begin)
                      ->Ids(ids.data() + begin)
                      ->Float32Vectors(points.data() + begin * dim)
                      ->Owner(false);
        auto result = begin ? index->Add(ds) : index->Build(ds);
        if (!result.has_value())
            FAIL(result.error().message);
        REQUIRE(result.value().empty());
        std::stringstream snapshot;
        REQUIRE(index->Serialize(snapshot).has_value());
        index = make();
        REQUIRE(index->Deserialize(snapshot).has_value());
        auto [logical, physical] =
            vsag::PiPNNQuantizedAddTestAccess::hgraph(index)->GetCodeStorageCounts();
        REQUIRE(logical == end);
        REQUIRE(physical == end - (end > 102) - (end > 140));
        if (end > 102)
            REQUIRE(vsag::PiPNNQuantizedAddTestAccess::same_code_slot(index, 0, 102));
        if (end > 140)
            REQUIRE(vsag::PiPNNQuantizedAddTestAccess::same_code_slot(index, 0, 140));
    }
}

TEST_CASE("PiPNN publishes new labels after graph and quantized codes are ready",
          "[ut][pipnn][add][quantized_add]") {
    constexpr int dim = 64;
    auto made = vsag::Factory::CreateIndex(
        "hgraph",
        R"({"dim":64,"dtype":"float32","metric_type":"l2","index_param":{"graph_type":"pipnn","pipnn_incremental_add":true,"base_quantization_type":"sq8","max_degree":16,"build_thread_count":1}})");
    REQUIRE(made.has_value());
    auto index = made.value();
    std::array<float, 34 * dim> rows{};
    std::mt19937 rng(401);
    for (auto& x : rows) x = std::generate_canonical<float, 24>(rng);
    std::array<int64_t, 34> ids;
    std::iota(ids.begin(), ids.end(), 100);
    ids.back() = ids[32];
    auto dataset = [&](int start, int count) {
        return vsag::Dataset::Make()
            ->Dim(dim)
            ->NumElements(count)
            ->Float32Vectors(rows.data() + dim * start)
            ->Ids(ids.data() + start)
            ->Owner(false);
    };
    REQUIRE(index->Build(dataset(0, 32)).has_value());
    auto gate = vsag::PiPNNQuantizedAddTestAccess::lock_graph(index);
    auto add = std::async(std::launch::async, [&] { return index->Add(dataset(32, 2)); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!vsag::PiPNNQuantizedAddTestAccess::codes_ready(index, 32) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    bool prepared = vsag::PiPNNQuantizedAddTestAccess::codes_ready(index, 32);
    bool published = index->CheckIdExist(ids[32]);
    gate.unlock();
    auto result = add.get();
    REQUIRE(prepared);
    REQUIRE_FALSE(published);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == std::vector<int64_t>{ids[32]});
    REQUIRE(index->CheckIdExist(ids[32]));
    REQUIRE(index->GetNumElements() == 33);
}
