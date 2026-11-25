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
    raw(const IndexPtr& index) {
        return hgraph(index)->raw_vector_;
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
    static std::vector<std::shared_ptr<PiPNNIncrementalState>>
    states(const IndexPtr& index, bool pyramid) {
        if (!pyramid)
            return {hgraph(index)->pipnn_add_state_};
        auto impl = std::dynamic_pointer_cast<IndexImpl<Pyramid>>(index);
        auto p = std::dynamic_pointer_cast<Pyramid>(impl->GetInnerIndex());
        REQUIRE(p->pipnn_exact_rows_.Size() == 0);
        REQUIRE(p->pipnn_projections_.projections.Size() ==
                static_cast<uint64_t>(index->GetNumElements()));
        std::vector<std::shared_ptr<PiPNNIncrementalState>> states;
        for (const auto& [graph, state] : p->pipnn_graph_states_) states.push_back(state);
        return states;
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
    const std::string codec = GENERATE("sq8", "rabitq", "rq3sq8");
    CAPTURE(codec);
    constexpr int dim = 64, base = 160, total = 304;
    auto config = vsag::JsonType::Parse(
        R"({"dim":64,"dtype":"float32","metric_type":"l2","index_param":{"graph_type":"pipnn","pipnn_incremental_add":true,"store_raw_vector":true,"max_degree":16,"build_thread_count":4}})");
    auto setting = config["index_param"];
    setting["base_quantization_type"].SetString(codec == "sq8" ? "sq8" : "rabitq");
    if (codec == "rq3sq8") {
        setting["rabitq_bits_per_dim_base"].SetInt(3);
        setting["use_reorder"].SetBool(true);
        setting["precise_quantization_type"].SetString("sq8");
    }
    auto made = vsag::Factory::CreateIndex("hgraph", config.Dump());
    REQUIRE(made.has_value());
    auto index = made.value();
    std::vector<float> rows(total * dim);
    std::mt19937 rng(401);
    for (auto& x : rows) x = std::generate_canonical<float, 24>(rng);
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 100);
    // Reject an existing label and an intra-batch duplicate on either side of a block boundary.
    ids[base + 4] = ids[0];
    ids[base + 90] = ids[base + 1];
    auto dataset = [&](int start, int count) {
        return vsag::Dataset::Make()
            ->Dim(dim)
            ->NumElements(count)
            ->Float32Vectors(rows.data() + dim * start)
            ->Ids(ids.data() + start)
            ->Owner(false);
    };
    REQUIRE(index->Build(dataset(0, base)).has_value());
    auto gate = vsag::PiPNNQuantizedAddTestAccess::lock_graph(index);
    auto add =
        std::async(std::launch::async, [&] { return index->Add(dataset(base, total - base)); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    constexpr auto last = total - 3;
    while (!vsag::PiPNNQuantizedAddTestAccess::codes_ready(index, last) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    bool prepared = vsag::PiPNNQuantizedAddTestAccess::codes_ready(index, last);
    bool published = index->CheckIdExist(ids[base]) || index->CheckIdExist(ids.back());
    gate.unlock();
    auto result = add.get();
    REQUIRE(prepared);
    REQUIRE_FALSE(published);
    REQUIRE(result.has_value());
    REQUIRE(result.value() == std::vector<int64_t>{ids[0], ids[base + 1]});
    REQUIRE(index->GetNumElements() == total - 2);
    auto topology = vsag::PiPNNQuantizedAddTestAccess::topology(index, false);
    std::stringstream snapshot;
    REQUIRE(index->Serialize(snapshot).has_value());
    auto reopened = vsag::Factory::CreateIndex("hgraph", config.Dump());
    REQUIRE(reopened.has_value());
    index = reopened.value();
    REQUIRE(index->Deserialize(snapshot).has_value());
    REQUIRE(vsag::PiPNNQuantizedAddTestAccess::topology(index, false) == topology);
    auto stores = vsag::PiPNNQuantizedAddTestAccess::codes(index, false);
    stores.push_back(vsag::PiPNNQuantizedAddTestAccess::raw(index));
    for (auto& store : stores) {
        REQUIRE(store);
        std::vector<uint8_t> expected(store->GetQuantizerCodeSize()), actual(expected.size());
        vsag::InnerIdType inner = base;
        for (int row = base; row < total; ++row) {
            if (row == base + 4 || row == base + 90)
                continue;
            REQUIRE(index->CheckIdExist(ids[row]));
            REQUIRE(store->Encode(rows.data() + dim * row, expected.data()));
            REQUIRE(store->GetCodesById(inner++, actual.data()));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("PiPNN Add reads stored code distances and original sketches after reload",
          "[ut][pipnn][add][coded_add]") {
    const bool pyramid = GENERATE(false, true);
    const bool split = GENERATE(false, true);
    CAPTURE(pyramid, split);
    const std::string kind = pyramid ? "pyramid" : "hgraph";
    constexpr uint32_t dim = 64, total = 370;
    auto cfg = vsag::JsonType::Parse(
        R"({"dtype":"float32","dim":64,"metric_type":"l2","index_param":{"graph_type":"pipnn","max_degree":16,"pipnn_incremental_add":true,"pipnn_add_use_codes":true,"store_raw_vector":false,"build_thread_count":4,"pipnn_max_leaf_size":48,"pipnn_min_leaf_size":12}})");
    auto setting = cfg["index_param"];
    setting["base_quantization_type"].SetString(split ? "rabitq" : "sq8");
    if (split) {
        setting["use_reorder"].SetBool(true);
        setting["precise_quantization_type"].SetString("rabitq");
        setting["rabitq_bits_per_dim_base"].SetInt(3);
        setting["rabitq_bits_per_dim_precise"].SetInt(5);
    }
    if (pyramid) {
        setting["root_graph_type"].SetString("multi_layer");
        setting["index_min_size"].SetInt(24);
        setting["no_build_levels"].SetJson(vsag::JsonType::Parse("[]"));
    }
    cfg["index_param"].SetJson(setting);
    auto create = [&]() {
        auto result = vsag::Factory::CreateIndex(kind, cfg.Dump());
        if (!result)
            FAIL(result.error().message);
        return result.value();
    };
    std::vector<float> values(dim * total);
    std::mt19937 rng(71);
    for (auto& v : values) v = std::generate_canonical<float, 24>(rng);
    std::vector<int64_t> labels(total);
    std::iota(labels.begin(), labels.end(), 0);
    auto ds = [&](uint32_t start, uint32_t n) {
        auto data = vsag::Dataset::Make()
                        ->Dim(dim)
                        ->NumElements(n)
                        ->Float32Vectors(values.data() + dim * start)
                        ->Ids(labels.data() + start)
                        ->Owner(false);
        if (pyramid) {
            std::vector<std::vector<std::string>> paths;
            for (auto i = start; i < start + n; ++i) {
                paths.push_back({i < 235 ? "a/x" : "b/z"});
                if (i % 9 == 0)
                    paths.back().push_back("a/shared");
            }
            data->Paths("", std::move(paths));
        }
        return data;
    };
    auto index = create();
    for (auto [start, end] : {std::pair{0U, 240U}, std::pair{240U, 320U}, std::pair{320U, total}}) {
        auto result =
            start ? index->Add(ds(start, end - start)) : index->Build(ds(start, end - start));
        if (!result)
            FAIL(result.error().message);
        REQUIRE(result->empty());
        const auto topology = vsag::PiPNNQuantizedAddTestAccess::topology(index, pyramid);
        std::stringstream wire;
        REQUIRE(index->Serialize(wire).has_value());
        index = create();
        auto loaded = index->Deserialize(wire);
        if (!loaded)
            FAIL(loaded.error().message);
        REQUIRE(index->GetNumElements() == end);
        REQUIRE(vsag::PiPNNQuantizedAddTestAccess::topology(index, pyramid) == topology);
        if (!start)
            continue;
        auto store = vsag::PiPNNQuantizedAddTestAccess::codes(index, pyramid).front();
        uint64_t checked = 0;
        bool promoted = false;
        for (const auto& state : vsag::PiPNNQuantizedAddTestAccess::states(index, pyramid)) {
            REQUIRE(state->Rows().empty());
            for (uint32_t i = 0; i < state->ids.size(); ++i) {
                const auto id = state->ids[i];
                float dot = 0;
                for (uint32_t j = 0; j < dim; ++j) dot += values[id * dim + j] * state->planes[j];
                REQUIRE(std::abs(state->projections.Row(i)[0] - dot) < 1e-4F);
            }
            if (!state->ids.empty() &&
                *std::min_element(state->ids.begin(), state->ids.end()) == 235)
                promoted = true;
            for (const auto& leaf : state->leaves) {
                if (leaf.nearest.Width() == 0)
                    continue;
                for (uint64_t i = 0; i < leaf.points.size(); ++i) {
                    const auto id = state->ids[leaf.points[i]];
                    if (id < start)
                        continue;
                    const auto e = leaf.nearest.Row(i).front();
                    const auto dist = store->ComputePairVectors(id, state->ids[e.target]);
                    REQUIRE(std::abs(dist - e.distance) <= 1e-4F * (1 + std::abs(dist)));
                    ++checked;
                }
            }
        }
        REQUIRE(checked > 0);
        if (pyramid)
            REQUIRE(promoted);
        auto q = index->KnnSearch(ds(end - 1, 1), 10, "{\"" + kind + "\":{\"ef_search\":128}}");
        REQUIRE(q.has_value());
        REQUIRE(q.value()->GetDim() == 10);
    }
}
