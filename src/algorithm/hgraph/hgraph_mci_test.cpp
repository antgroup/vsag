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

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "algorithm/mci/mci_builder.h"
#include "algorithm/mci/mci_coverage.h"
#include "algorithm/mci/mci_local_builder.h"
#include "algorithm/mci/mci_runner.h"
#include "analyzer/hgraph_analyzer.h"
#include "datacell/clique_datacell.h"
#include "hgraph.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization.h"
#include "unittest.h"
#include "vsag/bitset.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

// Observe the actual candidate-search entry used by both insertion and deletion repair.
class SearchCountingMCIHGraph : public vsag::HGraph {
public:
    using vsag::HGraph::HGraph;

    vsag::DatasetPtr
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const vsag::FilterPtr& filter) const override {
        INFO(parameters);
        const auto params = vsag::JsonType::Parse(parameters);
        // GetStats may also invoke public search to analyze the quantizer.
        if (params["hgraph"].Contains("use_mci")) {
            ++candidate_searches;
            REQUIRE_FALSE(params["hgraph"]["use_mci"].GetBool());
            REQUIRE(params["hgraph"]["ef_search"].GetInt() >= 100);
            if (fail_candidate_search) {
                throw std::bad_alloc();
            }
        }
        return vsag::HGraph::KnnSearch(query, k, parameters, filter);
    }

    mutable uint64_t candidate_searches{0};
    bool fail_candidate_search{false};
};

class Int8QueryCheckingHGraph : public vsag::HGraph {
public:
    using vsag::HGraph::HGraph;

    vsag::DatasetPtr
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const vsag::FilterPtr& filter) const override {
        REQUIRE(query->GetFloat32Vectors() == nullptr);
        REQUIRE(query->GetInt8Vectors() != nullptr);
        const auto dim = static_cast<uint64_t>(query->GetDim());
        bool matches = false;
        for (uint64_t offset = 0; offset < expected_vectors.size(); offset += dim) {
            matches |= std::equal(query->GetInt8Vectors(),
                                  query->GetInt8Vectors() + dim,
                                  expected_vectors.data() + offset);
        }
        REQUIRE(matches);
        ++queries;
        return vsag::HGraph::KnnSearch(query, k, parameters, filter);
    }

    std::vector<int8_t> expected_vectors;
    mutable uint64_t queries{0};
};

class HalfRatioAllValidFilter : public vsag::Filter {
public:
    explicit HalfRatioAllValidFilter(const std::vector<int64_t>& ids) : ids_(ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    float
    ValidRatio() const override {
        return 0.5F;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = ids_.data();
        count = static_cast<int64_t>(ids_.size());
    }

private:
    std::vector<int64_t> ids_;
};

class CountingValidIdsFilter : public HalfRatioAllValidFilter {
public:
    explicit CountingValidIdsFilter(const std::vector<int64_t>& ids)
        : HalfRatioAllValidFilter(ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        check_count_.fetch_add(1, std::memory_order_relaxed);
        return HalfRatioAllValidFilter::CheckValid(id);
    }

    [[nodiscard]] uint64_t
    CheckCount() const {
        return check_count_.load(std::memory_order_relaxed);
    }

private:
    mutable std::atomic<uint64_t> check_count_{0};
};

class CallbackOnlyFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id >= 0;
    }

    float
    ValidRatio() const override {
        return 0.5F;
    }
};

// Inject at each allocation ordinal until the operation succeeds, not at a byte limit.
class FlushFailureAllocator : public vsag::DefaultAllocator {
public:
    void*
    Allocate(uint64_t bytes) override {
        if (remaining == 0) {
            throw std::bad_alloc();
        }
        if (remaining > 0) {
            --remaining;
        }
        return vsag::DefaultAllocator::Allocate(bytes);
    }

    int64_t remaining{-1};
};

struct TestEdge {
    uint32_t u{0};
    uint32_t v{0};
};

class WorkerFailAllocator : public vsag::Allocator {
public:
    WorkerFailAllocator() : owner_(std::this_thread::get_id()) {
    }

    std::string
    Name() override {
        return "worker-fail-allocator";
    }

    void*
    Allocate(uint64_t size) override {
        if (std::this_thread::get_id() != owner_) {
            throw std::bad_alloc();
        }
        auto* result = std::malloc(size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }

    void
    Deallocate(void* pointer) override {
        std::free(pointer);
    }

    void*
    Reallocate(void* pointer, uint64_t size) override {
        if (std::this_thread::get_id() != owner_) {
            throw std::bad_alloc();
        }
        auto* result = std::realloc(pointer, size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }

private:
    std::thread::id owner_;
};

std::string
generate_hgraph_mci_params(int64_t dim,
                           const std::string& knng_source = vsag::HGRAPH_MCI_KNNG_SOURCE_HGRAPH) {
    auto params = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 6,
                "neighbor_sample_rate": 0.3,
                "mci_mcs": 8,
                "mci_clique_max": 4,
                "mci_incremental_clique_max": 4,
                "mci_alpha": 1.2
            }
        }
    )");
    params["dim"].SetInt(dim);
    params["index_param"][vsag::HGRAPH_MCI_KNNG_SOURCE].SetString(knng_source);
    return params.Dump();
}

vsag::DatasetPtr
make_dataset(std::vector<int64_t>& ids,
             std::vector<float>& vectors,
             int64_t offset,
             int64_t count,
             int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data() + offset)
        ->Float32Vectors(vectors.data() + offset * dim)
        ->Owner(false);
    return dataset;
}

struct FlushedMCISnapshot {
    std::string bytes;
    uint64_t begin;
    uint64_t end;
    std::vector<int64_t> labels;
    vsag::CliqueDataCellPtr cliques;
};

// Inspect the real serialized CSR without granting tests access to HGraph private state.
FlushedMCISnapshot
read_flushed_mci(const vsag::IndexPtr& index, vsag::Allocator* allocator) {
    REQUIRE(index->Flush().has_value());
    const auto stats = vsag::JsonType::Parse(index->GetStats());
    const uint64_t n = stats["mci_total_nodes"].GetInt();
    const uint64_t c = stats["mci_base_clique_count"].GetInt();
    const uint64_t m = stats["mci_base_membership_count"].GetInt();
    REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    vsag::IOStreamReader reader(stream);
    auto footer = vsag::Footer::Parse(reader);
    REQUIRE(footer != nullptr);
    REQUIRE_FALSE(footer->GetMetadata()->Get("has_conjugate_graph").GetBool());
    FlushedMCISnapshot snapshot;
    snapshot.bytes = stream.str();
    snapshot.end = snapshot.bytes.size() - footer->Length();
    // v2: four CSR vectors, clique count, empty delta rows, and inactive/retired masks.
    const uint64_t mci_bytes = 88 + 13 * (n + c) + 8 * m;
    REQUIRE(snapshot.end >= mci_bytes);
    snapshot.begin = snapshot.end - mci_bytes;
    reader.Seek(0);
    vsag::StreamReader::ReadVector(reader, snapshot.labels);
    snapshot.labels.resize(n);
    reader.Seek(snapshot.begin);
    snapshot.cliques = std::make_shared<vsag::CliqueDataCell>(allocator);
    snapshot.cliques->Deserialize(reader, 2);
    REQUIRE(reader.GetCursor() == snapshot.end);
    return snapshot;
}

uint64_t
snapshot_degree(const FlushedMCISnapshot& snapshot, vsag::InnerIdType node) {
    vsag::DefaultAllocator allocator;
    vsag::Vector<vsag::InnerIdType> cids(&allocator);
    snapshot.cliques->CollectNodeCliqueIds(node, cids);
    std::set<vsag::InnerIdType> neighbors;
    for (auto cid : cids) {
        vsag::Vector<vsag::InnerIdType> members(&allocator);
        snapshot.cliques->GetCliqueMembers(cid, members);
        neighbors.insert(members.begin(), members.end());
    }
    neighbors.erase(node);
    return neighbors.size();
}

// Replace only MCI in a valid small index, to control overlap and JOIN eligibility exactly.
vsag::IndexPtr
with_mci_fixture(const vsag::IndexPtr& index,
                 const std::string& parameters,
                 const std::vector<std::vector<vsag::InnerIdType>>& cliques) {
    vsag::DefaultAllocator allocator;
    const auto snapshot = read_flushed_mci(index, &allocator);
    vsag::CliqueDataCell replacement(&allocator);
    replacement.Clear(snapshot.labels.size());
    for (const auto& clique : cliques) {
        vsag::Vector<vsag::InnerIdType> members(clique.begin(), clique.end(), &allocator);
        replacement.AppendNewClique(members, snapshot.labels.size());
    }
    replacement.Flush(snapshot.labels.size());
    std::stringstream stream;
    stream.write(snapshot.bytes.data(), static_cast<std::streamsize>(snapshot.begin));
    vsag::IOStreamWriter writer(stream);
    replacement.Serialize(writer);
    stream.write(snapshot.bytes.data() + snapshot.end,
                 static_cast<std::streamsize>(snapshot.bytes.size() - snapshot.end));
    auto result = vsag::Factory::CreateIndex("hgraph", parameters);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->Deserialize(stream).has_value());
    return result.value();
}

}  // namespace

TEST_CASE("HGraph companion MCI selects the KNNG build source", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 32;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 8000);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto verify_source = [&](const std::string& source) {
        auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim, source));
        REQUIRE(index.has_value());
        auto build_result = index.value()->Build(make_dataset(ids, vectors, 0, total, dim));
        REQUIRE(build_result.has_value());
        REQUIRE(build_result.value().empty());
        const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
        REQUIRE(stats["mci_has_index"].GetBool());
        REQUIRE(stats["mci_covered_nodes"].GetInt() == total);
    };

    SECTION("completed HGraph") {
        verify_source(vsag::HGRAPH_MCI_KNNG_SOURCE_HGRAPH);
    }
    SECTION("dedicated ODescent") {
        verify_source(vsag::HGRAPH_MCI_KNNG_SOURCE_ODESCENT);
    }
}

TEST_CASE("HGraph companion MCI selectively repairs under-covered members", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 32;
    std::vector<int64_t> ids(total + 1);
    std::iota(ids.begin(), ids.end(), 9000);
    std::vector<float> vectors((total + 1) * dim);
    for (int64_t i = 0; i <= total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto parameters = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    parameters["index_param"]["mci_delete_clique_size_threshold"].SetInt(5);
    const auto parameter_string = parameters.Dump();
    auto index = vsag::Factory::CreateIndex("hgraph", parameter_string);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());

    auto remove_result = index.value()->Remove({ids[total / 2], ids[total / 2 + 1]});
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 2);
    auto stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 2);
    REQUIRE(stats["mci_retired_clique_count"].GetInt() > 0);
    REQUIRE(stats["mci_delta_clique_count"].GetInt() > 0);
    REQUIRE(stats["mci_covered_nodes"].GetInt() == total - 2);

    REQUIRE(index.value()->Add(make_dataset(ids, vectors, total, 1, dim)).has_value());
    stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 2);
    REQUIRE(stats["mci_covered_nodes"].GetInt() == total - 1);

    auto binary = index.value()->Serialize();
    REQUIRE(binary.has_value());
    auto restored = vsag::Factory::CreateIndex("hgraph", parameter_string);
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(binary.value()).has_value());
    const auto restored_stats = vsag::JsonType::Parse(restored.value()->GetStats());
    REQUIRE(restored_stats["mci_has_index"].GetBool());
    REQUIRE(restored_stats["mci_inactive_node_count"].GetInt() == 2);
    REQUIRE(restored_stats["mci_retired_clique_count"].GetInt() ==
            stats["mci_retired_clique_count"].GetInt());
    REQUIRE(restored_stats["mci_delta_clique_count"].GetInt() ==
            stats["mci_delta_clique_count"].GetInt());
}

TEST_CASE("HGraph MCI mark removal projects the complete batch before repair",
          "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t total = 8;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 100);
    std::vector<float> vectors(total * dim, 0.0F);
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["index_param"]["mci_delete_clique_size_threshold"].SetInt(3);
    const auto json = params.Dump();
    auto built = vsag::Factory::CreateIndex("hgraph", json);
    REQUIRE(built.has_value());
    REQUIRE(built.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());
    // Each deletion alone leaves the first clique at the threshold; together they retire it.
    auto index = with_mci_fixture(built.value(), json, {{0, 1, 2, 3}, {2, 4, 5}, {3, 6, 7}});
    const auto removed = index->Remove({ids[1], ids[0], ids[1]});
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() == 2);
    const auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 2);
    REQUIRE(stats["mci_retired_clique_count"].GetInt() == 1);
    REQUIRE(stats["mci_covered_nodes"].GetInt() == total - 2);
    vsag::DefaultAllocator allocator;
    const auto snapshot = read_flushed_mci(index, &allocator);
    REQUIRE(snapshot_degree(snapshot, 0) == 0);
    REQUIRE(snapshot_degree(snapshot, 1) == 0);
    REQUIRE(snapshot_degree(snapshot, 2) > 2);
    REQUIRE(snapshot_degree(snapshot, 3) > 2);
}

TEST_CASE("MCI default deletion threshold retires only affected cliques below thirty members",
          "[ut][hgraph][mci]") {
    const bool flush_before_delete = GENERATE(false, true);
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    constexpr uint64_t total = 64;
    cell.Clear(total);
    vsag::Vector<vsag::InnerIdType> members(&allocator);
    for (vsag::InnerIdType id = 0; id < 31; ++id) {
        members.push_back(id);
    }
    cell.AppendNewClique(members, total);
    members.clear();
    for (vsag::InnerIdType id = 31; id < 61; ++id) {
        members.push_back(id);
    }
    cell.AppendNewClique(members, total);
    members = {61, 62, 63};
    cell.AppendNewClique(members, total);
    if (flush_before_delete) {
        cell.Flush(total);
    }

    const vsag::HGraphMCIParameters defaults;
    const vsag::Vector<vsag::InnerIdType> removed({0, 31}, &allocator);
    const auto snapshot = cell.PrepareDelete(
        removed, defaults.delete_clique_size_threshold, defaults.delete_node_mct_threshold);
    // Remaining size 30 is retained; 29 is retired. The unrelated three-member clique stays.
    REQUIRE(snapshot.affected_clique_ids.size() == 2);
    REQUIRE(snapshot.retired_clique_ids.size() == 1);
    REQUIRE(snapshot.retired_clique_ids.front() == 1);
    REQUIRE(snapshot.repair_node_ids.size() == 29);
    for (uint64_t i = 0; i < snapshot.repair_node_ids.size(); ++i) {
        REQUIRE(snapshot.repair_node_ids[i] == i + 32);
    }
    cell.CommitDelete(removed, snapshot.retired_clique_ids, total);
    REQUIRE(cell.GetCliqueMemberCount(0) == 30);
    REQUIRE(cell.GetCliqueMemberCount(1) == 0);
    REQUIRE(cell.GetCliqueMemberCount(2) == 3);
}

TEST_CASE("Clique validation rejects empty node offsets and unknown formats", "[ut][hgraph][mci]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    vsag::Vector<vsag::InnerIdType> p_maxc(&allocator);
    p_maxc.push_back(0);
    vsag::Vector<vsag::InnerIdType> maxcs(&allocator);
    vsag::Vector<vsag::InnerIdType> p_node_to_cid(&allocator);
    vsag::Vector<vsag::InnerIdType> node_to_cids(&allocator);
    REQUIRE_THROWS_AS(cell.Assign(std::move(p_maxc),
                                  std::move(maxcs),
                                  std::move(p_node_to_cid),
                                  std::move(node_to_cids),
                                  0),
                      vsag::VsagException);
    std::stringstream stream;
    vsag::IOStreamReader reader(stream);
    REQUIRE_THROWS_AS(cell.Deserialize(reader, 0), vsag::VsagException);
    REQUIRE_THROWS_AS(cell.Deserialize(reader, 3), vsag::VsagException);
}

TEST_CASE("HGraph FP32 MCI deletion repairs use the complete Add candidate pipeline",
          "[ut][hgraph][mci][repair_add]") {
    const auto mode = GENERATE(vsag::RemoveMode::MARK_REMOVE, vsag::RemoveMode::FORCE_REMOVE);
    const std::string io = GENERATE("memory_io", "block_memory_io");
    CAPTURE(mode, io);
    constexpr int64_t dim = 16;
    constexpr int64_t total = 16;
    std::vector<int64_t> ids(total + 1);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors((total + 1) * dim);
    for (uint64_t i = 0; i < vectors.size(); ++i) {
        vectors[i] = i % dim == (i / dim) % dim ? 1.0F : 0.0F;
    }

    auto configuration = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    auto params = configuration["index_param"];
    params["graph_type"].SetString("nsw");
    params["build_thread_count"].SetInt(1);
    params["max_degree"].SetInt(32);
    params["support_force_remove"].SetBool(true);
    params["base_io_type"].SetString(io);
    params["base_quantization_type"].SetString("fp32");
    params["use_reorder"].SetBool(false);
    params["mci_mcs"].SetInt(total);
    params["mci_clique_max"].SetInt(total);
    params["mci_incremental_clique_max"].SetInt(total);
    params["mci_delete_clique_size_threshold"].SetInt(total + 1);
    params["mci_delete_node_mct_threshold"].SetInt(1);

    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    SearchCountingMCIHGraph index(vsag::HGraph::CheckAndMappingExternalParam(params, common),
                                  common);
    auto dataset = [&](int64_t begin, int64_t count) {
        return make_dataset(ids, vectors, begin, count, dim);
    };
    REQUIRE(index.Build(dataset(0, total)).empty());
    index.candidate_searches = 0;
    // All distinct vectors are equidistant. Rebuild a clique around inner ID 0
    // using higher IDs as well, not clamp its capacity/visibility to the old ID + 1 prefix.
    REQUIRE(index.Remove({ids[total - 1]}, mode) == 1);
    REQUIRE(index.candidate_searches > 0);
    REQUIRE(index.GetNumElements() == total - 1);
    auto stats = vsag::JsonType::Parse(index.GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
    REQUIRE(stats["mci_covered_nodes"].GetInt() == total - 1);
    REQUIRE(stats["mci_max_clique_size"].GetInt() == total - 1);
    REQUIRE(stats["mci_total_nodes"].GetInt() ==
            (mode == vsag::RemoveMode::FORCE_REMOVE ? total - 1 : total));

    index.candidate_searches = 0;
    REQUIRE(index.Add(dataset(total, 1)).empty());
    REQUIRE(index.candidate_searches > 0);
    REQUIRE(index.GetNumElements() == total);
    auto result = index.KnnSearch(
        dataset(0, 1), total + 1, R"({"hgraph":{"ef_search":100,"use_mci":false}})", nullptr);
    REQUIRE(result->GetDim() == total);
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        REQUIRE(result->GetIds()[i] != ids[total - 1]);
    }
    std::vector<int64_t> live_ids(ids.begin(), ids.end() - 2);
    live_ids.push_back(ids.back());
    auto filter = std::make_shared<HalfRatioAllValidFilter>(live_ids);
    auto mci_result = index.vsag::HGraph::KnnSearch(
        dataset(0, 1),
        total,
        R"({"hgraph":{"ef_search":100,"use_mci":true,"mci_seed_ratio":100,
                       "hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(mci_result->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(mci_result->GetDim() == total);
}

TEST_CASE("HGraph rejects mismatched serialized MCI node counts", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 6;
    const auto delta = GENERATE(-1, 1);
    const bool corrupt_metadata = GENERATE(false, true);
    auto config = vsag::JsonType::Parse(generate_hgraph_mci_params(2));
    config["index_param"]["support_force_remove"].SetBool(true);
    auto index = vsag::Factory::CreateIndex("hgraph", config.Dump()).value();
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};
    std::vector<float> vectors{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
    REQUIRE(index->Build(make_dataset(ids, vectors, 0, total, 2)).has_value());
    vsag::DefaultAllocator allocator;
    const auto snapshot = read_flushed_mci(index, &allocator);
    const auto wrong_total = static_cast<uint64_t>(static_cast<int64_t>(total) + delta);
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    if (corrupt_metadata) {
        std::stringstream original(snapshot.bytes);
        vsag::IOStreamReader reader(original);
        auto footer = vsag::Footer::Parse(reader);
        auto metadata = footer->GetMetadata();
        auto basic_info = metadata->Get(vsag::BASIC_INFO);
        basic_info["total_count"].SetUint64(wrong_total);
        metadata->Set(vsag::BASIC_INFO, basic_info);
        stream.write(snapshot.bytes.data(), static_cast<std::streamsize>(snapshot.end));
        footer->Write(writer);
    } else {
        // This MCI block is internally valid, but has a different physical ID domain.
        vsag::CliqueDataCell replacement(&allocator);
        replacement.Clear(wrong_total);
        vsag::Vector<vsag::InnerIdType> members({0, 1}, &allocator);
        replacement.AppendNewClique(members, wrong_total);
        replacement.Flush(wrong_total);
        stream.write(snapshot.bytes.data(), static_cast<std::streamsize>(snapshot.begin));
        replacement.Serialize(writer);
        stream.write(snapshot.bytes.data() + snapshot.end,
                     static_cast<std::streamsize>(snapshot.bytes.size() - snapshot.end));
    }
    auto restored = vsag::Factory::CreateIndex("hgraph", config.Dump()).value();
    auto result = restored->Deserialize(stream);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::INVALID_BINARY);
}

TEST_CASE("HGraph MCI stays unpublished when force-remove repair allocation fails",
          "[ut][hgraph][mci][force_remove]") {
    auto config = vsag::JsonType::Parse(generate_hgraph_mci_params(2));
    auto params = config["index_param"];
    params["support_force_remove"].SetBool(true);
    params["build_thread_count"].SetInt(1);
    params["mci_delete_clique_size_threshold"].SetInt(30);
    params["mci_delete_node_mct_threshold"].SetInt(3);
    vsag::IndexCommonParam common;
    common.dim_ = 2;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    SearchCountingMCIHGraph index(vsag::HGraph::CheckAndMappingExternalParam(params, common),
                                  common);
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};
    std::vector<float> vectors{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
    REQUIRE(index.Build(make_dataset(ids, vectors, 0, 6, 2)).empty());
    index.fail_candidate_search = true;
    REQUIRE_THROWS_AS(index.Remove({ids[0]}, vsag::RemoveMode::FORCE_REMOVE), std::bad_alloc);
    index.fail_candidate_search = false;
    REQUIRE(index.candidate_searches > 0);
    // Physical deletion is not rolled back. Queries must fall back to the repaired HGraph,
    // never read the old CSR with moved IDs, even if the caller subsequently flushes it.
    REQUIRE(index.GetNumElements() == 5);
    for (uint64_t pass = 0; pass < 2; ++pass) {
        REQUIRE_FALSE(vsag::JsonType::Parse(index.GetStats())["mci_has_index"].GetBool());
        auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
        auto result = index.vsag::HGraph::KnnSearch(
            make_dataset(ids, vectors, 5, 1, 2),
            5,
            R"({"hgraph":{"ef_search":100,"use_mci":true,"hgraph_valid_ratio_threshold":1}})",
            filter);
        REQUIRE(result->GetDim() == 5);
        REQUIRE(result->GetIds()[0] == ids[5]);
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            REQUIRE(result->GetIds()[i] != ids[0]);
        }
        index.Flush();
    }
}

TEST_CASE("HGraph companion MCI serializes concurrent initial Add", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t batch_count = 24;
    constexpr int64_t total = batch_count * 2;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 6000);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = static_cast<float>((i * 17) % 101);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    auto add_batch = [&](int64_t offset) {
        ready.fetch_add(1, std::memory_order_release);
        while (not start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return index.value()->Add(make_dataset(ids, vectors, offset, batch_count, dim));
    };
    auto first = std::async(std::launch::async, add_batch, 0);
    auto second = std::async(std::launch::async, add_batch, batch_count);
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    REQUIRE(first.get().has_value());
    REQUIRE(second.get().has_value());
    REQUIRE(index.value()->GetNumElements() == total);
    const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
}

TEST_CASE("HGraph MCI fast search survives mutations flush and serialization",
          "[ut][hgraph][mci][flush]") {
    constexpr int64_t total = 32;
    constexpr int64_t dim = 4;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 100);
    std::vector<float> vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = static_cast<float>((i * 17) % 101);
    }
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["index_param"]["base_io_type"].SetString("memory_io");
    params["index_param"]["mci_delete_clique_size_threshold"].SetInt(5);
    auto created = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index->Flush().has_value());
    REQUIRE(index->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto query = make_dataset(ids, vectors, 0, 1, dim);
    const auto search_params = R"({"hgraph":{"ef_search":64,"mci_seed_ratio":100,
        "hgraph_valid_ratio_threshold":1.0}})";
    auto verify = [&](const vsag::IndexPtr& target, bool removed) {
        auto result = target->KnnSearch(query, total, search_params, filter);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetStatistics({"mci_raw_float_csr"})[0] == "true");
        REQUIRE(result.value()->GetDim() == total - (removed ? 1 : 0));
        std::vector<int64_t> found(result.value()->GetIds(),
                                   result.value()->GetIds() + result.value()->GetDim());
        std::sort(found.begin(), found.end());
        auto expected = ids;
        if (removed) {
            expected.erase(expected.begin());
        }
        REQUIRE(found == expected);
        return result.value();
    };
    verify(index, false);
    REQUIRE(index->Remove({ids[0]}).value() == 1);
    verify(index, true);
    REQUIRE(index->Add(make_dataset(ids, vectors, 0, 1, dim)).has_value());
    auto before = verify(index, false);
    REQUIRE(index->Flush().has_value());
    auto after = verify(index, false);
    for (int64_t i = 0; i < before->GetDim(); ++i) {
        REQUIRE(before->GetIds()[i] == after->GetIds()[i]);
        REQUIRE(before->GetDistances()[i] == after->GetDistances()[i]);
    }
    auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    REQUIRE(stats["mci_retired_clique_count"].GetInt() == 0);
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 1);
    REQUIRE(index->Flush().has_value());
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(serialized.value()).has_value());
    verify(restored.value(), false);
    REQUIRE(restored.value()->Remove({ids[0]}).value() == 1);
    verify(restored.value(), true);
    REQUIRE(restored.value()->Flush().has_value());
    verify(restored.value(), true);

    std::atomic<bool> start{false};
    auto searches = std::async(std::launch::async, [&]() {
        while (not start.load()) {
            std::this_thread::yield();
        }
        for (uint64_t i = 0; i < 50; ++i) {
            auto result = index->KnnSearch(query, total, search_params, filter);
            if (not result.has_value() or result.value()->GetDim() != total) {
                return false;
            }
        }
        return true;
    });
    start.store(true);
    for (uint64_t i = 0; i < 20; ++i) {
        REQUIRE(index->Flush().has_value());
    }
    REQUIRE(searches.get());

    auto concurrent_queries = std::async(std::launch::async, [&]() {
        for (uint64_t i = 0; i < 200; ++i) {
            auto result = index->KnnSearch(query, total, search_params, filter);
            if (not result.has_value()) {
                return result.error().message;
            }
            const auto count = result.value()->GetDim();
            // An in-flight mutation unpublishes MCI and may fall back to approximate HGraph
            // traversal, which can be empty with a deleted entry point. Check result safety
            // here, and complete coverage after mutations finish.
            if (count < 0 or count > total) {
                return std::string("unexpected count: ") + std::to_string(count) +
                       " route: " + result.value()->GetStatistics({"mci_hybrid_route"})[0];
            }
            if (count == 0) {
                continue;
            }
            std::vector<int64_t> labels(result.value()->GetIds(), result.value()->GetIds() + count);
            std::sort(labels.begin(), labels.end());
            if (std::adjacent_find(labels.begin(), labels.end()) != labels.end()) {
                return std::string("duplicate labels");
            }
            if (not std::includes(ids.begin(), ids.end(), labels.begin(), labels.end())) {
                return std::string("unknown labels");
            }
        }
        return std::string();
    });
    for (uint64_t i = 0; i < 10; ++i) {
        REQUIRE(index->Remove({ids[0]}).value() == 1);
        REQUIRE(index->Add(make_dataset(ids, vectors, 0, 1, dim)).has_value());
        REQUIRE(index->Flush().has_value());
    }
    const auto concurrent_error = concurrent_queries.get();
    INFO(concurrent_error);
    REQUIRE(concurrent_error.empty());
    verify(index, false);
}

TEST_CASE("HGraph MCI physical deletion compacts slots and preserves mixed mutations",
          "[ut][hgraph][mci][force_remove]") {
    constexpr int64_t total = 48;
    constexpr int64_t dim = 8;
    const bool int8 = GENERATE(false, true);
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 100);
    std::vector<float> vectors(total * dim);
    std::vector<int8_t> int8_vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = int8_vectors[i] = static_cast<int8_t>((i * 17) % 101);
    }
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["dtype"].SetString(int8 ? "int8" : "float32");
    if (int8) {
        params["index_param"]["base_quantization_type"].SetString("int8");
    }
    params["index_param"]["graph_type"].SetString("nsw");
    params["index_param"]["base_io_type"].SetString("memory_io");
    params["index_param"]["support_force_remove"].SetBool(true);
    params["index_param"]["build_thread_count"].SetInt(2);
    params["index_param"]["mci_delete_clique_size_threshold"].SetInt(4);
    auto created = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(created.has_value());
    auto index = created.value();
    auto dataset = [&](int64_t begin, int64_t count) {
        auto data = make_dataset(ids, vectors, begin, count, dim);
        if (int8) {
            data->Int8Vectors(int8_vectors.data() + begin * dim);
        }
        return data;
    };
    REQUIRE(index->Build(dataset(0, total - 4)).has_value());
    REQUIRE(index->Add(dataset(total - 4, 4)).has_value());
    REQUIRE(index->Remove({ids[2], ids[47]}).value() == 2);
    const auto marked = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(marked["mci_total_nodes"].GetInt() == total);
    REQUIRE(marked["mci_inactive_node_count"].GetInt() == 2);
    auto memory_before = index->GetMemoryUsage();
    REQUIRE(index->Remove({ids[0], ids[47], ids[46], ids[0], -1}, vsag::RemoveMode::FORCE_REMOVE)
                .value() == 3);
    REQUIRE(index->GetMemoryUsage() < memory_before);
    auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["mci_total_nodes"].GetInt() == 45);
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 1);
    REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    REQUIRE(index->GetNumElements() == 44);
    std::vector<int64_t> active;
    for (auto id : ids) {
        if (id != ids[0] and id != ids[2] and id != ids[46] and id != ids[47]) {
            active.push_back(id);
        }
    }
    auto query = dataset(1, 1);
    const auto search_params = R"({"hgraph":{"ef_search":64,"mci_seed_ratio":100,
        "hgraph_valid_ratio_threshold":1.0}})";
    auto verify = [&]() {
        auto filter = std::make_shared<HalfRatioAllValidFilter>(active);
        auto result = index->KnnSearch(query, total, search_params, filter);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == static_cast<int64_t>(active.size()));
        REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
        if (not int8) {
            REQUIRE(result.value()->GetStatistics({"mci_raw_float_csr"})[0] == "true");
        }
        std::vector<int64_t> found;
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            const auto label = result.value()->GetIds()[i];
            REQUIRE(std::find(active.begin(), active.end(), label) != active.end());
            float distance = 0.0F;
            for (int64_t j = 0; j < dim; ++j) {
                const auto delta = vectors[(label - 100) * dim + j] - vectors[dim + j];
                distance += delta * delta;
            }
            REQUIRE(result.value()->GetDistances()[i] == distance);
            found.push_back(label);
        }
        std::sort(found.begin(), found.end());
        auto expected = active;
        std::sort(expected.begin(), expected.end());
        REQUIRE(found == expected);
    };
    verify();
    REQUIRE(index->Flush().has_value());
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(serialized.value()).has_value());
    index = restored.value();
    REQUIRE(index->GetNumElements() == 44);
    REQUIRE(index->GetNumberRemoved() == 1);
    verify();
    REQUIRE(index->Remove({ids[2]}, vsag::RemoveMode::FORCE_REMOVE).value() == 1);
    verify();
    REQUIRE(index->Add(dataset(46, 2)).has_value());
    active.push_back(ids[46]);
    active.push_back(ids[47]);
    verify();
    REQUIRE(index->Remove(active, vsag::RemoveMode::FORCE_REMOVE).value() == active.size());
    REQUIRE(index->GetNumElements() == 0);
    stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["mci_total_nodes"].GetInt() == 0);
    REQUIRE(stats["mci_inactive_node_count"].GetInt() == 0);
    REQUIRE(index->Remove({-1}, vsag::RemoveMode::FORCE_REMOVE).value() == 0);
    REQUIRE(index->Add(dataset(0, total)).has_value());
    active = ids;
    verify();
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto concurrent_search = std::async(std::launch::async, [&]() {
        for (uint64_t i = 0; i < 100; ++i) {
            auto result = index->KnnSearch(query, total, search_params, filter);
            if (not result.has_value()) {
                return false;
            }
            for (int64_t j = 0; j < result.value()->GetDim(); ++j) {
                const auto label = result.value()->GetIds()[j];
                if (label < ids.front() or label > ids.back()) {
                    return false;
                }
                float expected = 0.0F;
                for (int64_t d = 0; d < dim; ++d) {
                    const auto delta = vectors[(label - 100) * dim + d] - vectors[dim + d];
                    expected += delta * delta;
                }
                if (result.value()->GetDistances()[j] != expected) {
                    return false;
                }
            }
        }
        return true;
    });
    for (uint64_t i = 0; i < 8; ++i) {
        REQUIRE(index->Remove({ids[0]}, vsag::RemoveMode::FORCE_REMOVE).value() == 1);
        REQUIRE(index->Add(dataset(0, 1)).has_value());
        REQUIRE(index->Flush().has_value());
    }
    REQUIRE(concurrent_search.get());
    verify();
}

TEST_CASE("HGraph FP32 MCI fresh snapshots support physical mutation after reload",
          "[ut][hgraph][mci][force_remove][reload_remove]") {
    const bool stream_format = GENERATE(false, true);
    const std::string io = GENERATE("memory_io", "block_memory_io");
    CAPTURE(stream_format, io);
    constexpr int64_t total = 400;
    constexpr int64_t dim = 16;
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);
    auto vectors = fixtures::generate_vectors(total, dim);
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["metric_type"].SetString("cosine");
    auto build = params["index_param"];
    build["base_quantization_type"].SetString("fp32");
    build["base_io_type"].SetString(io);
    build["graph_type"].SetString("nsw");
    build["max_degree"].SetInt(32);
    build["ef_construction"].SetInt(200);
    build["build_thread_count"].SetInt(2);
    build["support_force_remove"].SetBool(true);
    build["mci_mcs"].SetInt(50);
    build["mci_clique_max"].SetInt(50);
    build["mci_incremental_clique_max"].SetInt(50);
    auto source = vsag::Factory::CreateIndex("hgraph", params.Dump()).value();
    REQUIRE(source->Build(make_dataset(ids, vectors, 0, total, dim)).value().empty());
    auto restored = vsag::Factory::CreateIndex("hgraph", params.Dump()).value();
    if (stream_format) {
        std::stringstream stream;
        REQUIRE(source->Serialize(stream).has_value());
        REQUIRE(restored->Deserialize(stream).has_value());
    } else {
        auto binary = source->Serialize();
        REQUIRE(binary.has_value());
        REQUIRE(restored->Deserialize(binary.value()).has_value());
    }
    // Exercise the un-compacted initial snapshot, not one already shrunk by FORCE_REMOVE.
    source.reset();
    for (int64_t stage = 0; stage < 2; ++stage) {
        std::vector<int64_t> removed(ids.begin() + stage * 40, ids.begin() + (stage + 1) * 40);
        REQUIRE(restored->Remove(removed, vsag::RemoveMode::FORCE_REMOVE).value() == 40);
        REQUIRE(restored->GetNumElements() == total - (stage + 1) * 40);
        const auto stats = vsag::JsonType::Parse(restored->GetStats());
        REQUIRE(stats["mci_total_nodes"].GetInt() == restored->GetNumElements());
        REQUIRE(stats["mci_covered_nodes"].GetInt() == restored->GetNumElements());
    }
    for (int64_t stage = 0; stage < 2; ++stage) {
        REQUIRE(restored->Add(make_dataset(ids, vectors, stage * 40, 40, dim)).value().empty());
        REQUIRE(restored->GetNumElements() == total - 40 + stage * 40);
    }
    REQUIRE(restored->Flush().has_value());
    for (const bool use_mci : {false, true}) {
        auto search = vsag::JsonType::Parse(R"({"hgraph":{"ef_search":400,
            "mci_seed_ratio":100,"hgraph_valid_ratio_threshold":1.0}})");
        search["hgraph"]["use_mci"].SetBool(use_mci);
        auto result = restored->KnnSearch(make_dataset(ids, vectors, 100, 1, dim),
                                          10,
                                          search.Dump(),
                                          std::make_shared<HalfRatioAllValidFilter>(ids));
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 10);
        REQUIRE(result.value()->GetIds()[0] == ids[100]);
    }
}

TEST_CASE("HGraph MCI force remove handles marked entry points and shadowed labels",
          "[ut][hgraph][mci][force_remove]") {
    std::vector<int64_t> ids{100, 101, 102, 103, 104, 105};
    std::vector<float> vectors{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(2));
    params["index_param"]["graph_type"].SetString("nsw");
    params["index_param"]["base_io_type"].SetString("memory_io");
    params["index_param"]["support_force_remove"].SetBool(true);
    params["index_param"]["build_thread_count"].SetInt(2);
    auto created = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index->Build(make_dataset(ids, vectors, 0, 6, 2)).has_value());

    SECTION("Entry point replacement skips soft-deleted nodes") {
        REQUIRE(index->Remove({100, 101}).value() == 2);
        REQUIRE(index->Remove({100}, vsag::RemoveMode::FORCE_REMOVE).value() == 1);
        auto result = index->KnnSearch(make_dataset(ids, vectors, 2, 1, 2),
                                       1,
                                       R"({"hgraph":{"ef_search":64,"use_mci":false}})");
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 1);
        REQUIRE(result.value()->GetIds()[0] == 102);
    }

    SECTION("Moving an old tombstone cannot hide the readded live label") {
        REQUIRE(index->Remove({104}).value() == 1);
        REQUIRE(index->Add(make_dataset(ids, vectors, 4, 1, 2)).has_value());
        for (auto label : {100, 105, 101}) {
            REQUIRE(index->Remove({label}, vsag::RemoveMode::FORCE_REMOVE).value() == 1);
        }
        REQUIRE(index->GetNumElements() == 3);
        REQUIRE(index->GetNumberRemoved() == 1);
        REQUIRE(index->CheckIdExist(104));
        auto blob = index->Serialize();
        REQUIRE(blob.has_value());
        auto restored = vsag::Factory::CreateIndex("hgraph", params.Dump());
        REQUIRE(restored.has_value());
        REQUIRE(restored.value()->Deserialize(blob.value()).has_value());
        REQUIRE(restored.value()->GetNumElements() == 3);
        REQUIRE(restored.value()->CheckIdExist(104));
        auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
        auto result = restored.value()->KnnSearch(
            make_dataset(ids, vectors, 4, 1, 2),
            1,
            R"({"hgraph":{"ef_search":64,"mci_seed_ratio":100,"hgraph_valid_ratio_threshold":1}})",
            filter);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 1);
        REQUIRE(result.value()->GetIds()[0] == 104);
        REQUIRE(result.value()->GetDistances()[0] == 0.0F);
    }
}

TEST_CASE("HGraph companion MCI incrementally updates cliques after Add", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 24;
    constexpr int64_t add_count = 4;
    constexpr int64_t total = base_count + add_count;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);

    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(make_dataset(ids, vectors, 0, base_count, dim));
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index.value()->GetMemoryUsageDetail().count("mci_cliques") == 1);
    std::stringstream streaming;
    REQUIRE_FALSE(index.value()->SerializeStreaming(streaming).has_value());
    const auto memory_before_add = index.value()->GetMemoryUsage();

    auto add_result = index.value()->Add(make_dataset(ids, vectors, base_count, add_count, dim));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index.value()->GetMemoryUsage() > memory_before_add);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(vectors.data() + (base_count + 1) * dim)
        ->Owner(false);

    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.5,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    const auto search_statistics = vsag::JsonType::Parse(result.value()->GetStatistics());
    REQUIRE(search_statistics["distance_evaluations_by_phase"]["approximate"].GetUint64() > 0);
    REQUIRE(search_statistics["distance_evaluations"].GetUint64() ==
            search_statistics["dist_cmp"].GetUint64());
    REQUIRE(search_statistics["distance_evaluations_by_backend"]["fp32"].GetUint64() ==
            search_statistics["distance_evaluations"].GetUint64());
    const auto expected_seed_count =
        static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(total)) * 0.5));
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) ==
            expected_seed_count);

    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0,"timeout_ms":0}})",
                                 filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"is_timeout"})[0] == "true");

    const auto stats = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(stats["mci_max_clique_size"].GetInt() > 0);

    auto callback_filter = std::make_shared<CallbackOnlyFilter>();
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 callback_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("disabled")");

    auto invalid = vsag::Bitset::Make();
    for (int64_t i = 0; i < total; i += 2) {
        invalid->Set(ids[i]);
    }
    result = index.value()->KnnSearch(
        query,
        3,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":100.0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        invalid);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == total / 2);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE_FALSE(invalid->Test(result.value()->GetIds()[i]));
    }

    auto all_invalid = vsag::Bitset::Make();
    for (auto id : ids) {
        all_invalid->Set(id);
    }
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 all_invalid);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");

    const std::vector<int64_t> mixed_ids{9000, ids[base_count + 1]};
    auto mixed_filter = std::make_shared<HalfRatioAllValidFilter>(mixed_ids);
    result =
        index.value()->KnnSearch(query,
                                 1,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":0.1,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 mixed_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    REQUIRE(result.value()->GetIds()[0] == ids[base_count + 1]);
    REQUIRE(std::stoull(result.value()->GetStatistics({"mci_seed_count"})[0]) == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");

    const std::vector<int64_t> stale_ids{9000, 9001, 9002};
    auto stale_filter = std::make_shared<CountingValidIdsFilter>(stale_ids);
    result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 stale_filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("hgraph")");
    REQUIRE(stale_filter->CheckCount() > 0);
}

TEST_CASE("HGraph cache-accelerated NSW build creates MCI companion", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 16;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 3000);
    std::vector<float> vectors(count * dim, 0.0F);
    std::vector<std::string> source_ids(count);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 3);
        source_ids[i] = "source-" + std::to_string(i);
    }
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(count)
                       ->Dim(dim)
                       ->Ids(ids.data())
                       ->Float32Vectors(vectors.data())
                       ->SourceID(source_ids.data())
                       ->Owner(false);
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["index_param"]["graph_type"].SetString("nsw");

    auto source_index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(source_index.has_value());
    auto result = source_index.value()->Build(dataset);
    REQUIRE(result.has_value());
    std::stringstream cache;
    REQUIRE(source_index.value()->ExportCache(cache).has_value());

    auto cached_index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(cached_index.has_value());
    REQUIRE(cached_index.value()->ImportCache(cache).has_value());
    result = cached_index.value()->Build(dataset);
    REQUIRE(result.has_value());
    const auto stats = vsag::JsonType::Parse(cached_index.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());
}

TEST_CASE("HGraph companion MCI incrementally adds INT8 vectors", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 12;
    constexpr int64_t add_count = 2;
    constexpr int64_t total = base_count + add_count;

    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    params["dtype"].SetString("int8");
    params["index_param"]["base_quantization_type"].SetString("int8");
    auto index = vsag::Factory::CreateIndex("hgraph", params.Dump());
    REQUIRE(index.has_value());

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 2000);
    std::vector<int8_t> vectors(total * dim);
    for (int64_t i = 0; i < total * dim; ++i) {
        vectors[i] = static_cast<int8_t>((i * 7) % 31);
    }
    auto make_int8_dataset = [&](int64_t offset, int64_t count) {
        return vsag::Dataset::Make()
            ->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data() + offset)
            ->Int8Vectors(vectors.data() + offset * dim)
            ->Owner(false);
    };

    auto result = index.value()->Build(make_int8_dataset(0, base_count));
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    result = index.value()->Add(make_int8_dataset(base_count, add_count));
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    REQUIRE(index.value()->GetNumElements() == total);

    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Int8Vectors(vectors.data())->Owner(false);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto search_result =
        index.value()->KnnSearch(query,
                                 3,
                                 R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
                                 R"("hgraph_valid_ratio_threshold":1.0}})",
                                 filter);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
    REQUIRE(std::stoull(search_result.value()->GetStatistics({"dist_cmp"})[0]) > 0);
    REQUIRE(std::stoull(search_result.value()->GetStatistics({"hops"})[0]) > 0);
}

TEST_CASE("HGraph analyzer sends only typed INT8 query buffers", "[ut][hgraph][mci]") {
    constexpr uint64_t dim = 4;
    constexpr uint64_t total = 12;
    auto config = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    auto params = config["index_param"];
    params["base_quantization_type"].SetString("int8");
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.data_type_ = vsag::DataTypes::DATA_TYPE_INT8;
    Int8QueryCheckingHGraph index(vsag::HGraph::CheckAndMappingExternalParam(params, common),
                                  common);
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 2000);
    index.expected_vectors.resize(total * dim);
    for (uint64_t i = 0; i < total * dim; ++i) {
        index.expected_vectors[i] = static_cast<int8_t>(static_cast<int>(i * 7 % 31) - 15);
    }
    auto dataset = vsag::Dataset::Make()
                       ->NumElements(total)
                       ->Dim(dim)
                       ->Ids(ids.data())
                       ->Int8Vectors(index.expected_vectors.data())
                       ->Owner(false);
    REQUIRE(index.Build(dataset).empty());
    index.queries = 0;
    vsag::AnalyzerParam analyzer_params(common.allocator_.get());
    analyzer_params.base_sample_size = total;
    analyzer_params.topk = 3;
    vsag::HGraphAnalyzer analyzer(&index, analyzer_params);
    REQUIRE(std::isfinite(analyzer.GetBaseSearchRecall(R"({"hgraph":{"ef_search":100}})")));
    REQUIRE(std::isfinite(analyzer.GetBaseSearchTimeCost(R"({"hgraph":{"ef_search":100}})")));
    REQUIRE(index.queries == total);
}

TEST_CASE("MCI final coverage deduplicates wide neighborhoods in input order",
          "[ut][hgraph][mci][shared_build][coverage_repair]") {
    vsag::DefaultAllocator allocator;
    constexpr uint64_t total = 512;
    constexpr uint64_t degree = 256;
    std::vector<std::atomic<int>> coverage(total);
    for (auto& count : coverage) {
        count.store(1);
    }
    coverage.back().store(0);
    uint64_t visited = 0;
    auto visit_neighbors = [&](vsag::InnerIdType seed, auto visitor) {
        REQUIRE(seed == total - 1);
        REQUIRE(visitor(seed));
        REQUIRE(visitor(total));
        for (vsag::InnerIdType id = total - 2; id > 0; --id) {
            ++visited;
            if (not visitor(id)) {
                break;
            }
            REQUIRE(visitor(id));
        }
    };
    vsag::Vector<vsag::Vector<vsag::InnerIdType>> cliques(&allocator);
    REQUIRE(vsag::EnsureMCICliqueCoverage(
                total, degree, 100, coverage, visit_neighbors, cliques, &allocator) == 1);
    REQUIRE(visited == degree - 1);
    REQUIRE(cliques.size() == 1);
    REQUIRE(cliques[0].size() == 100);
    for (uint64_t i = 0; i < 99; ++i) {
        REQUIRE(cliques[0][i] == total - degree + i);
    }
    REQUIRE(cliques[0].back() == total - 1);
    for (uint64_t id = 0; id < total; ++id) {
        const bool added = id >= total - degree and id < total - degree + 99;
        REQUIRE(coverage[id].load() == 1 + static_cast<int>(added));
    }
}

TEMPLATE_TEST_CASE("MCI final coverage pass preserves seeds and stored membership counts",
                   "[ut][hgraph][mci][shared_build][coverage_repair]",
                   int,
                   uint32_t) {
    vsag::DefaultAllocator allocator;
    constexpr uint64_t total = 8;
    std::vector<std::atomic<TestType>> coverage(total);
    for (auto& count : coverage) {
        count.store(1);
    }
    coverage[6].store(0);
    coverage[7].store(0);
    vsag::Vector<vsag::Vector<vsag::InnerIdType>> cliques(&allocator);
    cliques.emplace_back(&allocator);
    cliques.back().assign({0, 1, 2, 3, 4, 5});
    uint64_t visited = 0;
    auto visit_neighbors = [&](vsag::InnerIdType seed, auto visitor) {
        ++visited;
        if (seed == 6) {
            // Duplicate/self/invalid neighbors must not consume the candidate budget.
            const vsag::InnerIdType row[]{6, 99, 0, 0, 1, 2, 3, 4, 5};
            for (auto id : row) {
                if (not visitor(id)) {
                    break;
                }
            }
        }
        // Seed 7 is isolated and needs a singleton.
    };
    REQUIRE(vsag::EnsureMCICliqueCoverage(
                total, 5, 3, coverage, visit_neighbors, cliques, &allocator) == 2);
    REQUIRE(visited == 2);
    REQUIRE(cliques.size() == 3);
    // Normalization truncates a high-ID seed, then restores it in the stored clique.
    REQUIRE(std::vector<vsag::InnerIdType>(cliques[1].begin(), cliques[1].end()) ==
            std::vector<vsag::InnerIdType>{0, 1, 6});
    REQUIRE(cliques[2].size() == 1);
    REQUIRE(cliques[2].front() == 7);
    std::vector<uint64_t> actual(total, 0);
    for (const auto& clique : cliques) {
        for (auto id : clique) {
            ++actual[id];
        }
    }
    for (uint64_t id = 0; id < total; ++id) {
        REQUIRE(coverage[id].load() == actual[id]);
        REQUIRE(actual[id] > 0);
    }
    // A second pass must neither read graph rows nor append duplicate cliques.
    REQUIRE(vsag::EnsureMCICliqueCoverage(
                total, 5, 3, coverage, visit_neighbors, cliques, &allocator) == 0);
    REQUIRE(visited == 2);
    REQUIRE(cliques.size() == 3);
}

TEST_CASE("MCI final coverage pass handles empty and singleton graphs",
          "[ut][hgraph][mci][shared_build][coverage_repair]") {
    vsag::DefaultAllocator allocator;
    const uint64_t total = GENERATE(0, 1);
    std::vector<std::atomic<int>> coverage(total);
    for (auto& count : coverage) {
        count.store(0);
    }
    vsag::Vector<vsag::Vector<vsag::InnerIdType>> cliques(&allocator);
    auto visit_neighbors = [](vsag::InnerIdType, auto) { FAIL("unexpected graph access"); };
    REQUIRE(vsag::EnsureMCICliqueCoverage(
                total, 0, 1, coverage, visit_neighbors, cliques, &allocator) == total);
    REQUIRE(cliques.size() == total);
    if (total == 1) {
        REQUIRE(cliques.front().size() == 1);
        REQUIRE(cliques.front().front() == 0);
        REQUIRE(coverage[0].load() == 1);
    }
}

TEST_CASE("MCI full build covers duplicate vectors after alpha exhaustion",
          "[ut][hgraph][mci][shared_build][coverage_repair]") {
    vsag::DefaultAllocator allocator;
    constexpr uint64_t total = 256;
    constexpr uint64_t mcs = 200;
    std::vector<float> vectors(total, 1.0F);
    std::vector<vsag::InnerIdType> rows(total * mcs);
    for (uint64_t seed = 0; seed < total; ++seed) {
        uint64_t count = 0;
        for (vsag::InnerIdType id = 0; count < mcs; ++id) {
            if (id != seed) {
                rows[seed * mcs + count++] = id;
            }
        }
    }
    vsag::MCIGraphView graph;
    graph.neighbors = rows.data();
    graph.total = total;
    graph.row_stride = mcs;
    graph.uniform_count = mcs;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = 1;
    params.candidate_limit = mcs;
    params.clique_max = 50;
    params.thread_count = GENERATE(1, 4);
    params.metric = vsag::MetricType::METRIC_TYPE_COSINE;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    std::vector<bool> covered(total, false);
    for (const auto& clique : cliques) {
        REQUIRE_FALSE(clique.empty());
        REQUIRE(clique.size() <= params.clique_max);
        for (auto id : clique) {
            REQUIRE(id < total);
            covered[id] = true;
        }
    }
    REQUIRE(std::count(covered.begin(), covered.end(), true) == total);
}

TEST_CASE("MCI shared local builder expands past a two-node clique",
          "[ut][hgraph][mci][shared_build]") {
    vsag::DefaultAllocator allocator;
    vsag::MCIV3BuildParams params;
    params.total = 3;
    params.dim = 1;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;
    vsag::MCILocalCliqueBuilder builder(params, &allocator);
    std::vector<std::atomic<int>> coverage(3);
    coverage[0].store(0);
    coverage[1].store(1);
    coverage[2].store(1);
    const vsag::InnerIdType neighbors[]{1, 2};
    auto distance = [](vsag::InnerIdType lhs, vsag::InnerIdType rhs) {
        const float delta = static_cast<float>(lhs) - static_cast<float>(rhs);
        return delta * delta;
    };
    auto batch = [&](auto lhs, auto a, auto b, auto c, auto d, float* values) {
        const vsag::InnerIdType ids[]{a, b, c, d};
        for (uint64_t i = 0; i < 4; ++i) {
            values[i] = distance(lhs, ids[i]);
        }
    };
    std::vector<std::vector<vsag::InnerIdType>> emitted;
    auto emit = [&](const auto& clique) { emitted.emplace_back(clique.begin(), clique.end()); };
    builder.Build(0, neighbors, 2, 1.2F, coverage, distance, batch, emit);
    REQUIRE(emitted.empty());
    REQUIRE(coverage[0].load() == 0);
    builder.Build(0, neighbors, 2, 4.8F, coverage, distance, batch, emit);
    REQUIRE(emitted.size() == 1);
    std::sort(emitted[0].begin(), emitted[0].end());
    REQUIRE(emitted[0] == std::vector<vsag::InnerIdType>{0, 1, 2});
    REQUIRE(coverage[0].load() == 1);
    REQUIRE(coverage[1].load() == 2);
    REQUIRE(coverage[2].load() == 2);

    const std::vector<float> vectors{0.0F, 1.0F, 2.0F};
    const std::vector<vsag::InnerIdType> rows{1, 2, 0, 2, 0, 1};
    vsag::MCIGraphView graph;
    graph.neighbors = rows.data();
    graph.total = 3;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    const auto full = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(full.size() == 1);
    auto members = full[0];
    std::sort(members.begin(), members.end());
    REQUIRE(std::equal(members.begin(), members.end(), emitted[0].begin(), emitted[0].end()));
}

TEST_CASE("MCI shared builder fallback counts only stored members",
          "[ut][hgraph][mci][shared_build]") {
    vsag::DefaultAllocator allocator;
    vsag::MCIV3BuildParams params;
    params.total = 6;
    params.candidate_limit = 5;
    params.clique_max = 3;
    vsag::MCILocalCliqueBuilder builder(params, &allocator);
    std::vector<std::atomic<int>> coverage(6);
    for (auto& count : coverage) {
        count.store(1);
    }
    coverage[0].store(0);
    // All distances vanish: the unchanged strict seed-edge bound triggers high-alpha fallback.
    auto distance = [](auto, auto) { return 0.0F; };
    uint64_t batches = 0;
    auto batch = [&](auto, auto, auto, auto, auto, float* values) {
        ++batches;
        std::fill_n(values, 4, 0.0F);
    };
    const vsag::InnerIdType neighbors[]{1, 2, 3, 4, 5};
    std::vector<vsag::InnerIdType> stored;
    builder.Build(0, neighbors, 5, 101.0F, coverage, distance, batch, [&](const auto& clique) {
        stored.assign(clique.begin(), clique.end());
    });
    REQUIRE(batches > 0);
    REQUIRE(stored.size() == 3);
    REQUIRE(stored.front() == 0);
    for (uint64_t i = 0; i < coverage.size(); ++i) {
        const int before = i == 0 ? 0 : 1;
        const bool present = std::find(stored.begin(), stored.end(), i) != stored.end();
        REQUIRE(coverage[i].load() == before + static_cast<int>(present));
    }

    for (auto& count : coverage) {
        count.store(1);
    }
    coverage[0].store(0);
    // Duplicates, the seed and out-of-range IDs do not count toward the threshold.
    const vsag::InnerIdType sparse_neighbors[]{1, 1, 0, 99};
    stored.clear();
    builder.Build(
        0, sparse_neighbors, 4, 100.0F, coverage, distance, batch, [&](const auto& clique) {
            stored.assign(clique.begin(), clique.end());
        });
    REQUIRE(stored.empty());
    builder.Build(
        0, sparse_neighbors, 4, 101.0F, coverage, distance, batch, [&](const auto& clique) {
            stored.assign(clique.begin(), clique.end());
        });
    REQUIRE(stored == std::vector<vsag::InnerIdType>{0, 1});
    REQUIRE(coverage[0].load() == 1);
    REQUIRE(coverage[1].load() == 2);
    REQUIRE(coverage[2].load() == 1);
}

TEST_CASE("HGraph Add uses the full-build local clique size threshold",
          "[ut][hgraph][mci][shared_build]") {
    const std::string io = GENERATE("memory_io", "block_memory_io");
    constexpr int64_t dim = 4;
    auto config = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    auto params = config["index_param"];
    params["graph_type"].SetString("nsw");
    params["base_io_type"].SetString(io);
    params["build_thread_count"].SetInt(1);
    params["mci_mcs"].SetInt(3);
    params["mci_clique_max"].SetInt(3);
    params["mci_incremental_clique_max"].SetInt(3);
    auto index = vsag::Factory::CreateIndex("hgraph", config.Dump());
    REQUIRE(index.has_value());
    std::vector<int64_t> ids{10, 20, 30, 40};
    std::vector<float> vectors{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0};
    REQUIRE(index.value()->Build(make_dataset(ids, vectors, 0, 3, dim)).has_value());
    auto before = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(before["mci_total_clique_count"].GetInt() == 1);
    REQUIRE(before["mci_max_clique_size"].GetInt() == 3);
    // The existing clique is full. At alpha=1.2, the old greedy branch emitted only {0,1}.
    REQUIRE(index.value()->Add(make_dataset(ids, vectors, 3, 1, dim)).has_value());
    auto after = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(after["mci_covered_nodes"].GetInt() == 4);
    // The shared builder first emits three members, then adds the remaining neighbor in
    // a two-member clique to reach min(50, N-1)=3 distinct neighbors.
    REQUIRE(after["mci_delta_clique_count"].GetInt() == 2);
    REQUIRE(after["mci_delta_clique_membership_count"].GetInt() == 5);
    REQUIRE(index.value()->Flush().has_value());
    auto flushed = vsag::JsonType::Parse(index.value()->GetStats());
    REQUIRE(flushed["mci_delta_clique_count"].GetInt() == 0);
    REQUIRE(flushed["mci_covered_nodes"].GetInt() == 4);
    vsag::DefaultAllocator allocator;
    REQUIRE(snapshot_degree(read_flushed_mci(index.value(), &allocator), 3) == 3);
}

TEST_CASE("MCI Add stops at unique degree rather than clique count",
          "[ut][hgraph][mci][degree_add]") {
    const std::string scenario =
        GENERATE("join", "mixed", "build", "default", "configured", "exhausted", "marked");
    const std::string io = GENERATE("memory_io", "block_memory_io");
    CAPTURE(scenario, io);
    const bool large_index = scenario == "default" or scenario == "configured";
    const int64_t total = large_index ? 101 : 9;
    const int64_t dim = total;
    std::vector<int64_t> ids(total + 1);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<float> vectors((total + 1) * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim + i] = 1.0F;
    }
    auto configuration = vsag::JsonType::Parse(generate_hgraph_mci_params(dim));
    auto parameters = configuration["index_param"];
    parameters["graph_type"].SetString("nsw");
    parameters["max_degree"].SetInt(large_index ? 128 : 32);
    parameters["base_io_type"].SetString(io);
    parameters["build_thread_count"].SetInt(1);
    parameters["mci_mcs"].SetInt(large_index ? 200 : total);
    parameters["mci_clique_max"].SetInt(3);
    parameters["mci_incremental_clique_max"].SetInt(3);
    parameters["mci_incremental_join_ratio_threshold"].SetFloat(1.0F);
    if (scenario == "configured") {
        parameters["mci_incremental_degree_min"].SetInt(100);
    }
    if (not large_index) {
        parameters["mci_incremental_degree_n_divisor"].SetInt(scenario == "marked" ? 2 : 1);
        parameters["mci_incremental_degree_mcs_divisor"].SetInt(1);
    }
    const auto json = configuration.Dump();
    auto built = vsag::Factory::CreateIndex("hgraph", json);
    REQUIRE(built.has_value());
    REQUIRE(built.value()->Build(make_dataset(ids, vectors, 0, total, dim)).has_value());
    std::vector<std::vector<vsag::InnerIdType>> cliques;
    if (scenario == "build") {
        cliques = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};  // All existing cliques are full.
    } else if (scenario == "mixed") {
        cliques = {{0, 1}, {2, 3, 4}, {5, 6, 7}, {6, 7, 8}};
    } else if (scenario == "exhausted") {
        // Saturated memberships exclude every candidate from local construction.
        for (uint64_t repeat = 0; repeat < 3; ++repeat) {
            cliques.insert(cliques.end(), {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}});
        }
    } else {
        // The duplicate clique must be skipped because it contributes no new neighbor.
        cliques.push_back({0, 1});
        for (vsag::InnerIdType id = 1; id < total; ++id) {
            cliques.push_back(
                {0, id});  // Strong overlap: k joined cliques give only k+1 neighbors.
        }
    }
    auto index = with_mci_fixture(built.value(), json, cliques);
    REQUIRE(vsag::JsonType::Parse(index->GetStats())["mci_base_clique_count"].GetInt() ==
            static_cast<int64_t>(cliques.size()));
    if (scenario == "marked") {
        REQUIRE(index->Remove({ids[total - 1]}, vsag::RemoveMode::MARK_REMOVE).value() == 1);
    }
    REQUIRE(index->Add(make_dataset(ids, vectors, total, 1, dim)).has_value());
    const auto stats = vsag::JsonType::Parse(index->GetStats());
    INFO(stats.Dump());
    REQUIRE(stats["mci_covered_nodes"].GetInt() == total + (scenario == "marked" ? 0 : 1));
    if (scenario == "join") {
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 8);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    } else if (scenario == "mixed") {
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 1);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() > 0);
    } else if (scenario == "build") {
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 0);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() >= 5);
    } else if (scenario == "default") {
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 49);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    } else if (scenario == "configured") {
        // Target 100 requires 99 overlapping two-member cliques, not just the first JOIN.
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 99);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    } else if (scenario == "marked") {
        // The degree floor is capped at eight other live points, excluding the mark.
        REQUIRE(index->GetNumElements() == total);
        REQUIRE(stats["mci_delta_extra_membership_count"].GetInt() == 7);
        REQUIRE(stats["mci_delta_clique_count"].GetInt() == 0);
    } else {
        REQUIRE(stats["mci_delta_clique_count"].GetInt() == 1);
        REQUIRE(stats["mci_delta_clique_membership_count"].GetInt() == 1);
    }
    vsag::DefaultAllocator allocator;
    auto snapshot = read_flushed_mci(index, &allocator);
    REQUIRE(snapshot.labels[total] == ids[total]);
    const uint64_t expected = scenario == "default"      ? 50
                              : scenario == "configured" ? 100
                              : scenario == "exhausted"  ? 0
                              : scenario == "marked"     ? total - 1
                                                         : total;
    REQUIRE(snapshot_degree(snapshot, total) == expected);
    auto restored = vsag::Factory::CreateIndex("hgraph", json);
    REQUIRE(restored.has_value());
    std::stringstream stream(snapshot.bytes);
    REQUIRE(restored.value()->Deserialize(stream).has_value());
    REQUIRE(snapshot_degree(read_flushed_mci(restored.value(), &allocator), total) == expected);
}

TEST_CASE("MCI local builder relaxes negative similarities without accepting every edge",
          "[ut][hgraph][mci][shared_build]") {
    vsag::DefaultAllocator allocator;
    vsag::MCIV3BuildParams params;
    params.total = 3;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_IP;
    vsag::MCILocalCliqueBuilder builder(params, &allocator);
    const float vectors[3][2]{{1, 0}, {-1, 2}, {-1, -2}};
    auto distance = [&](auto lhs, auto rhs) {
        return 1.0F - vectors[lhs][0] * vectors[rhs][0] - vectors[lhs][1] * vectors[rhs][1];
    };
    auto batch = [&](auto lhs, auto a, auto b, auto c, auto d, float* values) {
        values[0] = distance(lhs, a);
        values[1] = distance(lhs, b);
        values[2] = distance(lhs, c);
        values[3] = distance(lhs, d);
    };
    std::vector<std::atomic<int>> coverage(3);
    coverage[0].store(0);
    coverage[1].store(1);
    coverage[2].store(1);
    const vsag::InnerIdType neighbors[]{1, 2};
    uint64_t emitted = 0;
    auto emit = [&](const auto& clique) {
        REQUIRE(clique.size() == 3);
        ++emitted;
    };
    const auto narrow = builder.Build(0, neighbors, 2, 2.0F, coverage, distance, batch, emit);
    REQUIRE(narrow.edges == 2);
    REQUIRE(emitted == 0);
    REQUIRE(coverage[0].load() == 0);
    const auto wide = builder.Build(0, neighbors, 2, 4.0F, coverage, distance, batch, emit);
    REQUIRE(wide.edges == 3);
    REQUIRE(emitted == 1);
    REQUIRE(coverage[0].load() == 1);
}

TEST_CASE("MCI compact local coverage retains global membership thresholds",
          "[ut][hgraph][mci][shared_build]") {
    vsag::DefaultAllocator allocator;
    vsag::MCIV3BuildParams params;
    params.total = 1000;
    params.candidate_limit = 2;
    params.clique_max = 3;
    vsag::MCILocalCliqueBuilder builder(params, &allocator);
    std::vector<std::atomic<int>> coverage(3);
    const auto memberships = GENERATE(9, 10);
    coverage[0].store(0);
    coverage[1].store(memberships);
    coverage[2].store(1);
    const vsag::InnerIdType neighbors[]{1, 2};
    auto distance = [](auto, auto) { return 1.0F; };
    auto batch = [](auto, auto, auto, auto, auto, float* values) { std::fill_n(values, 4, 1.0F); };
    uint64_t emitted = 0;
    auto stats = builder.Build(
        0, neighbors, 2, 2.0F, coverage, distance, batch, [&](const auto&) { ++emitted; });
    // Nine global memberships exceed the local vector length, but not 1000 / 100.
    REQUIRE(stats.candidates == (memberships == 9 ? 2 : 1));
    REQUIRE(emitted == (memberships == 9 ? 1 : 0));
    REQUIRE(coverage[0].load() == (memberships == 9 ? 1 : 0));
}

TEST_CASE("MCI builder expands negative inner-product distances", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{10.0F, 0.0F, 9.9F, 0.0F, 9.8F, 0.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_IP;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == total; }));
}

TEST_CASE("MCI builder preserves L2 differences for large coordinates", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{1.0e8F, 1.0F, 1.0e8F, 2.0F, 1.0e8F, 100.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(cliques.begin(), cliques.end(), [](const auto& clique) {
        return std::find(clique.begin(), clique.end(), 0) != clique.end() and
               std::find(clique.begin(), clique.end(), 1) != clique.end();
    }));
}

TEST_CASE("MCI builder includes duplicate-vector seed edges", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors(total * dim, 1.0F);
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 1;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == total; }));
}

TEST_CASE("MCI builder treats clique_max only as a storage cap", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 6;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{
        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 10.0F, 10.0F, 11.0F, 10.0F, 10.0F, 11.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1,
                                                   0, 1, 2, 2, 2, 2, 4, 5, 3, 3, 3, 3,
                                                   5, 3, 4, 4, 4, 4, 3, 4, 5, 5, 5, 5};
    const std::vector<uint32_t> counts(total, 2);

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.counts = counts.data();
    graph.total = total;
    graph.row_stride = 6;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 5;
    params.clique_max = 2;
    params.max_degree = 2;
    params.alpha = 2.1F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    vsag::DefaultAllocator allocator;
    const auto cliques = vsag::BuildMCICliques(vectors.data(), graph, params, &allocator);
    REQUIRE_FALSE(cliques.empty());
    REQUIRE(std::all_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() <= 2; }));
    REQUIRE(std::any_of(
        cliques.begin(), cliques.end(), [](const auto& clique) { return clique.size() == 2; }));
}

TEST_CASE("MCI builder transports worker exceptions", "[ut][hgraph][mci]") {
    constexpr uint64_t total = 3;
    constexpr uint64_t dim = 2;
    const std::vector<float> vectors{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<vsag::InnerIdType> neighbors{1, 2, 0, 2, 0, 1};

    vsag::MCIGraphView graph;
    graph.neighbors = neighbors.data();
    graph.total = total;
    graph.row_stride = 2;
    graph.uniform_count = 2;
    vsag::MCIV3BuildParams params;
    params.total = total;
    params.dim = dim;
    params.candidate_limit = 2;
    params.clique_max = 3;
    params.max_degree = 2;
    params.alpha = 1.2F;
    params.thread_count = 2;
    params.metric = vsag::MetricType::METRIC_TYPE_L2SQR;

    WorkerFailAllocator allocator;
    REQUIRE_THROWS_AS(vsag::BuildMCICliques(vectors.data(), graph, params, &allocator),
                      std::bad_alloc);
}

TEST_CASE("MCI runner handles empty and growing local graphs", "[ut][hgraph][mci]") {
    vsag::mci::ccrmce_runner<TestEdge, uint32_t> runner;
    std::vector<std::vector<uint32_t>> cliques(16);
    std::vector<std::atomic<int>> clique_counts(10);
    for (auto& count : clique_counts) {
        count.store(0, std::memory_order_relaxed);
    }

    std::vector<TestEdge> edges;
    REQUIRE(runner.run(edges, cliques, 2, clique_counts, 16) == 0);

    for (uint32_t vertex_count : {5U, 8U, 9U}) {
        edges.clear();
        for (uint32_t lhs = 0; lhs < vertex_count; ++lhs) {
            for (uint32_t rhs = lhs + 1; rhs < vertex_count; ++rhs) {
                edges.push_back({lhs, rhs});
            }
        }
        REQUIRE(runner.run(edges, cliques, 2, clique_counts, 16) > 0);
    }
}

TEST_CASE("MCI runner preserves the P extension in must cliques", "[ut][hgraph][mci]") {
    const std::vector<TestEdge> input_edges{{0, 1},
                                            {0, 2},
                                            {0, 4},
                                            {0, 5},
                                            {1, 3},
                                            {1, 4},
                                            {1, 5},
                                            {2, 3},
                                            {2, 4},
                                            {2, 5},
                                            {3, 4},
                                            {3, 5},
                                            {4, 5}};
    auto edges = input_edges;
    std::set<std::pair<uint32_t, uint32_t>> adjacency;
    for (const auto& edge : input_edges) {
        adjacency.emplace(std::min(edge.u, edge.v), std::max(edge.u, edge.v));
    }

    vsag::mci::ccrmce_runner<TestEdge, uint32_t> runner;
    std::vector<std::vector<uint32_t>> cliques(16);
    std::vector<std::atomic<int>> clique_counts(6);
    for (auto& count : clique_counts) {
        count.store(0, std::memory_order_relaxed);
    }

    const auto clique_count = runner.run(edges, cliques, 4, clique_counts, 16);
    REQUIRE(clique_count > 0);
    bool has_p_extension = false;
    for (uint32_t clique_id = 0; clique_id < clique_count; ++clique_id) {
        const auto& clique = cliques[clique_id];
        for (uint64_t i = 0; i < clique.size(); ++i) {
            for (uint64_t j = i + 1; j < clique.size(); ++j) {
                if (adjacency.count(
                        {std::min(clique[i], clique[j]), std::max(clique[i], clique[j])}) == 0) {
                    has_p_extension = true;
                }
            }
        }
    }
    REQUIRE(has_p_extension);
}

TEST_CASE("Clique base view pins CSR storage", "[ut][hgraph][mci]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    auto assign_clique = [&]() {
        vsag::Vector<vsag::InnerIdType> p_maxc(&allocator);
        vsag::Vector<vsag::InnerIdType> maxcs(&allocator);
        vsag::Vector<vsag::InnerIdType> p_node_to_cid(&allocator);
        vsag::Vector<vsag::InnerIdType> node_to_cids(&allocator);
        p_maxc.insert(p_maxc.end(), {0, 2});
        maxcs.insert(maxcs.end(), {0, 1});
        p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 2});
        node_to_cids.insert(node_to_cids.end(), {0, 0});
        cell.Assign(std::move(p_maxc),
                    std::move(maxcs),
                    std::move(p_node_to_cid),
                    std::move(node_to_cids),
                    2);
    };
    assign_clique();
    REQUIRE(cell.HasCliqueIndex(2));
    cell.MarkUnavailable();
    REQUIRE_FALSE(cell.HasCliqueIndex(2));
    cell.MarkAvailable(2);
    REQUIRE(cell.HasCliqueIndex(2));

    vsag::CliqueDataCellBaseView view;
    REQUIRE(cell.TryGetBaseView(2, view));
    std::atomic<bool> writer_started{false};
    auto writer = std::async(std::launch::async, [&]() {
        writer_started.store(true, std::memory_order_release);
        assign_clique();
    });
    while (not writer_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    REQUIRE(writer.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);

    view.guard.unlock();
    REQUIRE(writer.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    writer.get();
}

TEST_CASE("Clique delete retires only undersized cliques and shares the Add delta",
          "[ut][hgraph][mci]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    constexpr uint64_t total = 7;

    vsag::Vector<vsag::InnerIdType> p_maxc(&allocator);
    vsag::Vector<vsag::InnerIdType> maxcs(&allocator);
    vsag::Vector<vsag::InnerIdType> p_node_to_cid(&allocator);
    vsag::Vector<vsag::InnerIdType> node_to_cids(&allocator);
    p_maxc.insert(p_maxc.end(), {0, 3, 5});
    maxcs.insert(maxcs.end(), {0, 1, 2, 0, 3});
    p_node_to_cid.insert(p_node_to_cid.end(), {0, 2, 3, 4, 5, 5, 5, 5});
    node_to_cids.insert(node_to_cids.end(), {0, 1, 0, 0, 1});
    cell.Assign(std::move(p_maxc),
                std::move(maxcs),
                std::move(p_node_to_cid),
                std::move(node_to_cids),
                total);

    REQUIRE(cell.AppendNodeToClique(4, 0, total, 8));
    vsag::Vector<vsag::InnerIdType> added_clique(&allocator);
    added_clique.insert(added_clique.end(), {0, 5});
    cell.AppendNewClique(added_clique, total);

    vsag::Vector<vsag::InnerIdType> removed(&allocator);
    removed.push_back(0);
    const auto snapshot = cell.PrepareDelete(removed, 3, 3);
    REQUIRE(snapshot.affected_clique_ids == vsag::Vector<vsag::InnerIdType>({0, 1, 2}, &allocator));
    REQUIRE(snapshot.retired_clique_ids == vsag::Vector<vsag::InnerIdType>({1, 2}, &allocator));
    REQUIRE(snapshot.repair_node_ids == vsag::Vector<vsag::InnerIdType>({3, 5}, &allocator));

    cell.CommitDelete(removed, snapshot.retired_clique_ids, total);

    vsag::Vector<vsag::InnerIdType> ids(&allocator);
    cell.CollectNodeCliqueIds(0, ids);
    REQUIRE(ids.empty());
    cell.GetCliqueMembers(0, ids);
    REQUIRE(ids == vsag::Vector<vsag::InnerIdType>({1, 2, 4}, &allocator));
    for (vsag::InnerIdType clique_id = 1; clique_id < 3; ++clique_id) {
        ids.clear();
        cell.GetCliqueMembers(clique_id, ids);
        REQUIRE(ids.empty());
        REQUIRE_FALSE(cell.AppendNodeToClique(6, clique_id, total, 8));
    }
    for (auto node_id : {1U, 2U, 4U}) {
        ids.clear();
        cell.CollectNodeCliqueIds(node_id, ids);
        REQUIRE_FALSE(ids.empty());
    }
    for (auto node_id : {3U, 5U}) {
        ids.clear();
        cell.CollectNodeCliqueIds(node_id, ids);
        REQUIRE(ids.empty());
    }

    vsag::Vector<vsag::InnerIdType> repair_clique(&allocator);
    repair_clique.insert(repair_clique.end(), {3, 5});
    cell.AppendNewClique(repair_clique, total);
    REQUIRE(cell.AppendNodeToClique(6, 0, total, 4));
    ids.clear();
    cell.GetCliqueMembers(0, ids);
    REQUIRE(std::find(ids.begin(), ids.end(), 6) != ids.end());
    REQUIRE_FALSE(cell.AppendNodeToClique(6, 0, total, 4));
    const auto stats = cell.CollectStats(total);
    REQUIRE(stats.inactive_node_count == 1);
    REQUIRE(stats.retired_clique_count == 2);
    REQUIRE(stats.delta_clique_count == 2);
    REQUIRE(stats.covered_nodes == 6);

    vsag::CliqueDataCellBaseView view;
    REQUIRE_FALSE(cell.TryGetBaseView(total, view));

    cell.Flush(total);
    auto flushed = cell.CollectStats(total);
    REQUIRE(flushed.delta_clique_count == 0);
    REQUIRE(flushed.retired_clique_count == 0);
    REQUIRE(flushed.total_clique_count == 2);
    REQUIRE(flushed.covered_nodes == stats.covered_nodes);
    REQUIRE(flushed.total_membership_count == stats.total_membership_count);
    REQUIRE(flushed.inactive_node_count == 1);
    ids.clear();
    cell.CollectNodeCliqueIds(0, ids);
    REQUIRE(ids.empty());
    cell.GetCliqueMembers(0, ids);
    REQUIRE(ids == vsag::Vector<vsag::InnerIdType>({1, 2, 4, 6}, &allocator));
    REQUIRE_FALSE(cell.AppendNodeToClique(0, 0, total, 8));

    vsag::CliqueDataCellSearchView pinned;
    REQUIRE(cell.TryGetSearchView(total, pinned));
    std::atomic<bool> flushing{false};
    auto flush = std::async(std::launch::async, [&]() {
        flushing.store(true);
        cell.Flush(total);
    });
    while (not flushing.load()) {
        std::this_thread::yield();
    }
    const auto status = flush.wait_for(std::chrono::milliseconds(20));
    pinned.guard.unlock();
    flush.get();
    REQUIRE(status == std::future_status::timeout);
}

TEST_CASE("Clique delete repairs only members below the projected MCT threshold",
          "[ut][hgraph][mci]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    constexpr uint64_t total = 7;

    vsag::Vector<vsag::InnerIdType> p_maxc(&allocator);
    vsag::Vector<vsag::InnerIdType> maxcs(&allocator);
    vsag::Vector<vsag::InnerIdType> p_node_to_cid(&allocator);
    vsag::Vector<vsag::InnerIdType> node_to_cids(&allocator);
    p_maxc.insert(p_maxc.end(), {0, 3, 5, 7, 9});
    maxcs.insert(maxcs.end(), {0, 1, 2, 1, 3, 1, 4, 1, 5});
    p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 5, 6, 7, 8, 9, 9});
    node_to_cids.insert(node_to_cids.end(), {0, 0, 1, 2, 3, 0, 1, 2, 3});
    cell.Assign(std::move(p_maxc),
                std::move(maxcs),
                std::move(p_node_to_cid),
                std::move(node_to_cids),
                total);

    vsag::Vector<vsag::InnerIdType> removed(&allocator);
    removed.push_back(0);
    const auto snapshot = cell.PrepareDelete(removed, 3, 3);
    REQUIRE(snapshot.affected_clique_ids.size() == 1);
    REQUIRE(snapshot.affected_clique_ids.front() == 0);
    REQUIRE(snapshot.retired_clique_ids.size() == 1);
    REQUIRE(snapshot.retired_clique_ids.front() == 0);
    REQUIRE(snapshot.repair_node_ids.size() == 1);
    REQUIRE(snapshot.repair_node_ids.front() == 2);
}

TEST_CASE("HGraph Merge rebuilds the MCI companion", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 16;
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 5000);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto source = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    auto destination = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(source.value()->Build(make_dataset(ids, vectors, 0, count, dim)).has_value());

    vsag::MergeUnit unit;
    unit.index = source.value();
    unit.id_map_func = [](int64_t id) { return std::make_tuple(true, id); };
    REQUIRE(destination.value()->Merge({unit}).has_value());
    const auto stats = vsag::JsonType::Parse(destination.value()->GetStats());
    REQUIRE(stats["mci_has_index"].GetBool());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dim)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto result = destination.value()->KnnSearch(
        query,
        3,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"mci_seed_ratio":1.0,)"
        R"("hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
}

TEST_CASE("HGraph MCI flush preserves queries across allocation failures",
          "[ut][hgraph][mci][flush]") {
    FlushFailureAllocator allocator;
    auto params = vsag::JsonType::Parse(generate_hgraph_mci_params(2));
    params["index_param"]["graph_type"].SetString("nsw");
    params["index_param"]["max_degree"].SetInt(32);
    params["index_param"]["build_thread_count"].SetInt(1);
    auto index = vsag::Factory::CreateIndex("hgraph", params.Dump(), &allocator).value();
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};
    std::vector<float> vectors{0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
    REQUIRE(index->Build(make_dataset(ids, vectors, 0, 4, 2)).has_value());
    REQUIRE(index->Add(make_dataset(ids, vectors, 4, 2, 2)).has_value());
    const auto before = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(before["mci_delta_clique_count"].GetInt() > 0);
    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    bool succeeded = false;
    uint64_t failures = 0;
    for (int64_t budget = 0; budget < 128; ++budget) {
        allocator.remaining = budget;
        const auto flushed = index->Flush();
        allocator.remaining = -1;
        succeeded = flushed.has_value();
        failures += not succeeded;
        const auto after = vsag::JsonType::Parse(index->GetStats());
        REQUIRE(after["mci_has_index"].GetBool());
        REQUIRE(after["mci_covered_nodes"].GetInt() == before["mci_covered_nodes"].GetInt());
        REQUIRE(after["mci_total_membership_count"].GetInt() ==
                before["mci_total_membership_count"].GetInt());
        REQUIRE(after["mci_delta_clique_count"].GetInt() ==
                (succeeded ? 0 : before["mci_delta_clique_count"].GetInt()));
        auto result =
            index->KnnSearch(make_dataset(ids, vectors, 5, 1, 2),
                             6,
                             R"({"hgraph":{"ef_search":100,"use_mci":true,"mci_seed_ratio":100,
                           "hgraph_valid_ratio_threshold":1}})",
                             filter);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
        REQUIRE(result.value()->GetDim() == 6);
        REQUIRE(result.value()->GetIds()[0] == ids[5]);
        if (succeeded) {
            break;
        }
    }
    REQUIRE(succeeded);
    REQUIRE(failures > 0);
}

TEST_CASE("Clique flush is atomic on allocation failure", "[ut][hgraph][mci][flush]") {
    FlushFailureAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    cell.Clear(3);
    vsag::Vector<vsag::InnerIdType> members({0, 1, 2}, &allocator);
    cell.AppendNewClique(members, 3);
    cell.MarkAvailable(3);
    bool succeeded = false;
    uint64_t failures = 0;
    for (int64_t budget = 0; budget < 64; ++budget) {
        allocator.remaining = budget;
        try {
            cell.Flush(3);
            succeeded = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        allocator.remaining = -1;
        const auto stats = cell.CollectStats(3);
        REQUIRE(stats.has_index);
        REQUIRE(stats.covered_nodes == 3);
        REQUIRE(stats.total_membership_count == 3);
        REQUIRE(stats.delta_clique_count == (succeeded ? 0 : 1));
        if (succeeded) {
            break;
        }
    }
    REQUIRE(succeeded);
    REQUIRE(failures > 0);
}

TEST_CASE("Clique remapping preserves tombstones and is allocation failure atomic",
          "[ut][hgraph][mci][force_remove]") {
    FlushFailureAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    cell.Clear(5);
    vsag::Vector<vsag::InnerIdType> members({0, 1, 4}, &allocator);
    cell.AppendNewClique(members, 5);
    cell.MarkAvailable(5);
    cell.Flush(5);
    REQUIRE(cell.AppendNodeToClique(3, 0, 5, 5));
    members = vsag::Vector<vsag::InnerIdType>({2, 3, 4}, &allocator);
    cell.AppendNewClique(members, 5);
    vsag::Vector<vsag::InnerIdType> removed({1}, &allocator);
    auto snapshot = cell.PrepareDelete(removed, 1, 3);
    cell.CommitDelete(removed, snapshot.retired_clique_ids, 5);
    // Physically remove 0; tail 4 moves to 0. Keep marked slot 1 and remap its mask.
    const auto invalid = std::numeric_limits<vsag::InnerIdType>::max();
    vsag::Vector<vsag::InnerIdType> mapping({invalid, 1, 2, 3, 0}, &allocator);
    vsag::Vector<vsag::InnerIdType> bad_mapping({invalid, 1, 2, 3, 3}, &allocator);
    REQUIRE_THROWS(cell.RemapNodes(bad_mapping, 4));
    REQUIRE(cell.HasCliqueIndex(5));
    bool succeeded = false;
    uint64_t failures = 0;
    for (int64_t budget = 0; budget < 64; ++budget) {
        allocator.remaining = budget;
        try {
            cell.RemapNodes(mapping, 4);
            succeeded = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        allocator.remaining = -1;
        if (succeeded) {
            break;
        }
        REQUIRE(cell.HasCliqueIndex(5));
        REQUIRE(cell.CollectStats(5).delta_clique_count == 1);
    }
    REQUIRE(succeeded);
    REQUIRE(failures > 0);
    cell.MarkAvailable(4);
    const auto stats = cell.CollectStats(4);
    REQUIRE(stats.has_index);
    REQUIRE(stats.inactive_node_count == 1);
    REQUIRE(stats.delta_clique_count == 0);
    REQUIRE(stats.covered_nodes == 3);
    REQUIRE(stats.total_membership_count == 5);
    members.clear();
    cell.GetCliqueMembers(0, members);
    std::sort(members.begin(), members.end());
    REQUIRE(members == vsag::Vector<vsag::InnerIdType>({0, 3}, &allocator));
    vsag::CliqueDataCellSearchView view;
    REQUIRE(cell.TryGetSearchView(4, view));
    REQUIRE_FALSE(view.IsLiveNode(1));
    REQUIRE(view.IsLiveNode(0));
}

TEST_CASE("Clique flush handles fully deleted indexes and later additions",
          "[ut][hgraph][mci][flush]") {
    vsag::DefaultAllocator allocator;
    vsag::CliqueDataCell cell(&allocator);
    cell.Clear(2);
    vsag::Vector<vsag::InnerIdType> members({0, 1}, &allocator);
    cell.AppendNewClique(members, 2);
    const auto snapshot = cell.PrepareDelete(members, 3, 3);
    cell.CommitDelete(members, snapshot.retired_clique_ids, 2);
    cell.MarkAvailable(2);
    cell.Flush(2);
    auto stats = cell.CollectStats(2);
    REQUIRE(stats.total_clique_count == 0);
    REQUIRE(stats.inactive_node_count == 2);
    vsag::CliqueDataCellSearchView view;
    REQUIRE_FALSE(cell.TryGetSearchView(2, view));
    members.assign(1, 2);
    cell.AppendNewClique(members, 3);
    cell.MarkAvailable(3);
    cell.Flush(3);
    stats = cell.CollectStats(3);
    REQUIRE(stats.has_index);
    REQUIRE(stats.covered_nodes == 1);
    REQUIRE(stats.inactive_node_count == 2);
    REQUIRE(stats.delta_clique_count == 0);
    REQUIRE(cell.TryGetSearchView(3, view));
    REQUIRE_FALSE(view.IsLiveNode(0));
    REQUIRE(view.IsLiveNode(2));
    REQUIRE_FALSE(view.IsLiveNode(3));
}
