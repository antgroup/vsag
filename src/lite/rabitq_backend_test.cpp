// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "lite/backend.h"
#include "lite/rabitq_snapshot.h"
#include "vsag/lite/index.h"

using vsag::lite::Index;
using vsag::lite::VectorStorage;

namespace {
std::unique_ptr<Index>
make_index(uint64_t dim = 17) {
    auto created = Index::Create(dim);
    REQUIRE(created);
    auto index = std::move(*created);
    std::vector<float> vector(dim);
    for (int64_t row = 0; row < 12; ++row) {
        for (uint64_t d = 0; d < dim; ++d) {
            vector[d] = std::sin(static_cast<float>(row * 7 + d) * 0.17F);
        }
        REQUIRE(index->Add(row == 0 ? INT64_MIN : 100 - row * 3, vector.data(), dim));
    }
    REQUIRE(index->BuildGraph(VectorStorage::RABITQ8, 4, 16));
    return index;
}

std::string
snapshot(const Index& index) {
    std::ostringstream output(std::ios::binary);
    REQUIRE(index.Save(output));
    return output.str();
}

void
put_u64(std::string& bytes, uint64_t offset, uint64_t value) {
    REQUIRE(offset + 8 <= bytes.size());
    for (uint64_t i = 0; i < 8; ++i) {
        bytes[offset + i] = static_cast<char>((value >> (8 * i)) & 255);
    }
}

void
same_results(const std::vector<vsag::lite::Neighbor>& left,
             const std::vector<vsag::lite::Neighbor>& right) {
    REQUIRE(left.size() == right.size());
    for (uint64_t i = 0; i < left.size(); ++i) {
        REQUIRE(left[i].id == right[i].id);
        REQUIRE(left[i].distance == right[i].distance);
    }
}
}  // namespace

TEST_CASE("RaBitQ public candidate lifecycle and encoded persistence", "[lite-rabitq]") {
    for (uint64_t dim : {1, 7, 17, 128, 768, 960}) {
        auto index = make_index(dim);
        REQUIRE(index->ActiveVectorStorage() == VectorStorage::RABITQ8);
        REQUIRE(index->ActiveBackend() == vsag::lite::BackendKind::GRAPH);
        std::vector<float> query(dim, 0.25F);
        REQUIRE(index->Update(INT64_MIN, query.data(), dim));
        REQUIRE(index->Remove(97));
        REQUIRE_FALSE(index->Remove(97));
        REQUIRE(index->Add(INT64_MAX, query.data(), dim));
        const auto result = index->SearchWithOptions(query.data(), dim, 100, {100});
        REQUIRE(result);
        REQUIRE(result->size() == index->Size());
        for (const auto& neighbor : *result) {
            REQUIRE(std::isfinite(neighbor.distance));
        }
        const auto saved = snapshot(*index);
        REQUIRE(saved.substr(0, 8) == "VSAGLQ01");
        std::istringstream input(saved, std::ios::binary);
        auto loaded = Index::Load(input);
        REQUIRE(loaded);
        REQUIRE((*loaded)->ActiveVectorStorage() == VectorStorage::RABITQ8);
        REQUIRE((*loaded)->Size() == index->Size());
        same_results(*result, *(*loaded)->SearchWithOptions(query.data(), dim, 100, {100}));
        REQUIRE(snapshot(**loaded) == saved);
        REQUIRE((*loaded)->Remove(INT64_MAX));
        REQUIRE((*loaded)->Add(999, query.data(), dim));
        while ((*loaded)->Size() != 0) {
            auto all = (*loaded)->Search(query.data(), dim, UINT64_MAX);
            REQUIRE(all);
            REQUIRE_FALSE(all->empty());
            REQUIRE((*loaded)->Remove(all->front().id));
        }
        const auto empty = snapshot(**loaded);
        std::istringstream empty_input(empty, std::ios::binary);
        auto empty_loaded = Index::Load(empty_input);
        REQUIRE(empty_loaded);
        REQUIRE((*empty_loaded)->Size() == 0);
        REQUIRE((*empty_loaded)->Add(1, query.data(), dim));
    }
}

TEST_CASE("RaBitQ filters, budgets and expected errors preserve contents", "[lite-rabitq]") {
    auto index = make_index();
    std::vector<float> query(17, 0.25F);
    const auto before = snapshot(*index);
    REQUIRE_FALSE(index->Add(INT64_MIN, query.data(), 17));
    REQUIRE_FALSE(index->Update(-1, query.data(), 17));
    REQUIRE_FALSE(index->Add(9, nullptr, 17));
    REQUIRE_FALSE(index->Update(INT64_MIN, query.data(), 16));
    auto invalid = query;
    invalid[0] = NAN;
    REQUIRE_FALSE(index->Add(9, invalid.data(), 17));
    REQUIRE_FALSE(index->Search(invalid.data(), 17, 1));
    REQUIRE(index->Search(query.data(), 17, 0)->empty());
    REQUIRE(index->Search(query.data(), 17, 12, [](int64_t) { return false; })->empty());
    auto filtered = index->SearchWithOptions(
        query.data(), 17, 12, {100}, [](int64_t id) { return id == INT64_MIN or id == 97; });
    REQUIRE(filtered);
    REQUIRE(filtered->size() == 2);
    for (const auto& value : *filtered) {
        REQUIRE((value.id == INT64_MIN or value.id == 97));
    }
    auto thrown = index->Search(
        query.data(), 17, 12, [](int64_t) -> bool { throw std::runtime_error("caller failure"); });
    REQUIRE_FALSE(thrown);
    REQUIRE(thrown.error().type == vsag::ErrorType::INTERNAL_ERROR);
    REQUIRE(index->SearchWithOptions(query.data(), 17, 3, {1})->size() == 3);
    auto oom = index->Search(query.data(), 17, 12, [](int64_t) -> bool { throw std::bad_alloc(); });
    REQUIRE_FALSE(oom);
    REQUIRE(oom.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
    auto capacity = index->Search(
        query.data(), 17, 12, [](int64_t) -> bool { throw std::length_error("capacity"); });
    REQUIRE_FALSE(capacity);
    REQUIRE(capacity.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
    auto unknown = index->Search(query.data(), 17, 12, [](int64_t) -> bool { throw 1; });
    REQUIRE_FALSE(unknown);
    REQUIRE(unknown.error().type == vsag::ErrorType::INTERNAL_ERROR);
    std::vector<float> extreme(17, std::numeric_limits<float>::max());
    REQUIRE_FALSE(index->Update(INT64_MIN, extreme.data(), 17));
    REQUIRE_FALSE(index->Add(9, extreme.data(), 17));
    REQUIRE(snapshot(*index) == before);
    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    REQUIRE_FALSE(index->Save(failed));
}

TEST_CASE("RaBitQ rejects malformed encoded snapshots before large allocations", "[lite-rabitq]") {
    auto index = make_index();
    const auto saved = snapshot(*index);
    for (uint64_t length = 0; length < saved.size(); ++length) {
        std::istringstream input(saved.substr(0, length), std::ios::binary);
        REQUIRE_FALSE(Index::Load(input));
    }
    auto reject = [](const std::string& bytes) {
        std::istringstream input(bytes, std::ios::binary);
        auto loaded = Index::Load(input);
        REQUIRE_FALSE(loaded);
        REQUIRE(loaded.error().type == vsag::ErrorType::INVALID_BINARY);
    };
    reject(saved + "x");
    auto bad = saved;
    put_u64(bad, 8, 99);
    reject(bad);
    bad = saved;
    put_u64(bad, 24, UINT64_MAX);
    reject(bad);
    constexpr uint64_t model_bytes = 17 * 4 + 4 * 3;
    constexpr uint64_t code_bytes = 24 + 8 * 3;
    constexpr uint64_t tail = 48 + model_bytes + 12 * code_bytes;
    bad = saved;
    put_u64(bad, tail, 65);
    reject(bad);
    bad = saved;
    put_u64(bad, tail + 24 + 8, uint64_t(INT64_MIN));
    reject(bad);
    bad = saved;
    put_u64(bad, tail + 40 + uint64_t{12} * 8, 1);
    reject(bad);
    bad = saved;
    put_u64(bad, tail + 48 + uint64_t{12} * 16, 0);
    reject(bad);
    bad = saved;
    std::memset(bad.data() + 48 + model_bytes + 4, 0, 4);
    reject(bad);
}

TEST_CASE("RaBitQ snapshot slot map rejects duplicate external IDs", "[lite-rabitq]") {
    auto index = make_index();
    const auto saved = snapshot(*index);
    constexpr uint64_t ids_offset = 48 + 17 * 4 + 4 * 3 + 12 * (24 + 8 * 3) + 24;
    for (int64_t id : {INT64_MIN, int64_t{-7}, int64_t{0}, INT64_MAX}) {
        for (uint64_t duplicate_slot : {uint64_t{1}, uint64_t{11}}) {
            auto bad = saved;
            put_u64(bad, ids_offset, static_cast<uint64_t>(id));
            put_u64(bad, ids_offset + duplicate_slot * 8, static_cast<uint64_t>(id));
            std::istringstream input(bad, std::ios::binary);
            auto loaded = Index::Load(input);
            REQUIRE_FALSE(loaded);
            REQUIRE(loaded.error().type == vsag::ErrorType::INVALID_BINARY);
        }
    }
    std::istringstream valid(saved, std::ios::binary);
    auto loaded = Index::Load(valid);
    REQUIRE(loaded);
    REQUIRE(snapshot(**loaded) == saved);
}

TEST_CASE("RaBitQ training failure keeps flat index and old formats compatible", "[lite-rabitq]") {
    auto empty = Index::Create(3);
    REQUIRE(empty);
    REQUIRE_FALSE((*empty)->BuildGraph(VectorStorage::RABITQ8));
    REQUIRE((*empty)->ActiveBackend() == vsag::lite::BackendKind::BRUTE_FORCE);
    for (VectorStorage storage : {VectorStorage::FP32, VectorStorage::FP16}) {
        auto created = Index::Create(3);
        float vector[]{1, 2, 3};
        REQUIRE((*created)->Add(42, vector, 3));
        REQUIRE((*created)->BuildGraph(storage, 2, 2));
        const auto saved = snapshot(**created);
        REQUIRE(saved.substr(0, 8) == "VSAGLT01");
        std::istringstream input(saved, std::ios::binary);
        auto loaded = Index::Load(input);
        REQUIRE(loaded);
        REQUIRE((*loaded)->ActiveVectorStorage() == storage);
    }
}

TEST_CASE("RaBitQ internal backend reconstruction and graph metadata", "[lite-rabitq]") {
    auto source = vsag::lite::detail::make_brute_force_backend(17);
    REQUIRE(source);
    std::vector<float> vector(17, 0.25F);
    for (int64_t id : {-8, 42, 97, 123}) {
        vector[0] = static_cast<float>(id) / 128;
        REQUIRE((*source)->Add(id, vector.data(), 17));
    }
    auto built = vsag::lite::detail::make_rabitq_graph_backend(**source, 2, 8);
    REQUIRE(built);
    std::vector<float> first;
    std::vector<float> second;
    const auto* pointer = (*built)->VectorAt(0, first);
    REQUIRE(pointer == first.data());
    REQUIRE(first.size() == 17);
    const auto preserved = first;
    const auto* second_pointer = (*built)->VectorAt(1, second);
    REQUIRE(second_pointer == second.data());
    REQUIRE(first == preserved);
    REQUIRE(pointer != second.data());
    REQUIRE((*built)->LinkCountAt(0) > 0);
    REQUIRE((*built)->LinkAt(0, 0) < (*built)->Size());
    REQUIRE((*built)->IncomingLinkCountAt(0) == 0);
    REQUIRE((*built)->IncomingLinkAt(0, 0) == 0);
    REQUIRE((*built)->IncomingLogicalBytes() == 0);
    REQUIRE((*built)->IncomingCapacityBytes() == 0);
    REQUIRE((*built)->IncomingCompactionCount() == 0);
    REQUIRE((*source)->MaxDegree() == 0);
    REQUIRE((*source)->EfSearch() == 0);
    REQUIRE((*source)->LinkCountAt(0) == 0);
    REQUIRE((*source)->LinkAt(0, 0) == 0);
    std::ostringstream unsupported;
    REQUIRE_FALSE((*source)->SaveEncoded(unsupported));
    auto empty = vsag::lite::detail::make_brute_force_backend(1);
    REQUIRE_FALSE(vsag::lite::detail::make_rabitq_graph_backend(**empty, 2, 2));
    REQUIRE_FALSE(vsag::lite::detail::make_rabitq_graph_backend(**source, 1, 2));
    REQUIRE_FALSE(vsag::lite::detail::make_rabitq_graph_backend(**source, 4, 2));
}

TEST_CASE("RaBitQ incoming adjacency survives directed CRUD repair", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    std::vector<float> base{0, 1, 2, 3, 4, 5, 6, 7};
    auto model = codec::train(base, 8, 1, 47);
    codec::EncodedRecords codes(1);
    for (float value : base) {
        codes.Append(codec::encode(model, &value));
    }
    codec::GraphTopology graph{{0, 2, 4, 6, 8, 10, 12, 14, 16},
                               {1, 7, 2, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 0, 6}};
    for (bool incoming : {false, true}) {
        codec::MutableGraphState state(
            model, codes, graph, {0, 1, 2, 3, 4, 5, 6, 7}, 2, 8, true, incoming);
        float changed = 9.0F;
        REQUIRE(state.Update(3, &changed));
        state.Validate();
        REQUIRE(state.Remove(0));
        state.Validate();
        REQUIRE(state.Add(-8, &changed));
        state.Validate();
        REQUIRE(state.GetMutationScanTiming().update_wall_us >= 0);
        REQUIRE(state.GetMutationScanTiming().remove_wall_us >= 0);
        if (incoming) {
            const auto before = state.GetIncomingMemoryUsage();
            REQUIRE(before.logical_bytes ==
                    state.Size() * sizeof(std::vector<codec::IncomingSlot>) +
                        before.edges * sizeof(codec::IncomingSlot));
            state.CompactIncoming();
            REQUIRE(state.GetIncomingMemoryUsage().logical_bytes == before.logical_bytes);
            const auto distribution = state.GetIncomingDistribution();
            REQUIRE(distribution.nodes == 8);
            REQUIRE(distribution.edge_logical_bytes == before.edges * sizeof(codec::IncomingSlot));
            REQUIRE(distribution.outer_vector_bytes + distribution.edge_capacity_bytes ==
                    state.GetIncomingMemoryUsage().capacity_bytes);
        }
        std::stringstream stream;
        codec::save_mutable_snapshot(stream, state);
        auto loaded = codec::load_mutable_snapshot(stream, incoming);
        loaded.Validate();
        REQUIRE(loaded.GetIds() == state.GetIds());
        REQUIRE(loaded.GetGraph().neighbors == state.GetGraph().neighbors);
        REQUIRE(loaded.GetIncomingMemoryUsage().edges ==
                (incoming ? state.GetIncomingMemoryUsage().edges : 0));
    }
    codec::MutableGraphState tiny(model, codec::EncodedRecords(1), {{0}, {}}, {}, 2, 2);
    REQUIRE_FALSE(tiny.Update(9, base.data()));
    REQUIRE_FALSE(tiny.Remove(9));
    REQUIRE(tiny.Add(9, base.data()));
    REQUIRE(tiny.Update(9, base.data() + 1));
    REQUIRE(tiny.Remove(9));
    REQUIRE(tiny.Search(base.data(), 1).neighbors.empty());
    REQUIRE(tiny.GetMutationFallbacks() == 0);
}

TEST_CASE("RaBitQ owned adjacency matches CSR construction", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    const std::vector<float> base{0, 1, 2, 3};
    const auto model = codec::train(base, 4, 1, 47);
    codec::EncodedRecords codes(1);
    for (const float value : base) {
        codes.Append(codec::encode(model, &value));
    }
    const codec::GraphTopology graph{{0, 2, 4, 6, 8}, {1, 3, 0, 2, 1, 3, 0, 2}};
    const std::vector<int64_t> ids{INT64_MIN, -1, 0, INT64_MAX};
    const auto bytes = [](const codec::MutableGraphState& state) {
        std::stringstream stream;
        codec::save_mutable_snapshot(stream, state);
        return stream.str();
    };
    for (const bool reverse : {false, true}) {
        codec::MutableGraphState reference(model, codes, graph, ids, 2, 8, false, reverse);
        codec::MutableGraphState owned(codec::OwnedGraphAdjacency{codec::expand_graph(graph)},
                                       model,
                                       codes,
                                       ids,
                                       2,
                                       8,
                                       false,
                                       reverse);
        REQUIRE(bytes(owned) == bytes(reference));
        float changed = 7.0F;
        REQUIRE(owned.Update(INT64_MIN, &changed));
        REQUIRE(reference.Update(INT64_MIN, &changed));
        REQUIRE(bytes(owned) == bytes(reference));
        REQUIRE(owned.Remove(0));
        REQUIRE(reference.Remove(0));
        REQUIRE(bytes(owned) == bytes(reference));
        REQUIRE(owned.Add(42, &changed));
        REQUIRE(reference.Add(42, &changed));
        REQUIRE(bytes(owned) == bytes(reference));
        const auto result = owned.Search(&changed, 3).neighbors;
        const auto expected = reference.Search(&changed, 3).neighbors;
        REQUIRE(result.size() == expected.size());
        for (uint64_t i = 0; i < result.size(); ++i) {
            REQUIRE(result[i].id == expected[i].id);
            REQUIRE(result[i].distance == expected[i].distance);
        }
        for (uint64_t invalid = 0; invalid < 5; ++invalid) {
            auto rows = codec::expand_graph(graph);
            if (invalid == 0) {
                rows.pop_back();
            } else if (invalid == 1) {
                rows[0][0] = 4;
            } else if (invalid == 2) {
                rows[0][0] = 0;
            } else if (invalid == 3) {
                rows[0][1] = rows[0][0];
            } else {
                rows[0].push_back(2);
            }
            REQUIRE_THROWS(codec::MutableGraphState(codec::OwnedGraphAdjacency{std::move(rows)},
                                                    model,
                                                    codes,
                                                    ids,
                                                    2,
                                                    8,
                                                    false,
                                                    reverse));
        }
        REQUIRE_THROWS(
            codec::MutableGraphState(codec::OwnedGraphAdjacency{codec::expand_graph(graph)},
                                     model,
                                     codes,
                                     {1, 1, 2, 3},
                                     2,
                                     8,
                                     false,
                                     reverse));
    }
    codec::MutableGraphState empty(
        codec::OwnedGraphAdjacency{}, model, codec::EncodedRecords(1), {}, 2, 8);
    empty.Validate();
    REQUIRE(empty.Search(base.data(), 1).neighbors.empty());
}

TEST_CASE("RaBitQ unchanged complete encoding preserves exact snapshot", "[lite-rabitq]") {
    for (uint64_t dim : {1, 17, 128, 768, 960}) {
        auto index = make_index(dim);
        std::vector<float> vector(dim, 0.25F);
        REQUIRE(index->Update(INT64_MIN, vector.data(), dim));
        const auto before = snapshot(*index);
        for (uint64_t repeat = 0; repeat < 10; ++repeat) {
            REQUIRE(index->Update(INT64_MIN, vector.data(), dim));
        }
        REQUIRE(snapshot(*index) == before);
        REQUIRE_FALSE(index->Add(INT64_MIN, vector.data(), dim));
        REQUIRE_FALSE(index->Update(-1, vector.data(), dim));
        REQUIRE_FALSE(index->Remove(-1));
        REQUIRE(snapshot(*index) == before);
    }
}

TEST_CASE("RaBitQ no-op detection includes metadata and both planes", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    std::vector<float> base{0, 1, 2};
    const auto model = codec::train(base, 3, 1, 47);
    codec::EncodedRecords records(1);
    const auto code = codec::encode(model, base.data());
    records.Append(code);
    codec::MutableGraphState state(model, records, {{0, 0}, {}}, {42}, 2, 2);
    REQUIRE(state.Contains(42));
    REQUIRE_FALSE(state.Contains(7));
    REQUIRE(state.SameEncoding(42, code));
    REQUIRE_FALSE(state.SameEncoding(7, code));
    auto changed = code;
    changed.norm = std::nextafter(changed.norm, INFINITY);
    REQUIRE_FALSE(state.SameEncoding(42, changed));
    changed = code;
    changed.filter[0] ^= 1;
    REQUIRE_FALSE(state.SameEncoding(42, changed));
    changed = code;
    changed.supplement[0] ^= 1;
    REQUIRE_FALSE(state.SameEncoding(42, changed));
    changed.filter.clear();
    REQUIRE_FALSE(state.SameEncoding(42, changed));
}

TEST_CASE("RaBitQ prepared mutation preserves legacy bytes and rejects damaged preparation",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (uint64_t dim : {1, 17, 128, 768, 960}) {
        std::vector<float> base(4 * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * 0.17F);
        }
        const auto model = codec::train(base, 4, dim, 47);
        codec::EncodedRecords codes(dim);
        for (uint64_t row = 0; row < 4; ++row) {
            codes.Append(codec::encode(model, base.data() + row * dim));
        }
        const codec::GraphTopology graph{{0, 2, 4, 6, 8}, {1, 3, 2, 0, 3, 1, 0, 2}};
        codec::MutableGraphState regular(model, codes, graph, {-8, 42, 97, 123}, 2, 8);
        auto prepared_state = regular;
        std::vector<float> replacement(dim, 0.25F);
        REQUIRE(regular.Update(42, replacement.data()));
        auto prepared = codec::prepare_encoding(model, replacement.data());
        const bool applied = prepared_state.UpdatePrepared(42, std::move(prepared));
        REQUIRE(applied);
        regular.Validate();
        prepared_state.Validate();
        std::stringstream first;
        std::stringstream second;
        codec::save_mutable_snapshot(first, regular);
        codec::save_mutable_snapshot(second, prepared_state);
        REQUIRE(first.str() == second.str());
        const auto before = second.str();
        auto reject = [&](codec::PreparedEncoding damaged) {
            REQUIRE_THROWS_AS(prepared_state.UpdatePrepared(42, std::move(damaged)),
                              std::runtime_error);
            std::stringstream unchanged;
            codec::save_mutable_snapshot(unchanged, prepared_state);
            REQUIRE(unchanged.str() == before);
        };
        auto damaged = codec::prepare_encoding(model, replacement.data());
        damaged.query.clear();
        reject(std::move(damaged));
        damaged = codec::prepare_encoding(model, replacement.data());
        damaged.code.filter.clear();
        reject(std::move(damaged));
        damaged = codec::prepare_encoding(model, replacement.data());
        damaged.query[0] = NAN;
        reject(std::move(damaged));
        damaged = codec::prepare_encoding(model, replacement.data());
        damaged.code.code_norm = 0;
        reject(std::move(damaged));
        damaged = codec::prepare_encoding(model, replacement.data());
        damaged.code.norm = INFINITY;
        reject(std::move(damaged));
        REQUIRE_FALSE(
            prepared_state.UpdatePrepared(999, codec::prepare_encoding(model, replacement.data())));
    }
}

TEST_CASE("RaBitQ journaled Update matches copied transactions through repeated changes",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (bool incoming : {false, true}) {
        constexpr uint64_t dim = 17;
        std::vector<float> base(8 * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * 0.17F);
        }
        auto model = codec::train(base, 8, dim, 47);
        codec::EncodedRecords codes(dim);
        for (uint64_t row = 0; row < 8; ++row) {
            codes.Append(codec::encode(model, base.data() + row * dim));
        }
        codec::GraphTopology graph{{0, 2, 4, 6, 8, 10, 12, 14, 16},
                                   {1, 7, 2, 0, 3, 1, 4, 2, 5, 3, 6, 4, 7, 5, 0, 6}};
        codec::MutableGraphState reference(
            model, codes, graph, {-8, 42, 97, 123, 5, 6, 7, 8}, 2, 8, false, incoming);
        auto journaled = reference;
        std::vector<float> replacement(dim);
        for (uint64_t step = 0; step < 100; ++step) {
            for (uint64_t d = 0; d < dim; ++d) {
                replacement[d] = std::cos(static_cast<float>(step + d) * 0.13F);
            }
            const auto id = reference.IdAt(step % 8);
            auto copy = reference;
            const bool expected =
                copy.UpdatePrepared(id, codec::prepare_encoding(model, replacement.data()));
            REQUIRE(expected);
            reference = std::move(copy);
            const bool updated = journaled.UpdateTransactional(
                id, codec::prepare_encoding(model, replacement.data()));
            REQUIRE(updated);
            journaled.Validate();
            reference.Validate();
            std::stringstream first;
            std::stringstream second;
            codec::save_mutable_snapshot(first, reference);
            codec::save_mutable_snapshot(second, journaled);
            REQUIRE(first.str() == second.str());
        }
    }
}

TEST_CASE("RaBitQ journaled Add matches copied state across repeated growth", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (bool incoming : {false, true}) {
        std::vector<float> training{0, 1, 2, 3};
        auto model = codec::train(training, 4, 1, 47);
        codec::MutableGraphState reference(
            model, codec::EncodedRecords(1), {{0}, {}}, {}, 4, 16, false, incoming);
        auto journaled = reference;
        for (int64_t id = 0; id < 100; ++id) {
            const float vector = static_cast<float>(id) * 0.125F;
            auto copy = reference;
            const bool expected = copy.Add(id - 50, &vector);
            REQUIRE(expected);
            reference = std::move(copy);
            const bool added = journaled.AddTransactional(id - 50, &vector);
            REQUIRE(added);
            journaled.Validate();
            reference.Validate();
            std::stringstream first;
            std::stringstream second;
            codec::save_mutable_snapshot(first, reference);
            codec::save_mutable_snapshot(second, journaled);
            REQUIRE(first.str() == second.str());
        }
        const float vector = 0.25F;
        REQUIRE_FALSE(journaled.AddTransactional(0, &vector));
    }
}

TEST_CASE("RaBitQ journaled Remove matches copied hole compaction down to empty", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (bool incoming : {false, true}) {
        std::vector<float> training{0, 1, 2, 3};
        auto model = codec::train(training, 4, 1, 47);
        codec::MutableGraphState reference(
            model, codec::EncodedRecords(1), {{0}, {}}, {}, 4, 16, false, incoming);
        for (int64_t id = 0; id < 100; ++id) {
            const float vector = static_cast<float>(id) * 0.125F;
            REQUIRE(reference.Add(id - 50, &vector));
        }
        auto journaled = reference;
        uint64_t step = 0;
        while (reference.Size() != 0) {
            const uint64_t slot = step % 2 == 0 ? 0 : reference.Size() - 1;
            const auto id = reference.IdAt(slot);
            auto copy = reference;
            REQUIRE(copy.Remove<false>(id));
            reference = std::move(copy);
            REQUIRE(journaled.RemoveTransactional(id));
            journaled.Validate();
            reference.Validate();
            std::stringstream first;
            std::stringstream second;
            codec::save_mutable_snapshot(first, reference);
            codec::save_mutable_snapshot(second, journaled);
            REQUIRE(first.str() == second.str());
            ++step;
        }
        REQUIRE_FALSE(journaled.RemoveTransactional(0));
    }
}

TEST_CASE("RaBitQ cached query sums preserve complete distance bits", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (const uint64_t dim : {1, 7, 17, 128, 768, 960}) {
        constexpr uint64_t count = 8;
        std::vector<float> base(count * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * 0.17F);
        }
        const auto model = codec::train(base, count, dim, 42);
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = std::cos(static_cast<float>(d) * 0.13F);
        }
        float query_norm = 0.0F;
        query = codec::normalize(model, query.data(), query_norm);
        const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
        codec::EncodedRecords codes(dim);
        for (uint64_t slot = 0; slot < count; ++slot) {
            codes.Append(codec::encode(model, base.data() + slot * dim));
            const auto code = codes.At(slot);
            const auto coarse = codec::filter_estimate(query, query_norm, code);
            const float uncached =
                codec::full_distance(query, query_norm, code, coarse.centered_ip);
            const float cached =
                codec::full_distance(query, query_norm, code, coarse.centered_ip, query_sum);
            uint32_t uncached_bits = 0;
            uint32_t cached_bits = 0;
            std::memcpy(&uncached_bits, &uncached, sizeof(uncached));
            std::memcpy(&cached_bits, &cached, sizeof(cached));
            REQUIRE(uncached_bits == cached_bits);
        }
    }
}

TEST_CASE("RaBitQ one-bit code distance ignores plane tails", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (const uint64_t dim : {1, 7, 8, 9, 17}) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<uint8_t> left_filter(plane_bytes * codec::K_FILTER_BITS, 0);
        std::vector<uint8_t> right_filter(plane_bytes * codec::K_FILTER_BITS, 0);
        codec::EncodedMetadata left_metadata{};
        codec::EncodedMetadata right_metadata{};
        left_metadata.norm = 2.0F;
        right_metadata.norm = 3.0F;
        const codec::EncodedView left{left_filter.data(), nullptr, left_metadata};
        codec::EncodedView right{right_filter.data(), nullptr, right_metadata};
        REQUIRE(codec::one_bit_code_distance(left, right, dim) == 1.0F);

        for (uint64_t byte = 0; byte < plane_bytes; ++byte) {
            right_filter[byte] = 0xFFU;
        }
        REQUIRE(codec::one_bit_code_distance(left, right, dim) == 25.0F);

        if ((dim & 7U) != 0) {
            std::fill(right_filter.begin(), right_filter.end(), 0);
            const uint64_t tail = dim / 8;
            const auto used_mask = static_cast<uint8_t>((1U << (dim & 7U)) - 1U);
            right_filter[tail] = static_cast<uint8_t>(~used_mask);
            REQUIRE(codec::one_bit_code_distance(left, right, dim) == 1.0F);
        }

        std::fill(right_filter.begin(), right_filter.end(), 0);
        right_filter[0] = 1U;
        const float ip = 1.0F - 2.0F / static_cast<float>(dim);
        const float expected = codec::l2_distance(2.0F, 3.0F, ip);
        REQUIRE(codec::one_bit_code_distance(left, right, dim) == expected);
    }
}

TEST_CASE("RaBitQ filter Batch4 preserves single-estimate bits", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (const uint64_t dim : {1, 7, 17, 128, 768, 960}) {
        constexpr uint64_t count = 8;
        std::vector<float> base(count * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * 0.19F);
        }
        const auto model = codec::train(base, count, dim, 91);
        std::vector<float> query(dim);
        for (uint64_t d = 0; d < dim; ++d) {
            query[d] = std::cos(static_cast<float>(d) * 0.11F);
        }
        float query_norm = 0.0F;
        query = codec::normalize(model, query.data(), query_norm);
        codec::EncodedRecords records(dim);
        for (uint64_t slot = 0; slot < 4; ++slot) {
            records.Append(codec::encode(model, base.data() + slot * dim));
        }
        std::array<codec::EncodedView, 4> codes;
        for (uint64_t slot = 0; slot < 4; ++slot) {
            codes[slot] = records.At(slot);
        }
        const auto batch = codec::filter_estimate_batch4(query, query_norm, codes);
        for (uint64_t slot = 0; slot < 4; ++slot) {
            const auto single = codec::filter_estimate(query, query_norm, codes[slot]);
            REQUIRE(std::memcmp(reinterpret_cast<const unsigned char*>(&batch[slot].centered_ip),
                                reinterpret_cast<const unsigned char*>(&single.centered_ip),
                                sizeof(single.centered_ip)) == 0);
            REQUIRE(std::memcmp(reinterpret_cast<const unsigned char*>(&batch[slot].distance),
                                reinterpret_cast<const unsigned char*>(&single.distance),
                                sizeof(single.distance)) == 0);
            REQUIRE(std::memcmp(reinterpret_cast<const unsigned char*>(&batch[slot].lower_bound),
                                reinterpret_cast<const unsigned char*>(&single.lower_bound),
                                sizeof(single.lower_bound)) == 0);
        }
    }
}

TEST_CASE("RaBitQ supplement SIMD matches scalar estimates and plane tails", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    std::vector<codec::RaBitQFilterIP> kernels{codec::rabitq_supplement_ip_generic,
                                               codec::select_rabitq_supplement_ip()};
#ifdef VSAG_LITE_RABITQ_X86_SIMD
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) {
        kernels.push_back(codec::rabitq_supplement_ip_avx2);
    }
    if (__builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512dq") and
        __builtin_cpu_supports("avx512bw") and __builtin_cpu_supports("avx512vl")) {
        kernels.push_back(codec::rabitq_supplement_ip_avx512);
    }
#endif
    for (const uint64_t dim :
         {0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 127, 128, 129, 767, 768, 769, 959, 960, 961}) {
        const uint64_t plane_bytes = (dim + 7) / 8;
        std::vector<float> query(dim + 1);
        std::vector<uint8_t> planes(5 * plane_bytes + 1);
        for (uint64_t pattern = 0; pattern < 32; ++pattern) {
            for (uint64_t d = 0; d < dim; ++d) {
                query[d + 1] = std::sin(static_cast<float>(d * 7 + pattern) * 0.11F);
            }
            for (uint64_t byte = 0; byte < 5 * plane_bytes; ++byte) {
                planes[byte + 1] = static_cast<uint8_t>((byte * 37 + pattern * 29) & 255U);
            }
            float expected = 0.0F;
            float absolute_sum = 0.0F;
            for (uint64_t d = 0; d < dim; ++d) {
                const auto code = codec::read_plane_code(
                    planes.data() + 1, plane_bytes, d, codec::K_SUPPLEMENT_BITS, false);
                const float product = query[d + 1] * static_cast<float>(code);
                expected += product;
                absolute_sum += std::fabs(product);
            }
            uint32_t expected_bits = 0;
            std::memcpy(&expected_bits, &expected, sizeof(expected));
            for (const auto kernel : kernels) {
                const float actual = kernel(query.data() + 1, planes.data() + 1, dim);
                uint32_t actual_bits = 0;
                std::memcpy(&actual_bits, &actual, sizeof(actual));
                if (kernel == codec::rabitq_supplement_ip_generic) {
                    REQUIRE(actual_bits == expected_bits);
                } else {
                    // SIMD FMA and lane reduction change rounding, not decoded values.
                    const float tolerance = 32.0F * std::numeric_limits<float>::epsilon() *
                                            std::max(1.0F, absolute_sum);
                    REQUIRE(std::fabs(actual - expected) <= tolerance);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ streamed graph preserves CSR bytes across blocks", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (uint64_t count : {0, 1, 8191, 8192, 8193, 9000}) {
        std::vector<std::vector<uint64_t>> rows(count);
        for (uint64_t slot = 0; slot < count; ++slot) {
            for (uint64_t edge = 0; edge < slot % 17; ++edge) {
                rows[slot].push_back((slot + edge + 1) % count);
            }
        }
        const auto graph = codec::compact_graph(rows);
        std::ostringstream reference(std::ios::binary);
        codec::write_u64(reference, graph.offsets.size());
        codec::write_u64(reference, graph.neighbors.size());
        codec::write_integer_array(reference, graph.offsets);
        codec::write_integer_array(reference, graph.neighbors);
        for (bool bulk : {false, true}) {
            std::ostringstream actual(std::ios::binary);
            codec::write_graph_adjacency(actual, rows, bulk);
            REQUIRE(actual.str() == reference.str());
        }
        class LimitedBuffer : public std::stringbuf {
        public:
            explicit LimitedBuffer(uint64_t limit) : remaining_(limit) {
            }
            std::streamsize
            xsputn(const char* data, std::streamsize count) override {
                const auto allowed = std::min<uint64_t>(remaining_, static_cast<uint64_t>(count));
                remaining_ -= allowed;
                return std::stringbuf::xsputn(data, static_cast<std::streamsize>(allowed));
            }

        private:
            uint64_t remaining_;
        };
        for (const uint64_t limit :
             {uint64_t{0}, uint64_t{8}, static_cast<uint64_t>(reference.str().size()) - 1}) {
            LimitedBuffer buffer(limit);
            std::ostream output(&buffer);
            REQUIRE_THROWS(codec::write_graph_adjacency(output, rows));
        }
    }
}

TEST_CASE("RaBitQ integer arrays preserve scalar endian bytes and stream errors", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    const auto check = [](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        std::ostringstream scalar(std::ios::binary);
        std::ostringstream block(std::ios::binary);
        codec::write_integer_array(scalar, values, false);
        codec::write_integer_array(block, values);
        REQUIRE(block.str() == scalar.str());
        for (const bool bulk : {false, codec::snapshot_little_endian()}) {
            std::vector<T> loaded(values.size());
            std::istringstream input(block.str(), std::ios::binary);
            codec::read_integer_array(input, loaded, bulk);
            REQUIRE(loaded == values);
            if (not values.empty()) {
                std::istringstream short_input(block.str().substr(0, block.str().size() - 1),
                                               std::ios::binary);
                REQUIRE_THROWS(codec::read_integer_array(short_input, loaded, bulk));
            }
        }
        struct ShortWrite : std::stringbuf {
            std::streamsize
            xsputn(const char* data, std::streamsize count) override {
                return std::stringbuf::xsputn(data, count > 0 ? count - 1 : count);
            }
        } short_buffer;
        std::ostream output(&short_buffer);
        if (not values.empty()) {
            REQUIRE_THROWS(codec::write_integer_array(output, values));
        }
    };
    check(std::vector<int64_t>{INT64_MIN, -1, 0, 1, INT64_MAX});
    check(std::vector<uint64_t>{0, 1, UINT64_MAX, 0x123456789abcdef0ULL});
    check(std::vector<int64_t>{});
    check(std::vector<uint64_t>{});
}

TEST_CASE("RaBitQ routing estimate omits only the unused lower bound", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (uint64_t dim : {uint64_t{1}, uint64_t{17}, uint64_t{128}, uint64_t{768}, uint64_t{960}}) {
        std::vector<float> base(4 * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * .13F);
        }
        const auto model = codec::train(base, 4, dim, 47);
        codec::EncodedRecords codes(dim);
        std::array<codec::EncodedView, 4> views;
        for (uint64_t row = 0; row < 4; ++row) {
            codes.Append(codec::encode(model, base.data() + row * dim));
        }
        for (uint64_t row = 0; row < 4; ++row) {
            views[row] = codes.At(row);
        }
        float norm = 0;
        const auto query = codec::normalize(model, base.data(), norm);
        const auto full_batch = codec::filter_estimate_batch4(query, norm, views);
        const auto route_batch = codec::filter_estimate_batch4<false>(query, norm, views);
        for (uint64_t row = 0; row < 4; ++row) {
            const auto full = codec::filter_estimate(query, norm, views[row]);
            const auto route = codec::filter_estimate<false>(query, norm, views[row]);
            REQUIRE(route.distance == full.distance);
            REQUIRE(route.centered_ip == full.centered_ip);
            REQUIRE(route.lower_bound == 0.0F);
            REQUIRE(route_batch[row].distance == full_batch[row].distance);
            REQUIRE(route_batch[row].centered_ip == full_batch[row].centered_ip);
            REQUIRE(route_batch[row].lower_bound == 0.0F);
        }
    }
}

TEST_CASE("RaBitQ bounded heap admission preserves the unpruned reference", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t count = 96;
    for (uint64_t dim : {uint64_t{1}, uint64_t{17}, uint64_t{128}, uint64_t{768}, uint64_t{960}}) {
        std::vector<float> base(count * dim);
        std::vector<int64_t> ids(count);
        codec::GraphTopology graph;
        graph.offsets.push_back(0);
        for (uint64_t row = 0; row < count; ++row) {
            ids[row] = 4000 - static_cast<int64_t>(row * 77);
            for (uint64_t d = 0; d < dim; ++d) {
                // Repeated vectors exercise equal-distance and non-monotonic ID ties.
                base[row * dim + d] = std::sin(static_cast<float>((row % 16) * 7 + d) * .17F);
            }
            for (uint64_t step : {uint64_t{1}, uint64_t{13}, uint64_t{37}, uint64_t{95}}) {
                graph.neighbors.push_back((row + step) % count);
            }
            graph.offsets.push_back(graph.neighbors.size());
        }
        const auto model = codec::train(base, count, dim, 47);
        codec::EncodedRecords codes(dim);
        for (uint64_t row = 0; row < count; ++row) {
            codes.Append(codec::encode(model, base.data() + row * dim));
        }
        codec::MutableGraphState state(model, codes, graph, ids, 4, 32);
        for (uint64_t epoch = 0; epoch < 2; ++epoch) {
            if (epoch != 0) {
                std::vector<float> changed(dim, 2.25F);
                REQUIRE(state.Update(ids[13], changed.data()));
                REQUIRE(state.Remove(ids[24]));
                REQUIRE(state.Add(-10000, changed.data()));
            }
            const auto topology = state.GetGraph();
            const auto range = [&topology](uint64_t slot) {
                return std::make_pair(
                    topology.neighbors.begin() + static_cast<int64_t>(topology.offsets[slot]),
                    topology.neighbors.begin() + static_cast<int64_t>(topology.offsets[slot + 1]));
            };
            for (uint64_t query_id = 0; query_id < 8; ++query_id) {
                float norm = 0;
                const auto query = codec::normalize(model, base.data() + query_id * dim, norm);
                for (uint64_t ef : {uint64_t{1}, uint64_t{8}, uint64_t{32}, count}) {
                    for (uint64_t k : {uint64_t{1}, uint64_t{10}}) {
                        for (uint64_t filter_mode = 0; filter_mode < 3; ++filter_mode) {
                            std::function<bool(int64_t)> filter;
                            if (filter_mode != 0) {
                                filter = [filter_mode](int64_t id) {
                                    return filter_mode == 1 and id % 3 == 0;
                                };
                            }
                            const auto run = [&](auto prune, auto prefetch, auto select) {
                                return codec::graph_search_impl<decltype(range),
                                                                decltype(prune)::value,
                                                                decltype(prefetch)::value,
                                                                decltype(select)::value>(
                                    query,
                                    norm,
                                    state.GetCodes(),
                                    range,
                                    k,
                                    ef,
                                    &state.GetIds(),
                                    filter);
                            };
                            const auto reference =
                                run(std::false_type{}, std::false_type{}, std::false_type{});
                            if (not filter) {
                                const auto fast = codec::graph_search_impl<decltype(range),
                                                                           true,
                                                                           true,
                                                                           true,
                                                                           true,
                                                                           false>(
                                    query, norm, state.GetCodes(), range, k, ef, &state.GetIds());
                                REQUIRE(fast.visited == reference.visited);
                                REQUIRE(fast.reordered == reference.reordered);
                                REQUIRE(fast.neighbors.size() == reference.neighbors.size());
                                for (uint64_t rank = 0; rank < fast.neighbors.size(); ++rank) {
                                    REQUIRE(fast.neighbors[rank].id ==
                                            reference.neighbors[rank].id);
                                    REQUIRE(fast.neighbors[rank].distance ==
                                            reference.neighbors[rank].distance);
                                }
                            }
                            for (const auto& candidate :
                                 {run(std::true_type{}, std::false_type{}, std::true_type{}),
                                  run(std::true_type{}, std::true_type{}, std::true_type{})}) {
                                REQUIRE(candidate.visited == reference.visited);
                                REQUIRE(candidate.reordered == reference.reordered);
                                REQUIRE(candidate.neighbors.size() == reference.neighbors.size());
                                for (uint64_t rank = 0; rank < candidate.neighbors.size(); ++rank) {
                                    REQUIRE(candidate.neighbors[rank].id ==
                                            reference.neighbors[rank].id);
                                    REQUIRE(candidate.neighbors[rank].distance ==
                                            reference.neighbors[rank].distance);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("RaBitQ row training supports reusable non-contiguous scratch", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (const uint64_t dim : {1ULL, 17ULL, 128ULL}) {
        constexpr uint64_t count = 7;
        std::vector<float> base(count * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i * 13) * .17F);
        }
        const auto expected = codec::train(base, count, dim, 47);
        std::vector<float> scratch(dim);
        uint64_t calls = 0;
        const auto model = codec::train_rows(count, dim, 47, [&](uint64_t row) {
            REQUIRE(row == calls % count);
            ++calls;
            std::copy_n(base.data() + row * dim, dim, scratch.data());
            return scratch.data();
        });
        REQUIRE(calls == count);
        REQUIRE(model.dim == expected.dim);
        REQUIRE(model.flips == expected.flips);
        REQUIRE(std::memcmp(model.centroid.data(), expected.centroid.data(), dim * sizeof(float)) ==
                0);
        for (uint64_t row = 0; row < count; ++row) {
            const auto a = codec::encode(expected, base.data() + row * dim);
            const auto b = codec::encode(model, base.data() + row * dim);
            REQUIRE(a.filter == b.filter);
            REQUIRE(a.supplement == b.supplement);
            REQUIRE(a.scalar == b.scalar);
        }
        for (const float invalid :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            base.back() = invalid;
            REQUIRE_THROWS_AS(
                codec::train_rows(
                    count, dim, 47, [&](uint64_t row) { return base.data() + row * dim; }),
                std::runtime_error);
        }
    }
}

TEST_CASE("RaBitQ skipped removal rows preserve legacy high-dimensional compaction",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t count = 64;
    for (uint64_t dim : {uint64_t{17}, uint64_t{128}, uint64_t{768}}) {
        std::vector<float> base(count * dim);
        std::vector<int64_t> ids(count);
        codec::GraphTopology graph;
        graph.offsets.push_back(0);
        for (uint64_t row = 0; row < count; ++row) {
            ids[row] = 6000 - static_cast<int64_t>(row * 73);
            for (uint64_t d = 0; d < dim; ++d) {
                base[row * dim + d] = std::sin(static_cast<float>(row * 7 + d) * .17F);
            }
            for (uint64_t step : {uint64_t{1}, uint64_t{13}, uint64_t{37}}) {
                graph.neighbors.push_back((row + step) % count);
            }
            graph.offsets.push_back(graph.neighbors.size());
        }
        const auto model = codec::train(base, count, dim, 47);
        codec::EncodedRecords codes(dim);
        for (uint64_t row = 0; row < count; ++row) {
            codes.Append(codec::encode(model, base.data() + row * dim));
        }
        codec::MutableGraphState reference(model, codes, graph, ids, 4, 32);
        auto candidate = reference;
        uint64_t step = 0;
        while (reference.Size() != 0) {
            const uint64_t slot = step % 3 == 0   ? 0
                                  : step % 3 == 1 ? reference.Size() - 1
                                                  : reference.Size() / 2;
            const auto id = reference.IdAt(slot);
            REQUIRE(reference.Remove<false>(id));
            REQUIRE(candidate.RemoveTransactional(id));
            reference.Validate();
            candidate.Validate();
            std::stringstream first;
            std::stringstream second;
            codec::save_mutable_snapshot(first, reference);
            codec::save_mutable_snapshot(second, candidate);
            REQUIRE(first.str() == second.str());
            if (reference.Size() != 0) {
                const auto expected =
                    reference.Search(base.data(), std::min(uint64_t{10}, reference.Size()));
                const auto actual =
                    candidate.Search(base.data(), std::min(uint64_t{10}, candidate.Size()));
                REQUIRE(expected.neighbors.size() == actual.neighbors.size());
                for (uint64_t i = 0; i < expected.neighbors.size(); ++i) {
                    REQUIRE(expected.neighbors[i].id == actual.neighbors[i].id);
                    REQUIRE(expected.neighbors[i].distance == actual.neighbors[i].distance);
                }
            }
            ++step;
        }
    }
}

TEST_CASE("RaBitQ cached incoming counts match recount protection through structural CRUD",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t count = 48;
    for (uint64_t dim : {uint64_t{1}, uint64_t{17}, uint64_t{128}}) {
        for (uint64_t configuration = 0; configuration < 16; ++configuration) {
            const bool incoming = (configuration & 8) != 0;
            const uint64_t stages = configuration & 7;
            std::vector<float> base(count * dim);
            std::vector<int64_t> ids(count);
            codec::GraphTopology graph;
            graph.offsets.push_back(0);
            for (uint64_t row = 0; row < count; ++row) {
                ids[row] = 4000 - static_cast<int64_t>(row * 77);
                for (uint64_t d = 0; d < dim; ++d) {
                    base[row * dim + d] = std::sin(static_cast<float>(row * 7 + d) * .17F);
                }
                for (uint64_t step : {uint64_t{1}, uint64_t{13}, uint64_t{37}}) {
                    graph.neighbors.push_back((row + step) % count);
                }
                graph.offsets.push_back(graph.neighbors.size());
            }
            const auto model = codec::train(base, count, dim, 47);
            codec::EncodedRecords codes(dim);
            for (uint64_t row = 0; row < count; ++row) {
                codes.Append(codec::encode(model, base.data() + row * dim));
            }
            codec::MutableGraphState reference(model, codes, graph, ids, 4, 32, false, incoming);
            auto cached = reference;
            reference.ConfigureIncomingProtection(codec::IncomingProtection::RECOUNT,
                                                  (stages & 1) != 0,
                                                  (stages & 2) != 0,
                                                  (stages & 4) != 0);
            cached.ConfigureIncomingProtection(codec::IncomingProtection::CACHED,
                                               (stages & 1) != 0,
                                               (stages & 2) != 0,
                                               (stages & 4) != 0);
            REQUIRE_THROWS(
                cached.ConfigureIncomingProtection(static_cast<codec::IncomingProtection>(99)));
            const auto check = [&] {
                reference.Validate();
                cached.Validate();
                std::stringstream first;
                std::stringstream second;
                codec::save_mutable_snapshot(first, reference);
                codec::save_mutable_snapshot(second, cached);
                REQUIRE(first.str() == second.str());
                const auto expected = reference.Search(base.data(), 10);
                const auto actual = cached.Search(base.data(), 10);
                REQUIRE(expected.neighbors.size() == actual.neighbors.size());
                for (uint64_t i = 0; i < expected.neighbors.size(); ++i) {
                    REQUIRE(expected.neighbors[i].id == actual.neighbors[i].id);
                    REQUIRE(expected.neighbors[i].distance == actual.neighbors[i].distance);
                }
            };
            for (uint64_t step = 0; step < 120; ++step) {
                const auto slot = step * 7 % cached.Size();
                const auto id = cached.IdAt(slot);
                std::vector<float> changed(dim);
                for (uint64_t d = 0; d < dim; ++d) {
                    changed[d] = std::cos(static_cast<float>(step * 11 + d) * .23F);
                }
                if (step == 40 or step == 80) {
                    REQUIRE(reference.RemoveTransactional(id));
                    REQUIRE(cached.RemoveTransactional(id));
                    check();
                    REQUIRE(reference.AddTransactional(id, changed.data()));
                    REQUIRE(cached.AddTransactional(id, changed.data()));
                    check();
                }
                REQUIRE(reference.UpdateTransactional(
                    id, codec::prepare_encoding(model, changed.data())));
                REQUIRE(
                    cached.UpdateTransactional(id, codec::prepare_encoding(model, changed.data())));
                check();
            }
            REQUIRE(cached.IncomingCountRebuilds() == 1);
            REQUIRE(cached.IncomingCountBytes() >= count * sizeof(uint64_t));
            REQUIRE(cached.GetMemoryUsage().incoming_count_capacity_bytes ==
                    cached.IncomingCountBytes());
            std::stringstream before_failure;
            codec::save_mutable_snapshot(before_failure, cached);
            auto invalid = codec::prepare_encoding(model, base.data());
            invalid.code.filter.clear();
            REQUIRE_THROWS(cached.UpdateTransactional(cached.IdAt(0), std::move(invalid)));
            std::stringstream after_failure;
            codec::save_mutable_snapshot(after_failure, cached);
            REQUIRE(before_failure.str() == after_failure.str());
            cached.Validate();
            REQUIRE(reference.UpdateTransactional(reference.IdAt(0),
                                                  codec::prepare_encoding(model, base.data())));
            REQUIRE(cached.UpdateTransactional(cached.IdAt(0),
                                               codec::prepare_encoding(model, base.data())));
            REQUIRE(cached.IncomingCountRebuilds() == (incoming ? 1 : 2));
            check();
            const auto rebuilt = cached.IncomingCountRebuilds();
            while (cached.Size() != 0) {
                const auto slot = cached.Size() % 2 == 0 ? cached.Size() / 2 : cached.Size() - 1;
                const auto id = cached.IdAt(slot);
                REQUIRE(reference.RemoveTransactional(id));
                REQUIRE(cached.RemoveTransactional(id));
                check();
                REQUIRE(cached.IncomingCountRebuilds() == rebuilt);
            }
            REQUIRE(cached.GetMemoryUsage().incoming_count_logical_bytes == 0);
            for (uint64_t row = 0; row < 65; ++row) {
                std::vector<float> changed(dim, 1.0F + static_cast<float>(row) * .01F);
                const auto id = static_cast<int64_t>(9000 + row * 13);
                REQUIRE(reference.AddTransactional(id, changed.data()));
                REQUIRE(cached.AddTransactional(id, changed.data()));
                check();
                REQUIRE(reference.UpdateTransactional(
                    id, codec::prepare_encoding(model, changed.data())));
                REQUIRE(
                    cached.UpdateTransactional(id, codec::prepare_encoding(model, changed.data())));
                check();
                REQUIRE(cached.IncomingCountRebuilds() == rebuilt);
            }
            cached.ConfigureIncomingProtection(codec::IncomingProtection::NONE);
            REQUIRE(cached.UpdateTransactional(cached.IdAt(0),
                                               codec::prepare_encoding(model, base.data())));
            cached.Validate();
        }
    }
}

TEST_CASE("RaBitQ safe link pruning respects prior incoming counts", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    const std::vector<codec::Candidate> ranked{{0, 0.0F}, {1, 1.0F}, {2, 2.0F}};
    REQUIRE(codec::safe_incoming_drop(ranked, {2, 1, 0}, 2) == 0);
    REQUIRE(codec::safe_incoming_drop(ranked, {1, 2, 0}, 2) == 1);
    REQUIRE(codec::safe_incoming_drop(ranked, {1, 1, 1}, 2) == 2);
    REQUIRE(codec::safe_incoming_drop(ranked, {1, 1, 0}, 2) == 2);
    REQUIRE_THROWS(codec::safe_incoming_drop({}, {}, 0));
    REQUIRE_THROWS(codec::safe_incoming_drop(ranked, {1}, 2));
}

TEST_CASE("RaBitQ safe link pruning preserves cache and mutable persistence", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (bool reverse : {false, true}) {
        const std::vector<float> base{0.0F, 100.0F, 1.0F, 2.0F};
        const auto model = codec::train(base, 4, 1, 47);
        codec::EncodedRecords codes(1);
        for (const float value : base) {
            codes.Append(codec::encode(model, &value));
        }
        const codec::GraphTopology graph{{0, 2, 4, 6, 8}, {1, 2, 2, 3, 0, 3, 0, 2}};
        codec::MutableGraphState state(model, codes, graph, {0, 1, 2, 3}, 2, 8, false, reverse);
        REQUIRE_THROWS(state.ConfigureIncomingProtection(
            codec::IncomingProtection::NONE, false, false, false, true));
        state.ConfigureIncomingProtection(
            codec::IncomingProtection::CACHED, false, false, false, true);
        codec::MutableGraphState control(model, codes, graph, {0, 1, 2, 3}, 2, 8, false, reverse);
        control.ConfigureIncomingProtection(codec::IncomingProtection::CACHED);
        REQUIRE(control.AddTransactional(4, base.data()));
        REQUIRE(state.AddTransactional(4, base.data()));
        const auto control_graph = control.GetGraph();
        const auto protected_graph = state.GetGraph();
        REQUIRE(std::count(control_graph.neighbors.begin(), control_graph.neighbors.end(), 1) == 0);
        REQUIRE(std::count(protected_graph.neighbors.begin(), protected_graph.neighbors.end(), 1) ==
                1);
        for (int64_t id = 5; id < 28; ++id) {
            const float changed = static_cast<float>(id) * .17F;
            REQUIRE(state.AddTransactional(id, &changed));
            REQUIRE(state.UpdateTransactional(id - 1, codec::prepare_encoding(model, &changed)));
            if (id % 3 == 0) {
                REQUIRE(state.RemoveTransactional(id - 2));
            }
            state.Validate();
            std::stringstream saved;
            codec::save_mutable_snapshot(saved, state);
            auto loaded = codec::load_mutable_snapshot(saved);
            loaded.Validate();
            REQUIRE(loaded.GetIds() == state.GetIds());
            REQUIRE(loaded.GetGraph().neighbors == state.GetGraph().neighbors);
        }
        state.ConfigureIncomingProtection(codec::IncomingProtection::NONE);
        REQUIRE(state.AddTransactional(100, base.data()));
        state.Validate();
    }
}

TEST_CASE("RaBitQ Add can protect targets displaced from saturated reverse links",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    const std::vector<float> base{0.0F, 100.0F, 1.0F, 2.0F};
    const auto model = codec::train(base, 4, 1, 47);
    codec::EncodedRecords codes(1);
    for (const float value : base) {
        codes.Append(codec::encode(model, &value));
    }
    const codec::GraphTopology graph{{0, 2, 4, 6, 8}, {1, 2, 2, 3, 0, 3, 0, 2}};
    const auto incoming = [](const codec::MutableGraphState& state, uint64_t target) {
        const auto current = state.GetGraph();
        return std::count(current.neighbors.begin(), current.neighbors.end(), target);
    };
    for (bool reverse : {false, true}) {
        for (bool transactional : {false, true}) {
            codec::MutableGraphState control(
                model, codes, graph, {0, 1, 2, 3}, 2, 32, false, reverse);
            auto candidate = control;
            control.ConfigureIncomingProtection(codec::IncomingProtection::CACHED, true, true);
            candidate.ConfigureIncomingProtection(
                codec::IncomingProtection::CACHED, true, true, true);
            const float added = .5F;
            REQUIRE(incoming(control, 1) == 1);
            REQUIRE((transactional ? control.AddTransactional(4, &added) : control.Add(4, &added)));
            REQUIRE(
                (transactional ? candidate.AddTransactional(4, &added) : candidate.Add(4, &added)));
            control.Validate();
            candidate.Validate();
            REQUIRE(incoming(control, 1) == 0);
            REQUIRE(incoming(candidate, 1) == 1);
            REQUIRE(incoming(candidate, 4) > 0);
        }
    }
}

TEST_CASE("RaBitQ saturated link scratch covers the maximum degree and tied codes",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t count = 65;
    constexpr uint64_t dim = 17;
    std::vector<float> base(count * dim);
    codec::GraphTopology graph;
    graph.offsets.push_back(0);
    std::vector<int64_t> ids(count);
    for (uint64_t row = 0; row < count; ++row) {
        ids[row] = static_cast<int64_t>(row);
        for (uint64_t d = 0; d < dim; ++d) {
            base[row * dim + d] = static_cast<float>((row % 5 + d) % 7) * .25F;
        }
        for (uint64_t target = 0; target < count; ++target) {
            if (target != row) {
                graph.neighbors.push_back(target);
            }
        }
        graph.offsets.push_back(graph.neighbors.size());
    }
    const auto model = codec::train(base, count, dim, 47);
    codec::EncodedRecords codes(dim);
    for (uint64_t row = 0; row < count; ++row) {
        codes.Append(codec::encode(model, base.data() + row * dim));
    }
    for (bool reverse : {false, true}) {
        codec::MutableGraphState state(model, codes, graph, ids, 64, 128, false, reverse);
        state.ConfigureIncomingProtection(codec::IncomingProtection::CACHED, true, true, true);
        REQUIRE(state.AddTransactional(65, base.data()));
        REQUIRE(state.UpdateTransactional(0, codec::prepare_encoding(model, base.data() + dim)));
        state.Validate();
        const auto current = state.GetGraph();
        for (uint64_t row = 0; row < state.Size(); ++row) {
            REQUIRE(current.offsets[row + 1] - current.offsets[row] <= 64);
        }
        std::stringstream output;
        codec::save_mutable_snapshot(output, state);
        auto restored = codec::load_mutable_snapshot(output);
        const auto expected = state.Search(base.data(), state.Size());
        const auto actual = restored.Search(base.data(), restored.Size());
        REQUIRE(expected.neighbors.size() == 66);
        REQUIRE(actual.neighbors.size() == expected.neighbors.size());
        for (uint64_t i = 0; i < expected.neighbors.size(); ++i) {
            REQUIRE(actual.neighbors[i].id == expected.neighbors[i].id);
            REQUIRE(actual.neighbors[i].distance == expected.neighbors[i].distance);
        }
        REQUIRE(state.RemoveTransactional(3));
        state.Validate();
    }
}

TEST_CASE("RaBitQ maintenance Batch4 preserves scalar candidate order and distance bits",
          "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    for (const uint64_t dim : {1, 7, 8, 9, 15, 16, 17, 33, 128, 129, 768, 960, 961}) {
        constexpr uint64_t count = 65;
        std::vector<float> base(count * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i % (11 * dim)) * 0.19F);
        }
        const auto model = codec::train(base, count, dim, 91);
        codec::EncodedRecords records(dim);
        for (uint64_t slot = 0; slot < count; ++slot) {
            records.Append(codec::encode(model, base.data() + slot * dim));
        }
        float norm = 0.0F;
        const auto query = codec::normalize(model, base.data(), norm);
        const float sum = std::accumulate(query.begin(), query.end(), 0.0F);
        for (uint64_t length = 0; length <= count; ++length) {
            std::vector<codec::Candidate> batch;
            for (uint64_t i = 0; i < length; ++i) {
                batch.push_back({count - 1 - i, -1.0F});
            }
            auto scalar = batch;
            codec::score_mutation_candidates(query, norm, sum, records, batch);
            codec::score_mutation_candidates<false>(query, norm, sum, records, scalar);
            for (uint64_t i = 0; i < length; ++i) {
                REQUIRE(batch[i].id == scalar[i].id);
                REQUIRE(std::memcmp(reinterpret_cast<const unsigned char*>(&batch[i].distance),
                                    reinterpret_cast<const unsigned char*>(&scalar[i].distance),
                                    sizeof(float)) == 0);
            }
            std::sort(batch.begin(), batch.end(), codec::better);
            std::sort(scalar.begin(), scalar.end(), codec::better);
            for (uint64_t i = 0; i < length; ++i) {
                REQUIRE(batch[i].id == scalar[i].id);
            }
        }
    }
}

TEST_CASE("RaBitQ incoming slot narrowing is checked", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    static_assert(sizeof(codec::IncomingSlot) == sizeof(uint32_t));
    const auto maximum = std::numeric_limits<uint32_t>::max();
    REQUIRE(codec::checked_incoming_slot(maximum) == maximum);
    REQUIRE_THROWS(codec::checked_incoming_slot(static_cast<uint64_t>(maximum) + 1));
}

TEST_CASE("RaBitQ contiguous incoming validation checks exact edge multisets", "[lite-rabitq]") {
    namespace codec = vsag::lite::detail::rabitq;
    REQUIRE_NOTHROW(codec::validate_incoming_adjacency({}, {}));
    REQUIRE_NOTHROW(codec::validate_incoming_adjacency({{}}, {{}}));
    const std::vector<std::vector<uint64_t>> outgoing{{1, 2}, {2}, {0}, {2}};
    const std::vector<std::vector<codec::IncomingSlot>> incoming{{2}, {0}, {3, 0, 1}, {}};
    REQUIRE_NOTHROW(codec::validate_incoming_adjacency(outgoing, incoming));
    auto damaged = incoming;
    damaged.pop_back();
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    damaged = incoming;
    damaged[2].pop_back();
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    damaged = incoming;
    damaged[2].push_back(0);
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    damaged = incoming;
    damaged[2][0] = 0;
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    damaged = incoming;
    damaged[2][0] = std::numeric_limits<codec::IncomingSlot>::max();
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    damaged = incoming;
    damaged[3].push_back(0);
    REQUIRE_THROWS(codec::validate_incoming_adjacency(outgoing, damaged));
    auto invalid = outgoing;
    invalid[0].push_back(outgoing.size());
    REQUIRE_THROWS(codec::validate_incoming_adjacency(invalid, incoming));
    std::vector<std::vector<uint64_t>> skewed(257);
    std::vector<std::vector<codec::IncomingSlot>> reverse(257);
    for (uint64_t source = 1; source < skewed.size(); ++source) {
        skewed[source].push_back(0);
        reverse[0].push_back(codec::checked_incoming_slot(source));
    }
    std::reverse(reverse[0].begin(), reverse[0].end());
    REQUIRE_NOTHROW(codec::validate_incoming_adjacency(skewed, reverse));
}
