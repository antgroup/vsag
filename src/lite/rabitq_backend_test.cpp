// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
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
            state.CompactIncoming();
            REQUIRE(state.GetIncomingMemoryUsage().logical_bytes == before.logical_bytes);
            REQUIRE(state.GetIncomingDistribution().nodes == 8);
        }
        std::stringstream stream;
        codec::save_mutable_snapshot(stream, state);
        auto loaded = codec::load_mutable_snapshot(stream);
        loaded.Validate();
        REQUIRE(loaded.GetIds() == state.GetIds());
        REQUIRE(loaded.GetGraph().neighbors == state.GetGraph().neighbors);
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
            REQUIRE(copy.Remove(id));
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
