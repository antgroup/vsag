// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lite/backend.h"

using vsag::lite::detail::make_brute_force_backend;
using vsag::lite::detail::make_fp16_graph_backend;
using vsag::lite::detail::make_graph_backend;

TEST_CASE("Lite graph backend validates input and survives CRUD", "[lite-graph]") {
    auto flat = make_brute_force_backend(2);
    REQUIRE(flat);
    const std::array<float, 2> a{0, 0};
    const std::array<float, 2> b{1, 0};
    const std::array<float, 2> c{0, 1};
    REQUIRE((*flat)->Add(1, a.data(), 2));
    REQUIRE((*flat)->Add(2, b.data(), 2));
    REQUIRE((*flat)->Add(3, c.data(), 2));
    REQUIRE_FALSE(make_graph_backend(**flat, 1, 8));
    REQUIRE_FALSE(make_graph_backend(**flat, 8, 2));
    auto graph = make_graph_backend(**flat, 4, 32);
    REQUIRE(graph);
    REQUIRE((*graph)->Size() == 3);
    const std::array<float, 2> nonfinite{std::numeric_limits<float>::quiet_NaN(), 0};
    REQUIRE_FALSE((*graph)->Add(4, nonfinite.data(), 2));
    REQUIRE_FALSE((*graph)->Update(2, nonfinite.data(), 2));
    REQUIRE_FALSE((*graph)->Search(nonfinite.data(), 2, 1));
    REQUIRE_FALSE((*graph)->Search(nullptr, 2, 1));
    REQUIRE((*graph)->Search(a.data(), 2, 0)->empty());
    REQUIRE((*graph)->Search(a.data(), 2, 3)->front().id == 1);
    REQUIRE_FALSE((*graph)->Add(1, a.data(), 2));
    REQUIRE_FALSE((*graph)->Add(4, nullptr, 2));
    REQUIRE_FALSE((*graph)->Add(4, a.data(), 1));
    REQUIRE_FALSE((*graph)->Update(9, a.data(), 2));
    REQUIRE((*graph)->Update(2, a.data(), 2));
    REQUIRE((*graph)->Search(a.data(), 2, 2)->size() == 2);
    REQUIRE((*graph)->Remove(1));
    REQUIRE_FALSE((*graph)->Remove(1));
    REQUIRE((*graph)->Size() == 2);
    for (uint64_t slot = 0; slot < (*graph)->Size(); ++slot) {
        for (uint64_t edge = 0; edge < (*graph)->LinkCountAt(slot); ++edge) {
            REQUIRE((*graph)->LinkAt(slot, edge) != slot);
        }
    }
    const auto result = (*graph)->Search(a.data(), 2, 2);
    REQUIRE(result);
    REQUIRE(result->size() == 2);
    REQUIRE(std::none_of(result->begin(), result->end(), [](const auto& n) { return n.id == 1; }));
    REQUIRE((*graph)->Add(1, b.data(), 2));
    REQUIRE((*graph)->Search(b.data(), 2, 3)->size() == 3);
}

TEST_CASE("Lite graph removal cleans asymmetric restored links", "[lite-graph]") {
    using vsag::lite::detail::restore_graph_backend;
    auto graph = restore_graph_backend(
        2, 2, 8, {10, 11, 12, 13}, {0, 0, 1, 0, 0, 1, 1, 1}, {{1}, {}, {3}, {}});
    REQUIRE(graph);
    REQUIRE((*graph)->Remove(11));
    REQUIRE((*graph)->Size() == 3);
    REQUIRE((*graph)->LinkCountAt(0) == 0);
    REQUIRE((*graph)->LinkCountAt(2) == 1);
    REQUIRE((*graph)->LinkAt(2, 0) == 1);
    for (uint64_t slot = 0; slot < (*graph)->Size(); ++slot) {
        for (uint64_t edge = 0; edge < (*graph)->LinkCountAt(slot); ++edge) {
            REQUIRE((*graph)->LinkAt(slot, edge) < (*graph)->Size());
            REQUIRE((*graph)->LinkAt(slot, edge) != slot);
        }
    }
}

TEST_CASE("Lite FP16 VectorAt uses caller-owned scratch", "[lite-graph]") {
    auto flat = make_brute_force_backend(2);
    REQUIRE(flat);
    const std::array<float, 2> first_vector{0.125F, -0.25F};
    const std::array<float, 2> second_vector{1.5F, 2.0F};
    REQUIRE((*flat)->Add(1, first_vector.data(), 2));
    REQUIRE((*flat)->Add(2, second_vector.data(), 2));
    auto graph = make_fp16_graph_backend(**flat, 4, 32);
    REQUIRE(graph);

    std::vector<float> first_scratch;
    std::vector<float> second_scratch;
    const float* first = (*graph)->VectorAt(0, first_scratch);
    const float* second = (*graph)->VectorAt(1, second_scratch);
    REQUIRE(first == first_scratch.data());
    REQUIRE(second == second_scratch.data());
    REQUIRE(first[0] == first_vector[0]);
    REQUIRE(first[1] == first_vector[1]);
    REQUIRE(second[0] == second_vector[0]);
    REQUIRE(second[1] == second_vector[1]);
}

TEST_CASE("Lite graph filter traverses rejected IDs and survives snapshot load", "[lite-graph]") {
    auto created = vsag::lite::Index::Create(2);
    REQUIRE(created);
    auto& index = **created;
    std::array<float, 2> vector{};
    for (int64_t slot = 0; slot < 32; ++slot) {
        vector[0] = static_cast<float>(slot);
        REQUIRE(index.Add(1000 + slot, vector.data(), 2));
    }
    REQUIRE(index.BuildGraph(4, 32));
    const std::array<float, 2> query{15, 0};
    const auto unfiltered = index.Search(query.data(), 2, 8);
    REQUIRE(unfiltered);
    const vsag::lite::IdFilter empty_filter;
    const auto without_filter = index.Search(query.data(), 2, 8, empty_filter);
    REQUIRE(without_filter);
    REQUIRE(without_filter->size() == unfiltered->size());
    for (uint64_t i = 0; i < unfiltered->size(); ++i) {
        REQUIRE((*without_filter)[i].id == (*unfiltered)[i].id);
        REQUIRE((*without_filter)[i].distance == (*unfiltered)[i].distance);
    }

    const auto closest_even =
        index.Search(query.data(), 2, 3, [](int64_t id) { return id % 2 == 0; });
    REQUIRE(closest_even);
    REQUIRE(closest_even->size() == 3);
    REQUIRE((*closest_even)[0].id == 1014);
    REQUIRE((*closest_even)[1].id == 1016);
    REQUIRE((*closest_even)[2].id == 1012);
    REQUIRE((*closest_even)[0].distance == 1);
    REQUIRE((*closest_even)[1].distance == 1);
    REQUIRE((*closest_even)[2].distance == 9);

    std::unordered_set<int64_t> checked_ids;
    const vsag::lite::IdFilter only_middle = [&checked_ids](int64_t id) {
        checked_ids.insert(id);
        return id == 1015;
    };
    const auto selected = index.Search(query.data(), 2, 8, only_middle);
    REQUIRE(selected);
    REQUIRE(selected->size() == 1);
    REQUIRE(selected->front().id == 1015);
    REQUIRE(selected->front().distance == 0);
    REQUIRE(checked_ids.count(1015) == 1);
    REQUIRE(checked_ids.size() > 1);

    const auto rejected = index.Search(query.data(), 2, 8, [](int64_t) { return false; });
    REQUIRE(rejected);
    REQUIRE(rejected->empty());
    REQUIRE_FALSE(index.Search(nullptr, 2, 8, only_middle));
    REQUIRE(index.Search(query.data(), 2, 0, only_middle)->empty());

    const std::array<float, 2> updated{14.5F, 0};
    REQUIRE(index.Update(1015, updated.data(), 2));
    REQUIRE(index.Remove(1001));
    const auto after_crud = index.Search(query.data(), 2, 8, only_middle);
    REQUIRE(after_crud);
    REQUIRE(after_crud->size() == 1);
    REQUIRE(after_crud->front().id == 1015);
    REQUIRE(after_crud->front().distance == 0.25F);

    std::stringstream snapshot;
    REQUIRE(index.Save(snapshot));
    auto loaded = vsag::lite::Index::Load(snapshot);
    REQUIRE(loaded);
    const auto restored = (*loaded)->Search(query.data(), 2, 8, only_middle);
    REQUIRE(restored);
    REQUIRE(restored->size() == after_crud->size());
    REQUIRE(restored->front().id == after_crud->front().id);
    REQUIRE(restored->front().distance == after_crud->front().distance);
}

TEST_CASE("Lite graph update removes obsolete reverse links", "[lite-graph]") {
    auto flat = make_brute_force_backend(2);
    REQUIRE(flat);
    for (int64_t id = 0; id < 8; ++id) {
        const std::array<float, 2> vector{static_cast<float>(id), 0};
        REQUIRE((*flat)->Add(id, vector.data(), 2));
    }
    auto graph = make_graph_backend(**flat, 2, 8);
    REQUIRE(graph);

    constexpr uint64_t updated_slot = 0;
    std::unordered_set<uint64_t> old_neighbors;
    for (uint64_t i = 0; i < (*graph)->LinkCountAt(updated_slot); ++i) {
        old_neighbors.insert((*graph)->LinkAt(updated_slot, i));
    }

    const std::array<float, 2> moved{100, 0};
    REQUIRE((*graph)->Update(0, moved.data(), 2));
    std::unordered_set<uint64_t> new_neighbors;
    for (uint64_t i = 0; i < (*graph)->LinkCountAt(updated_slot); ++i) {
        new_neighbors.insert((*graph)->LinkAt(updated_slot, i));
    }

    uint64_t obsolete = 0;
    for (uint64_t old_neighbor : old_neighbors) {
        if (new_neighbors.count(old_neighbor) != 0) {
            continue;
        }
        ++obsolete;
        bool still_linked = false;
        for (uint64_t i = 0; i < (*graph)->LinkCountAt(old_neighbor); ++i) {
            still_linked |= (*graph)->LinkAt(old_neighbor, i) == updated_slot;
        }
        REQUIRE_FALSE(still_linked);
    }
    REQUIRE(obsolete > 0);
}

TEST_CASE("Lite graph backend validates restored adjacency bounds", "[lite-graph]") {
    using vsag::lite::detail::restore_graph_backend;
    REQUIRE_FALSE(restore_graph_backend(0, 2, 16, {}, {}, {}));
    REQUIRE_FALSE(restore_graph_backend(
        2, 2, 16, {1, 2, 3, 4}, {0, 0, 1, 0, 0, 1, 1, 1}, {{1, 2, 3}, {0}, {0}, {0}}));
    auto empty = make_brute_force_backend(2);
    REQUIRE(empty);
    auto graph = make_graph_backend(**empty, 4, 32);
    REQUIRE(graph);
    const std::array<float, 2> query{0, 0};
    auto result = (*graph)->Search(query.data(), 2, 1);
    REQUIRE(result);
    REQUIRE(result->empty());
}

TEST_CASE("Lite graph backend randomized CRUD keeps IDs and distances valid", "[lite-graph]") {
    constexpr uint64_t dim = 8;
    auto flat = vsag::lite::Index::Create(dim);
    REQUIRE(flat);
    std::mt19937 rng(20260915);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::unordered_map<int64_t, std::array<float, dim>> active;
    for (int64_t id = 0; id < 200; ++id) {
        std::array<float, dim> values{};
        for (float& value : values) {
            value = uniform(rng);
        }
        REQUIRE((*flat)->Add(id, values.data(), dim));
        active.emplace(id, values);
    }
    auto graph = std::move(*flat);
    REQUIRE(graph->BuildGraph(12, 64));
    int64_t next_id = 200;
    for (uint64_t step = 0; step < 500; ++step) {
        const auto selected =
            std::next(active.begin(), static_cast<std::ptrdiff_t>(rng() % active.size()))->first;
        if (step % 3 == 0) {
            REQUIRE(graph->Remove(selected));
            active.erase(selected);
        } else {
            std::array<float, dim> values{};
            for (float& value : values) {
                value = uniform(rng);
            }
            if (step % 3 == 1) {
                REQUIRE(graph->Update(selected, values.data(), dim));
                active.at(selected) = values;
            } else {
                REQUIRE(graph->Add(next_id, values.data(), dim));
                active.emplace(next_id++, values);
            }
        }
        REQUIRE(graph->Size() == active.size());
        std::array<float, dim> query{};
        for (float& value : query) {
            value = uniform(rng);
        }
        const auto result = graph->Search(query.data(), dim, 10);
        REQUIRE(result);
        REQUIRE(result->size() == std::min<uint64_t>(10, active.size()));
        std::unordered_set<int64_t> seen;
        for (const auto& neighbor : *result) {
            REQUIRE(seen.insert(neighbor.id).second);
            const auto& vector = active.at(neighbor.id);
            float distance = 0;
            for (uint64_t d = 0; d < dim; ++d) {
                const float delta = query[d] - vector[d];
                distance += delta * delta;
            }
            REQUIRE(std::abs(distance - neighbor.distance) < 1e-4F);
        }
    }
    std::stringstream output;
    REQUIRE(graph->Save(output));
    auto restored = vsag::lite::Index::Load(output);
    REQUIRE(restored);
    REQUIRE((*restored)->ActiveBackend() == vsag::lite::BackendKind::GRAPH);
    REQUIRE((*restored)->Size() == graph->Size());
    for (uint64_t q = 0; q < 20; ++q) {
        std::array<float, dim> query{};
        for (float& value : query) {
            value = uniform(rng);
        }
        auto before = graph->Search(query.data(), dim, 10);
        auto after = (*restored)->Search(query.data(), dim, 10);
        REQUIRE(before);
        REQUIRE(after);
        REQUIRE(before->size() == after->size());
        for (uint64_t i = 0; i < before->size(); ++i) {
            REQUIRE((*before)[i].id == (*after)[i].id);
            REQUIRE((*before)[i].distance == (*after)[i].distance);
        }
    }
}

TEST_CASE("Lite graph backend recall on independent queries", "[lite-graph]") {
    constexpr uint64_t count = 1000;
    constexpr uint64_t dim = 16;
    constexpr uint64_t queries = 100;
    constexpr uint64_t k = 10;
    std::mt19937 rng(20260914);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    auto flat = make_brute_force_backend(dim);
    REQUIRE(flat);
    std::vector<float> values(count * dim);
    for (uint64_t id = 0; id < count; ++id) {
        for (uint64_t d = 0; d < dim; ++d) {
            values[id * dim + d] = normal(rng);
        }
        REQUIRE((*flat)->Add(id, values.data() + id * dim, dim));
    }
    auto graph = make_graph_backend(**flat, 16, 128);
    REQUIRE(graph);
    uint64_t hits = 0;
    std::array<float, dim> query{};
    for (uint64_t q = 0; q < queries; ++q) {
        for (float& value : query) {
            value = normal(rng);
        }
        auto exact = (*flat)->Search(query.data(), dim, k);
        auto approximate = (*graph)->Search(query.data(), dim, k);
        REQUIRE(exact);
        REQUIRE(approximate);
        REQUIRE(approximate->size() == k);
        for (const auto& neighbor : *approximate) {
            hits += static_cast<uint64_t>(
                std::any_of(exact->begin(), exact->end(), [&](const auto& expected) {
                    return expected.id == neighbor.id;
                }));
        }
    }
    INFO("Recall@10=" << static_cast<double>(hits) / (queries * k));
    REQUIRE(hits >= 700);
}

namespace {
template <typename T>
std::vector<std::vector<T>>
read_sift(const std::string& path, int32_t expected_dim) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input);
    std::vector<std::vector<T>> rows;
    int32_t dim = 0;
    while (input.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        REQUIRE(dim == expected_dim);
        std::vector<T> row(dim);
        input.read(reinterpret_cast<char*>(row.data()), sizeof(T) * dim);
        REQUIRE(input);
        rows.push_back(std::move(row));
    }
    REQUIRE(input.eof());
    return rows;
}
}  // namespace

TEST_CASE("Lite public FP16 graph CRUD and v3 snapshot", "[lite-graph]") {
    using vsag::lite::Index;
    using vsag::lite::VectorStorage;
    auto created = Index::Create(2);
    const std::array<float, 2> a{0.125F, -0.25F};
    const std::array<float, 2> b{1.5F, 2.0F};
    const std::array<float, 2> c{-3.0F, 4.0F};
    REQUIRE((*created)->Add(10, a.data(), 2));
    REQUIRE((*created)->Add(20, b.data(), 2));
    REQUIRE((*created)->Add(30, c.data(), 2));
    REQUIRE((*created)->BuildGraph(VectorStorage::FP16, 4, 32));
    REQUIRE((*created)->ActiveVectorStorage() == VectorStorage::FP16);
    REQUIRE((*created)->Update(20, a.data(), 2));
    REQUIRE((*created)->Remove(30));
    REQUIRE((*created)->Add(40, c.data(), 2));
    const auto before = (*created)->Search(a.data(), 2, 3, [](int64_t id) { return id != 10; });
    REQUIRE(before);
    std::stringstream output;
    REQUIRE((*created)->Save(output));
    const auto bytes = output.str();
    REQUIRE(static_cast<unsigned char>(bytes[8]) == 3);
    REQUIRE(static_cast<unsigned char>(bytes[40]) == 3);
    auto loaded = Index::Load(output);
    REQUIRE(loaded);
    REQUIRE((*loaded)->ActiveVectorStorage() == VectorStorage::FP16);
    const auto after = (*loaded)->Search(a.data(), 2, 3, [](int64_t id) { return id != 10; });
    REQUIRE(after);
    REQUIRE(after->size() == before->size());
    for (uint64_t i = 0; i < after->size(); ++i) {
        REQUIRE((*after)[i].id == (*before)[i].id);
        REQUIRE((*after)[i].distance == (*before)[i].distance);
    }
    for (uint64_t n = 0; n < bytes.size(); ++n) {
        std::stringstream truncated(bytes.substr(0, n));
        REQUIRE_FALSE(Index::Load(truncated));
    }
    auto nonfinite = bytes;
    constexpr uint64_t vector_offset = 48 + 16 + 3 * 8;
    nonfinite[vector_offset] = 1;
    nonfinite[vector_offset + 1] = static_cast<char>(0x7c);
    std::stringstream invalid_half(nonfinite);
    REQUIRE_FALSE(Index::Load(invalid_half));
    auto bad_representation = bytes;
    bad_representation[40] = 2;
    std::stringstream invalid_representation(bad_representation);
    REQUIRE_FALSE(Index::Load(invalid_representation));
}

TEST_CASE("Lite graph public transition and v2 snapshot roundtrip", "[lite-graph]") {
    using vsag::lite::BackendKind;
    using vsag::lite::Index;
    auto created = Index::Create(2);
    REQUIRE(created);
    auto& index = **created;
    REQUIRE(index.ActiveBackend() == BackendKind::BRUTE_FORCE);
    std::array<float, 2> a{0, 0};
    std::array<float, 2> b{1, 0};
    std::array<float, 2> c{0, 1};
    REQUIRE(index.Add(-1, a.data(), 2));
    REQUIRE(index.Add(2, b.data(), 2));
    REQUIRE(index.Add(3, c.data(), 2));
    REQUIRE_FALSE(index.BuildGraph(1, 8));
    REQUIRE(index.ActiveBackend() == BackendKind::BRUTE_FORCE);
    REQUIRE(index.Size() == 3);
    REQUIRE(index.BuildGraph(4, 32));
    REQUIRE(index.ActiveBackend() == BackendKind::GRAPH);
    REQUIRE_FALSE(index.BuildGraph());
    REQUIRE(index.Update(2, a.data(), 2));
    REQUIRE(index.Remove(3));
    REQUIRE(index.Add(4, c.data(), 2));
    std::stringstream output;
    REQUIRE(index.Save(output));
    const auto bytes = output.str();
    REQUIRE(bytes.substr(0, 8) == "VSAGLT01");
    REQUIRE(static_cast<unsigned char>(bytes[8]) == 2);
    REQUIRE(static_cast<unsigned char>(bytes[40]) == 2);
    std::stringstream input(bytes);
    auto loaded = Index::Load(input);
    REQUIRE(loaded);
    REQUIRE((*loaded)->ActiveBackend() == BackendKind::GRAPH);
    REQUIRE((*loaded)->Size() == index.Size());
    for (const auto& query : {a, b, c}) {
        auto before = index.Search(query.data(), 2, 3);
        auto after = (*loaded)->Search(query.data(), 2, 3);
        REQUIRE(before);
        REQUIRE(after);
        REQUIRE(before->size() == after->size());
        for (uint64_t i = 0; i < before->size(); ++i) {
            REQUIRE((*before)[i].id == (*after)[i].id);
            REQUIRE((*before)[i].distance == (*after)[i].distance);
        }
    }
    auto corrupt = bytes;
    corrupt.push_back('x');
    std::stringstream extra(corrupt);
    REQUIRE_FALSE(Index::Load(extra));
    for (uint64_t n = 0; n < bytes.size(); ++n) {
        std::stringstream truncated(bytes.substr(0, n));
        REQUIRE_FALSE(Index::Load(truncated));
    }
    corrupt = bytes;
    corrupt[48] = 1;  // degree below 2
    std::stringstream bad_degree(corrupt);
    REQUIRE_FALSE(Index::Load(bad_degree));
    corrupt = bytes;
    corrupt.replace(72, 8, corrupt.substr(64, 8));  // duplicate ID
    std::stringstream duplicate_id(corrupt);
    REQUIRE_FALSE(Index::Load(duplicate_id));
    corrupt = bytes;
    corrupt[88] = 0;
    corrupt[89] = 0;
    corrupt[90] = static_cast<char>(0xc0);
    corrupt[91] = 0x7f;  // NaN vector
    std::stringstream nonfinite(corrupt);
    REQUIRE_FALSE(Index::Load(nonfinite));
    corrupt = bytes;
    corrupt[112] = 65;  // link count above maximum degree
    std::stringstream bad_degree_count(corrupt);
    REQUIRE_FALSE(Index::Load(bad_degree_count));
    corrupt = bytes;
    REQUIRE(static_cast<unsigned char>(corrupt[112]) > 0);
    corrupt[120] = static_cast<char>(0xff);  // first adjacency target outside index
    std::stringstream bad_neighbor(corrupt);
    REQUIRE_FALSE(Index::Load(bad_neighbor));
    corrupt = bytes;
    corrupt[120] = 0;  // self-link at slot zero
    std::stringstream self_link(corrupt);
    REQUIRE_FALSE(Index::Load(self_link));
    corrupt = bytes;
    REQUIRE(static_cast<unsigned char>(corrupt[112]) >= 2);
    corrupt.replace(128, 8, corrupt.substr(120, 8));  // duplicate adjacency target
    std::stringstream duplicate_link(corrupt);
    REQUIRE_FALSE(Index::Load(duplicate_link));
    corrupt = bytes;
    corrupt[56] = 1;  // ef_search below degree
    std::stringstream bad_ef(corrupt);
    REQUIRE_FALSE(Index::Load(bad_ef));
    corrupt = bytes;
    corrupt[32] ^= 1;  // mismatched payload length
    std::stringstream bad_payload(corrupt);
    REQUIRE_FALSE(Index::Load(bad_payload));

    auto empty = Index::Create(2);
    REQUIRE((*empty)->BuildGraph());
    std::stringstream empty_output;
    REQUIRE((*empty)->Save(empty_output));
    auto empty_loaded = Index::Load(empty_output);
    REQUIRE(empty_loaded);
    REQUIRE((*empty_loaded)->ActiveBackend() == BackendKind::GRAPH);
    REQUIRE((*empty_loaded)->Size() == 0);
}

TEST_CASE("Lite graph SIFT independent-query probe", "[.][lite-sift]") {
    const char* directory = std::getenv("VSAG_SIFT_DIR");
    REQUIRE(directory != nullptr);
    const std::string root(directory);
    const auto base = read_sift<float>(root + "/base.fvecs", 128);
    const auto queries = read_sift<float>(root + "/queries.fvecs", 128);
    const auto truth = read_sift<int32_t>(root + "/groundtruth.ivecs", 10);
    REQUIRE(queries.size() == truth.size());
    auto graph = vsag::lite::Index::Create(128);
    REQUIRE(graph);
    for (uint64_t id = 0; id < base.size(); ++id) {
        REQUIRE((*graph)->Add(id, base[id].data(), 128));
    }
    auto start = std::chrono::steady_clock::now();
    REQUIRE((*graph)->BuildGraph(16, 128));
    const auto build_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::vector<double> latency;
    std::vector<std::vector<vsag::lite::Neighbor>> before;
    uint64_t hits = 0;
    for (uint64_t q = 0; q < queries.size(); ++q) {
        start = std::chrono::steady_clock::now();
        auto result = (*graph)->Search(queries[q].data(), 128, 10);
        latency.push_back(
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count());
        REQUIRE(result);
        REQUIRE(result->size() == 10);
        const std::unordered_set<int64_t> expected(truth[q].begin(), truth[q].end());
        for (const auto& neighbor : *result) {
            hits += expected.count(neighbor.id);
        }
        before.push_back(std::move(*result));
    }
    const char* snapshot_path = std::getenv("VSAG_GRAPH_SNAPSHOT");
    REQUIRE(snapshot_path != nullptr);
    start = std::chrono::steady_clock::now();
    std::ofstream output(snapshot_path, std::ios::binary);
    REQUIRE(output);
    REQUIRE((*graph)->Save(output));
    output.close();
    REQUIRE(output);
    const auto save_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    start = std::chrono::steady_clock::now();
    std::ifstream input(snapshot_path, std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    REQUIRE(loaded);
    const auto load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    REQUIRE((*loaded)->ActiveBackend() == vsag::lite::BackendKind::GRAPH);
    for (uint64_t q = 0; q < queries.size(); ++q) {
        auto after = (*loaded)->Search(queries[q].data(), 128, 10);
        REQUIRE(after);
        REQUIRE(after->size() == before[q].size());
        for (uint64_t i = 0; i < after->size(); ++i) {
            REQUIRE((*after)[i].id == before[q][i].id);
            REQUIRE((*after)[i].distance == before[q][i].distance);
        }
    }
    std::sort(latency.begin(), latency.end());
    std::cout << "base_count,query_count,degree,ef,recall_at_10,build_ms,search_p50_us,"
                 "search_p99_us,save_ms,load_ms,snapshot_bytes\n";
    std::cout << base.size() << ',' << queries.size() << ",16,128,"
              << static_cast<double>(hits) / static_cast<double>(queries.size() * 10) << ','
              << build_ms << ',' << latency[(latency.size() - 1) / 2] << ','
              << latency[(latency.size() * 99 - 1) / 100] << ',' << save_ms << ',' << load_ms << ','
              << std::filesystem::file_size(snapshot_path) << '\n';
    REQUIRE(hits >= queries.size() * 5);
}

TEST_CASE("Lite graph SIFT fresh-process load probe", "[.][lite-sift-load]") {
    const char* directory = std::getenv("VSAG_SIFT_DIR");
    const char* snapshot_path = std::getenv("VSAG_GRAPH_SNAPSHOT");
    REQUIRE(directory != nullptr);
    REQUIRE(snapshot_path != nullptr);
    const std::string root(directory);
    const auto queries = read_sift<float>(root + "/queries.fvecs", 128);
    const auto truth = read_sift<int32_t>(root + "/groundtruth.ivecs", 10);
    REQUIRE(queries.size() == truth.size());
    auto start = std::chrono::steady_clock::now();
    std::ifstream input(snapshot_path, std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    REQUIRE(loaded);
    REQUIRE((*loaded)->ActiveBackend() == vsag::lite::BackendKind::GRAPH);
    const auto load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    uint64_t hits = 0;
    std::vector<double> latency;
    for (uint64_t q = 0; q < queries.size(); ++q) {
        start = std::chrono::steady_clock::now();
        auto result = (*loaded)->Search(queries[q].data(), 128, 10);
        latency.push_back(
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                .count());
        REQUIRE(result);
        REQUIRE(result->size() == 10);
        const std::unordered_set<int64_t> expected(truth[q].begin(), truth[q].end());
        for (const auto& neighbor : *result) {
            hits += expected.count(neighbor.id);
        }
    }
    std::sort(latency.begin(), latency.end());
    rusage usage{};
    REQUIRE(getrusage(RUSAGE_SELF, &usage) == 0);
    std::cout << "query_count,recall_at_10,load_ms,search_p50_us,search_p99_us,"
                 "process_peak_rss_kib,snapshot_bytes\n";
    std::cout << queries.size() << ','
              << static_cast<double>(hits) / static_cast<double>(queries.size() * 10) << ','
              << load_ms << ',' << latency[(latency.size() - 1) / 2] << ','
              << latency[(latency.size() * 99 - 1) / 100] << ',' << usage.ru_maxrss << ','
              << std::filesystem::file_size(snapshot_path) << '\n';
    REQUIRE(hits >= queries.size() * 5);
}
