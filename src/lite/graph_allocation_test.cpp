// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/lite/index.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <sstream>

#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
#include "lite/rabitq_snapshot.h"
#endif
static long remaining = -1;
static bool measure_allocations = false;
static uint64_t allocated_bytes = 0;
static uint64_t largest_allocation = 0;
void*
operator new(std::size_t n) {
    if (remaining >= 0 && remaining-- == 0) {
        remaining = -1;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n != 0U ? n : 1)) {
        if (measure_allocations) {
            allocated_bytes += n;
            largest_allocation = std::max(largest_allocation, static_cast<uint64_t>(n));
        }
        return p;
    }
    throw std::bad_alloc();
}
void
operator delete(void* p) noexcept {
    std::free(p);
}
void
operator delete(void* p, std::size_t /*unused*/) noexcept {
    std::free(p);
}
void*
operator new[](std::size_t n) {
    return ::operator new(n);
}
void
operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void
operator delete[](void* p, std::size_t /*unused*/) noexcept {
    ::operator delete(p);
}
std::string
save(vsag::lite::Index& i) {
    std::ostringstream o;
    auto r = i.Save(o);
    if (!r) {
        return "SAVE_ERROR";
    }
    return o.str();
}

#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
int
check_reverse_mutation_allocation(int operation, uint64_t count = 16, bool reverse = true) {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t dim = 17;
    std::vector<float> base(count * dim);
    std::vector<int64_t> ids(count);
    codec::GraphTopology graph;
    graph.offsets.push_back(0);
    for (uint64_t row = 0; row < count; ++row) {
        ids[row] = static_cast<int64_t>(row);
        for (uint64_t d = 0; d < dim; ++d) {
            base[row * dim + d] = std::sin(static_cast<float>(row * 7 + d) * 0.17F);
        }
        for (uint64_t step : {uint64_t{1}, uint64_t{3}, uint64_t{7}}) {
            const auto neighbor = (row + step) % count;
            if (neighbor != row and std::find(graph.neighbors.begin() +
                                                  static_cast<std::ptrdiff_t>(graph.offsets.back()),
                                              graph.neighbors.end(),
                                              neighbor) == graph.neighbors.end()) {
                graph.neighbors.push_back(neighbor);
            }
        }
        if (operation >= 2 and count > 1) {
            // Cover mutual deleted/moved edges and a source referencing both slots.
            const auto append_unique = [&](uint64_t target) {
                if (target != row and
                    std::find(
                        graph.neighbors.begin() + static_cast<std::ptrdiff_t>(graph.offsets.back()),
                        graph.neighbors.end(),
                        target) == graph.neighbors.end()) {
                    graph.neighbors.push_back(target);
                }
            };
            if (row == 0 or row == count / 2) {
                append_unique(count - 1);
            }
            if (row == 0 or row == count - 1) {
                append_unique(count / 2);
            }
        }
        graph.offsets.push_back(graph.neighbors.size());
    }
    const auto model = codec::train(base, count, dim, 47);
    codec::EncodedRecords codes(dim);
    for (uint64_t row = 0; row < count; ++row) {
        codes.Append(codec::encode(model, base.data() + row * dim));
    }
    std::vector<float> replacement(dim);
    for (uint64_t d = 0; d < dim; ++d) {
        replacement[d] = std::cos(static_cast<float>(d) * 0.31F);
    }
    int problems = 0;
    for (int policy = 0; policy < (operation >= 2 ? 8 : 6); ++policy) {
        codec::MutableGraphState initial(
            model, codes, graph, ids, operation >= 2 ? 6 : 4, 16, false, reverse);
        initial.ConfigureIncomingProtection(policy == 0
                                                ? codec::IncomingProtection::NONE
                                                : (policy == 1 ? codec::IncomingProtection::RECOUNT
                                                               : codec::IncomingProtection::CACHED),
                                            policy < 6,
                                            policy < 6,
                                            policy < 6,
                                            policy >= 4);
        if ((policy == 3 or policy == 5 or policy == 7) and
            not initial.UpdateTransactional(0,
                                            codec::prepare_encoding(model, replacement.data()))) {
            return 1;
        }
        std::ostringstream before_stream;
        codec::save_mutable_snapshot(before_stream, initial);
        const auto before = before_stream.str();
        const auto rebuilds = initial.IncomingCountRebuilds();
        const auto mutate = [&](codec::MutableGraphState& state) {
            if (operation >= 2) {
                return state.RemoveTransactional(
                    static_cast<int64_t>(operation == 2 ? count / 2 : count - 1));
            }
            return operation == 1 ? state.AddTransactional(100, replacement.data())
                                  : state.UpdateTransactional(
                                        3, codec::prepare_encoding(model, replacement.data()));
        };
        auto golden = initial;
        const bool expected =
            operation >= 2
                ? golden.Remove(static_cast<int64_t>(operation == 2 ? count / 2 : count - 1))
                : (operation == 1 ? golden.Add(100, replacement.data())
                                  : golden.UpdatePrepared(
                                        3, codec::prepare_encoding(model, replacement.data())));
        if (not expected) {
            return 1;
        }
        std::ostringstream golden_stream;
        codec::save_mutable_snapshot(golden_stream, golden);
        int failures = 0;
        int successes = 0;
        for (int limit = 0; limit < 500; ++limit) {
            auto state = initial;
            bool failed = false;
            remaining = limit;
            try {
                if (not mutate(state)) {
                    ++problems;
                }
            } catch (const std::bad_alloc&) {
                failed = true;
            } catch (...) {
                remaining = -1;
                ++problems;
                failed = true;
            }
            remaining = -1;
            state.Validate();
            std::ostringstream after_stream;
            codec::save_mutable_snapshot(after_stream, state);
            if (failed) {
                ++failures;
                // Non-reverse rollback invalidates the cache; attempted rebuilds are diagnostic.
                if (after_stream.str() != before or
                    (reverse and state.IncomingCountRebuilds() != rebuilds)) {
                    ++problems;
                }
                if (not mutate(state)) {
                    ++problems;
                }
                state.Validate();
                std::ostringstream retry_stream;
                codec::save_mutable_snapshot(retry_stream, state);
                if (retry_stream.str() != golden_stream.str() or
                    (reverse and state.IncomingCountRebuilds() != golden.IncomingCountRebuilds())) {
                    ++problems;
                }
            } else {
                ++successes;
                if (after_stream.str() != golden_stream.str() or
                    (reverse and state.IncomingCountRebuilds() != golden.IncomingCountRebuilds())) {
                    ++problems;
                }
            }
            if (operation >= 2) {
                auto continued_golden = golden;
                if (not state.AddTransactional(100, replacement.data()) or
                    not continued_golden.Add(100, replacement.data()) or
                    not state.UpdateTransactional(100,
                                                  codec::prepare_encoding(model, base.data())) or
                    not continued_golden.UpdatePrepared(
                        100, codec::prepare_encoding(model, base.data()))) {
                    ++problems;
                }
                state.Validate();
                std::ostringstream continued_stream;
                std::ostringstream continued_golden_stream;
                codec::save_mutable_snapshot(continued_stream, state);
                codec::save_mutable_snapshot(continued_golden_stream, continued_golden);
                if (continued_stream.str() != continued_golden_stream.str() or
                    (reverse and
                     state.IncomingCountRebuilds() != continued_golden.IncomingCountRebuilds())) {
                    ++problems;
                }
            }
            const auto memory = state.GetMemoryUsage();
            const auto incoming = state.GetIncomingMemoryUsage();
            if (memory.incoming_adjacency_logical_bytes != incoming.logical_bytes or
                memory.incoming_adjacency_capacity_bytes != incoming.capacity_bytes) {
                ++problems;
            }
        }
        if (failures == 0 or successes == 0) {
            ++problems;
        }
        const char* label = operation == 0 ? "UPDATE" : (operation == 1 ? "ADD" : "REMOVE");
        std::cout << "REVERSE_" << label << " operation=" << operation << " count=" << count
                  << " reverse=" << reverse << " policy=" << policy << " failures=" << failures
                  << " successes=" << successes << " cumulative_problems=" << problems << "\n";
    }
    return problems;
}

int
check_cold_remove_allocations() {
    namespace codec = vsag::lite::detail::rabitq;
    constexpr uint64_t count = 1024;
    constexpr uint64_t dim = 17;
    std::vector<float> base(count * dim);
    std::vector<int64_t> ids(count);
    codec::GraphTopology graph;
    graph.offsets.push_back(0);
    for (uint64_t row = 0; row < count; ++row) {
        ids[row] = static_cast<int64_t>(row);
        for (uint64_t d = 0; d < dim; ++d) {
            base[row * dim + d] = std::sin(static_cast<float>(row * 7 + d) * 0.17F);
        }
        for (const uint64_t step : {uint64_t{1}, uint64_t{3}, uint64_t{7}}) {
            graph.neighbors.push_back((row + step) % count);
        }
        graph.offsets.push_back(graph.neighbors.size());
    }
    const auto model = codec::train(base, count, dim, 47);
    codec::EncodedRecords codes(dim);
    for (uint64_t row = 0; row < count; ++row) {
        codes.Append(codec::encode(model, base.data() + row * dim));
    }
    codec::MutableGraphState initial(model, codes, graph, ids, 6, 16, false, true);
    int problems = 0;
    for (const bool loaded : {false, true}) {
        auto state = initial;
        if (loaded) {
            std::stringstream snapshot;
            codec::save_mutable_snapshot(snapshot, state);
            state = codec::load_mutable_snapshot(snapshot, true);
        }
        state.ConfigureIncomingProtection(
            codec::IncomingProtection::CACHED, false, false, false, true);
        auto golden = state;
        for (uint64_t step = 0; step < 8; ++step) {
            const uint64_t slot =
                step % 3 == 0 ? state.Size() / 2 : (step % 3 == 1 ? state.Size() - 1 : 0);
            const int64_t id = state.GetIds()[slot];
            allocated_bytes = 0;
            largest_allocation = 0;
            measure_allocations = true;
            const bool removed = state.RemoveTransactional(id);
            measure_allocations = false;
            const auto bytes = allocated_bytes;
            const auto largest = largest_allocation;
            // A full adjacency-vector copy alone requests Size()*sizeof(vector).
            // This bounded-degree fixture needs only local journal/repair rows.
            const uint64_t full_copy_request = (state.Size() + 1) * sizeof(std::vector<uint64_t>);
            const bool expected_removed = golden.Remove(id);
            if (not removed or largest >= full_copy_request or not expected_removed) {
                ++problems;
            }
            state.Validate();
            std::ostringstream actual_snapshot;
            std::ostringstream golden_snapshot;
            codec::save_mutable_snapshot(actual_snapshot, state);
            codec::save_mutable_snapshot(golden_snapshot, golden);
            if (actual_snapshot.str() != golden_snapshot.str() or
                state.IncomingCountRebuilds() != 0) {
                ++problems;
            }
            const auto actual = state.Search(base.data(), 10);
            const auto expected = golden.Search(base.data(), 10);
            if (actual.neighbors.size() != expected.neighbors.size()) {
                ++problems;
            } else {
                for (uint64_t i = 0; i < actual.neighbors.size(); ++i) {
                    if (actual.neighbors[i].id != expected.neighbors[i].id or
                        actual.neighbors[i].distance != expected.neighbors[i].distance) {
                        ++problems;
                    }
                }
            }
            std::cout << "COLD_REMOVE loaded=" << loaded << " step=" << step << " bytes=" << bytes
                      << " largest=" << largest << " full_copy_request=" << full_copy_request
                      << " cumulative_problems=" << problems << "\n";
        }
    }
    return problems;
}
#endif

int
main() {
    int problems = 0;
    // A non-allocating stream isolates Save's own staging allocation.
    struct Sink : std::streambuf {
        std::streamsize
        xsputn(const char* /*unused*/, std::streamsize count) override {
            return count;
        }
        int_type
        overflow(int_type value) override {
            return traits_type::not_eof(value);
        }
    };
    auto created = vsag::lite::Index::Create(2);
    const float vector[]{1, 2};
    if (not created or not(*created)->Add(1, vector, 2) or
        not(*created)->BuildGraph(vsag::lite::VectorStorage::FP16)) {
        return 1;
    }
    const auto before_save = save(**created);
    Sink sink;
    std::ostream output(&sink);
    remaining = 0;
    bool save_failed = false;
    try {
        const auto saved = (*created)->Save(output);
        remaining = -1;
        save_failed = not saved and saved.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY;
    } catch (...) {
        remaining = -1;
        ++problems;
    }
    if (not save_failed or save(**created) != before_save) {
        ++problems;
    }
    std::cout << "SAVE_ALLOCATION problems=" << problems << "\n";
#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
    try {
        problems += check_reverse_mutation_allocation(0);
        problems += check_reverse_mutation_allocation(1);
        problems += check_reverse_mutation_allocation(2);
        problems += check_reverse_mutation_allocation(3);
        problems += check_reverse_mutation_allocation(3, 1);
        problems += check_reverse_mutation_allocation(0, 16, false);
        problems += check_cold_remove_allocations();
        problems += check_reverse_mutation_allocation(1, 16, false);
        problems += check_reverse_mutation_allocation(2, 16, false);
    } catch (const std::exception& error) {
        remaining = -1;
        std::cerr << "Reverse mutation fixture failed: " << error.what() << "\n";
        return 1;
    } catch (...) {
        remaining = -1;
        return 1;
    }
#endif
    constexpr int modes =
#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
        4;
#else
        2;
#endif
    for (int mode = 0; mode < modes; ++mode) {
        for (int op = 0; op < 4; ++op) {
            int failures = 0;
            int changed = 0;
            int invalid = 0;
            int escaped = 0;
            for (int limit = 0; limit < 500; ++limit) {
                auto made = vsag::lite::Index::Create(2);
                auto index = std::move(*made);
                for (int id = 0; id < 8; ++id) {
                    float v[2] = {float(id), float(id % 3)};
                    if (!index->Add(id, v, 2)) {
                        return 2;
                    }
                }
                const auto storage = mode >= 2 ? vsag::lite::VectorStorage::RABITQ8
                                               : (mode != 0 ? vsag::lite::VectorStorage::FP16
                                                            : vsag::lite::VectorStorage::FP32);
                if (op != 3 and not index->BuildGraph(storage, 3, 16)) {
                    return 3;
                }
                // Warm experimental incoming counts before mutation fault injection.
                // The default adapter keeps the same unprotected public behavior.
                if (mode == 2 and op != 3) {
                    const float warm[2] = {-7, 2};
                    if (not index->Update(0, warm, 2)) {
                        return 4;
                    }
                }
                if (mode == 3 and op != 3) {
                    std::istringstream snapshot(save(*index));
                    auto loaded = vsag::lite::Index::Load(snapshot);
                    if (not loaded) {
                        return 5;
                    }
                    index = std::move(*loaded);
                }
                const auto before = save(*index);
                bool failed = false;
                remaining = limit;
                try {
                    float v[2] = {23, 5};
                    if (op == 0) {
                        failed = !index->Add(20, v, 2);
                    } else if (op == 1) {
                        failed = !index->Update(3, v, 2);
                    } else if (op == 2) {
                        failed = !index->Remove(3);
                    } else {
                        failed = !index->BuildGraph(storage, 3, 16);
                    }
                } catch (const std::exception&) {
                    failed = true;
                    ++escaped;
                }
                remaining = -1;
                if (failed) {
                    ++failures;
                    auto after = save(*index);
                    if (after != before) {
                        ++changed;
                        if (changed == 1) {
                            std::cout << "FIRST_CHANGED mode=" << mode << " op=" << op
                                      << " allocation=" << limit << " size=" << index->Size()
                                      << "\n";
                        }
                    }
                    std::istringstream in(after);
                    if (!vsag::lite::Index::Load(in)) {
                        ++invalid;
                    }
                    const float retry_vector[2] = {23, 5};
                    const bool retried =
                        op == 0   ? static_cast<bool>(index->Add(20, retry_vector, 2))
                        : op == 1 ? static_cast<bool>(index->Update(3, retry_vector, 2))
                        : op == 2 ? index->Remove(3)
                                  : static_cast<bool>(index->BuildGraph(storage, 3, 16));
                    if (not retried) {
                        ++invalid;
                    }
                    if (mode >= 2 and op != 3 and retried) {
                        const float continued[2] = {-6, 3};
                        if (not index->Update(1, continued, 2)) {
                            ++invalid;
                        }
                        std::istringstream retried_snapshot(save(*index));
                        if (not vsag::lite::Index::Load(retried_snapshot)) {
                            ++invalid;
                        }
                    }
                }
            }
            problems += changed + invalid + escaped;
            std::cout << "SUMMARY mode=" << mode << " op=" << op << " failures=" << failures
                      << " changed_after_failure=" << changed << " invalid_snapshot=" << invalid
                      << " escaped=" << escaped << "\n";
        }
    }
    return problems == 0 ? 0 : 1;
}
