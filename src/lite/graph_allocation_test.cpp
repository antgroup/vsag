// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/lite/index.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <sstream>

#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
#include "lite/rabitq_snapshot.h"
#endif
static long remaining = -1;
void*
operator new(std::size_t n) {
    if (remaining >= 0 && remaining-- == 0) {
        remaining = -1;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n != 0U ? n : 1)) {
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
    for (int policy = 0; policy < 6; ++policy) {
        codec::MutableGraphState initial(
            model, codes, graph, ids, operation >= 2 ? 6 : 4, 16, false, reverse);
        initial.ConfigureIncomingProtection(policy == 0
                                                ? codec::IncomingProtection::NONE
                                                : (policy == 1 ? codec::IncomingProtection::RECOUNT
                                                               : codec::IncomingProtection::CACHED),
                                            true,
                                            true,
                                            true,
                                            policy >= 4);
        if ((policy == 3 or policy == 5) and
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
#endif

int
main() {
    int problems = 0;
#if defined(VSAG_LITE_HAS_RABITQ_BACKEND)
    try {
        problems += check_reverse_mutation_allocation(0);
        problems += check_reverse_mutation_allocation(1);
        problems += check_reverse_mutation_allocation(2);
        problems += check_reverse_mutation_allocation(3);
        problems += check_reverse_mutation_allocation(3, 1);
        problems += check_reverse_mutation_allocation(0, 16, false);
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
        3;
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
                const auto storage = mode == 2 ? vsag::lite::VectorStorage::RABITQ8
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
                    if (mode == 2 and op != 3 and retried) {
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
