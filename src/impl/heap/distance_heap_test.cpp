
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

#include "distance_heap.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "memmove_heap.h"
#include "standard_heap.h"
#include "unittest.h"
using namespace vsag;

namespace {
class RecordingAllocator : public DefaultAllocator {
public:
    void*
    Allocate(uint64_t size) override {
        allocation_sizes.push_back(size);
        return DefaultAllocator::Allocate(size);
    }

    std::vector<uint64_t> allocation_sizes;
};
}  // namespace

class TestDistanceHeap {
public:
    TestDistanceHeap() {
        uint64_t data_count = 1000;
        auto dists = fixtures::GenerateVectors<float>(data_count, 1, 473, false);
        for (int i = 0; i < data_count; ++i) {
            data.emplace_back(dists[i], i);
        }
        sorted_data_greater = data;

        std::sort(sorted_data_greater.begin(), sorted_data_greater.end());
        sorted_data_less.resize(data_count);
        std::reverse_copy(
            sorted_data_greater.begin(), sorted_data_greater.end(), sorted_data_less.begin());
    }

    void
    RunBasicTest(DistanceHeap& heap, bool use_max) {
        for (auto& it : data) {
            heap.Push(it);
        }
        auto gt = &sorted_data_less;
        if (use_max) {
            gt = &sorted_data_greater;
        }

        auto size = heap.Size();
        std::vector<DistanceHeap::DistanceRecord> temp;
        std::vector<DistanceHeap::DistanceRecord> temp2(size);

        const auto* data = heap.GetData();
        memcpy(temp2.data(), data, size * sizeof(DistanceHeap::DistanceRecord));
        std::sort(temp2.begin(), temp2.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        if (use_max) {
            std::reverse(temp2.begin(), temp2.end());
        }
        while (not heap.Empty()) {
            temp.emplace_back(heap.Top());
            heap.Pop();
        }
        REQUIRE(temp.size() == size);
        for (int i = 0; i < size; ++i) {
            REQUIRE(gt->at(size - i - 1) == temp[i]);
            REQUIRE(gt->at(size - i - 1) == temp2[i]);
        }
    }

private:
    std::vector<DistanceHeap::DistanceRecord> data;

    std::vector<DistanceHeap::DistanceRecord> sorted_data_greater;

    std::vector<DistanceHeap::DistanceRecord> sorted_data_less;
};

TEST_CASE_METHOD(TestDistanceHeap, "standard_heap test", "[ut][distance_heap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        const int64_t max_size = 10;
        StandardHeap<true, true> heap1(allocator.get(), max_size);
        RunBasicTest(heap1, true);
        StandardHeap<true, false> heap2(allocator.get(), max_size);
        RunBasicTest(heap2, true);
        StandardHeap<false, true> heap3(allocator.get(), max_size);
        RunBasicTest(heap3, false);
        StandardHeap<false, false> heap4(allocator.get(), max_size);
        RunBasicTest(heap4, false);
    }
}

TEST_CASE("standard_heap bounds initial reserve", "[ut][distance_heap]") {
    // Since callers pass the expected frontier size (e.g. ef) as a reservation
    // hint, the constructor clamps it into [64, 512]: small hints are raised so
    // a single allocation covers typical searches, huge ones are capped.
    constexpr uint64_t min_capacity = 64;
    constexpr uint64_t max_capacity = 512;
    const auto expected_bytes = [](uint64_t capacity) {
        return capacity * sizeof(DistanceHeap::DistanceRecord);
    };
    const auto check_initial_reserve = [&](int64_t max_size, uint64_t expected_capacity) {
        RecordingAllocator allocator;
        {
            StandardHeap<true, true> heap(&allocator, max_size);
            REQUIRE(heap.Empty());
        }
        REQUIRE(allocator.allocation_sizes.size() == 1);
        REQUIRE(allocator.allocation_sizes.front() == expected_bytes(expected_capacity));
    };

    check_initial_reserve(3, min_capacity);
    check_initial_reserve(std::numeric_limits<int64_t>::max(), max_capacity);
    check_initial_reserve(200, 200);
    check_initial_reserve(-1, min_capacity);
}

TEST_CASE("standard_heap small fixed and dynamic behavior", "[ut][distance_heap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    StandardHeap<true, true> fixed_heap(allocator.get(), 3);
    fixed_heap.Push(3.0F, 3);
    fixed_heap.Push(1.0F, 1);
    fixed_heap.Push(2.0F, 2);
    fixed_heap.Push(4.0F, 4);
    REQUIRE(fixed_heap.Size() == 3);
    REQUIRE(fixed_heap.Top().first == 3.0F);

    StandardHeap<true, false> dynamic_heap(allocator.get(), -1);
    dynamic_heap.Push(3.0F, 3);
    dynamic_heap.Push(1.0F, 1);
    dynamic_heap.Push(2.0F, 2);
    REQUIRE(dynamic_heap.Size() == 3);
    REQUIRE(dynamic_heap.Top().first == 3.0F);
    dynamic_heap.Pop();
    REQUIRE(dynamic_heap.Top().first == 2.0F);
    dynamic_heap.Pop();
    REQUIRE(dynamic_heap.Top().first == 1.0F);
}

TEST_CASE("standard_heap randomized differential vs std heap", "[ut][distance_heap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::mt19937_64 rng(20260822);

    // Distances are made strictly increasing by appending a tiny per-index
    // epsilon, so every key is unique and the extraction order is fully
    // determined — the hand-rolled sift must match std::push_heap/pop_heap
    // element for element.
    for (int iter = 0; iter < 20; ++iter) {
        const auto count = 1 + static_cast<int>(rng() % 300);
        const int64_t cap = 1 + static_cast<int64_t>(rng() % 256);
        std::vector<float> base_dists =
            fixtures::GenerateVectors<float>(count, 1, static_cast<int>(rng()), false);
        for (int i = 0; i < count; ++i) {
            base_dists[i] += static_cast<float>(i) * 1e-3F;
        }

        for (auto max_heap : {true, false}) {
            for (auto fixed : {true, false}) {
                std::vector<DistanceHeap::DistanceRecord> ref;
                auto push_ref = [&](float dist, InnerIdType id) {
                    ref.emplace_back(dist, id);
                    if (max_heap) {
                        std::push_heap(ref.begin(), ref.end(), DistanceHeap::CompareMax());
                    } else {
                        std::push_heap(ref.begin(), ref.end(), DistanceHeap::CompareMin());
                    }
                    if (fixed && static_cast<int64_t>(ref.size()) > cap) {
                        if (max_heap) {
                            std::pop_heap(ref.begin(), ref.end(), DistanceHeap::CompareMax());
                        } else {
                            std::pop_heap(ref.begin(), ref.end(), DistanceHeap::CompareMin());
                        }
                        ref.pop_back();
                    }
                };

                std::unique_ptr<DistanceHeap> heap;
                if (max_heap && fixed) {
                    heap = std::make_unique<StandardHeap<true, true>>(allocator.get(), cap);
                } else if (max_heap) {
                    heap = std::make_unique<StandardHeap<true, false>>(allocator.get(), -1);
                } else if (fixed) {
                    heap = std::make_unique<StandardHeap<false, true>>(allocator.get(), cap);
                } else {
                    heap = std::make_unique<StandardHeap<false, false>>(allocator.get(), -1);
                }

                std::vector<DistanceHeap::DistanceRecord> popped_std;
                std::vector<DistanceHeap::DistanceRecord> popped_impl;
                int pushed = 0;
                int pop_pressure = 0;
                while (pushed < count || !ref.empty()) {
                    const bool do_push = (pushed < count) &&
                                         ((rng() % 3 != 0) || ref.empty());
                    if (do_push) {
                        const float dist = base_dists[pushed];
                        push_ref(dist, static_cast<InnerIdType>(pushed));
                        heap->Push(dist, static_cast<InnerIdType>(pushed));
                        ++pushed;
                    } else {
                        REQUIRE(!heap->Empty());
                        REQUIRE(heap->Top().first == ref.front().first);
                        popped_std.push_back(ref.front());
                        popped_impl.push_back(heap->Top());
                        if (max_heap) {
                            std::pop_heap(
                                ref.begin(), ref.end(), DistanceHeap::CompareMax());
                        } else {
                            std::pop_heap(
                                ref.begin(), ref.end(), DistanceHeap::CompareMin());
                        }
                        ref.pop_back();
                        heap->Pop();
                    }
                    REQUIRE(heap->Size() == ref.size());
                    if (!ref.empty()) {
                        REQUIRE(heap->Top().first == ref.front().first);
                    }
                }
                // Full drain order must match exactly (keys are unique).
                REQUIRE(popped_std.size() == popped_impl.size());
                for (size_t i = 0; i < popped_std.size(); ++i) {
                    REQUIRE(popped_std[i].first == popped_impl[i].first);
                    REQUIRE(popped_std[i].second == popped_impl[i].second);
                }
            }
        }
    }
}

TEST_CASE_METHOD(TestDistanceHeap, "memmove_heap test", "[ut][distance_heap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        const int64_t max_size = 10;
        MemmoveHeap<true, true> heap1(allocator.get(), max_size);
        RunBasicTest(heap1, true);
        MemmoveHeap<true, false> heap2(allocator.get(), max_size);
        RunBasicTest(heap2, true);
        MemmoveHeap<false, true> heap3(allocator.get(), max_size);
        RunBasicTest(heap3, false);
        MemmoveHeap<false, false> heap4(allocator.get(), max_size);
        RunBasicTest(heap4, false);
    }
}

// Regression for the non-fixed MemmoveHeap buffer overflow hit by IVF::search
// when topk < 10. The non-fixed heap's Pop() only decrements cur_size_ without
// shrinking ordered_buffer_, so a stale tail accumulates. IVF interleaves
// Push with a bounded Pop (the loop replayed below); the non-fixed Push then
// searched the whole buffer (including the stale tail) and could compute a
// negative memmove size, overrunning the heap. This exercises that exact
// interleaving and validates the kept records against a brute-force top-k.
//
// RunBasicTest does not catch it because it pushes all elements then pops all,
// never interleaving Push/Pop; IVF's own tests use top_k == 10, which selects
// StandardHeap rather than MemmoveHeap.
TEST_CASE_METHOD(TestDistanceHeap,
                 "memmove_heap non-fixed Pop-then-Push (topk<10)",
                 "[ut][distance_heap]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    // seed 0 produces a distance sequence that triggers the overflow within a
    // few dozen iterations for several of these topk values.
    auto dists = fixtures::GenerateVectors<float>(1000, 1, 0, false);

    for (int64_t topk = 1; topk < 10; ++topk) {
        MemmoveHeap<true, false> heap(allocator.get(), topk);
        auto cur_heap_top = std::numeric_limits<float>::max();

        // Mirror the KNN scan loop in IVF::search (src/algorithm/ivf/ivf.cpp).
        for (uint64_t i = 0; i < 1000; ++i) {
            float d = dists[i];
            if (heap.Size() < static_cast<uint64_t>(topk) or d < cur_heap_top) {
                heap.Push(d, static_cast<InnerIdType>(i));
            }
            if (heap.Size() > static_cast<uint64_t>(topk)) {
                heap.Pop();
            }
            if (not heap.Empty() and heap.Size() == static_cast<uint64_t>(topk)) {
                cur_heap_top = heap.Top().first;
            }
        }

        REQUIRE(heap.Size() == static_cast<uint64_t>(topk));

        // The retained records must be exactly the topk smallest distances.
        std::vector<float> got;
        const auto* heap_data = heap.GetData();
        for (uint64_t i = 0; i < heap.Size(); ++i) {
            got.emplace_back(heap_data[i].first);
        }
        std::sort(got.begin(), got.end());

        std::vector<float> expected(dists.begin(), dists.begin() + 1000);
        std::sort(expected.begin(), expected.end());
        expected.resize(topk);

        REQUIRE(got == expected);
    }
}
