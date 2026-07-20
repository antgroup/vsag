
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
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "memmove_heap.h"
#include "standard_heap.h"
#include "unittest.h"
using namespace vsag;

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
