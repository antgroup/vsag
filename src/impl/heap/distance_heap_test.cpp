
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

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "memmove_heap.h"
#include "standard_heap.h"

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

        std::vector<DistanceHeap::DistanceRecord> temp;
        auto size = heap.Size();
        while (not heap.Empty()) {
            temp.emplace_back(heap.Top());
            heap.Pop();
        }
        REQUIRE(temp.size() == size);
        for (int i = 0; i < size; ++i) {
            REQUIRE(gt->at(size - i - 1) == temp[i]);
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
