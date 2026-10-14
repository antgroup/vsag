
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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "memmove_heap.h"
#include "standard_heap.h"
#include "unittest.h"
using namespace vsag;

namespace {

bool
IsNaNBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) == 0x7F800000U and (bits & 0x007FFFFFU) != 0U;
}

template <bool max_heap>
void
RequireFixedAuxiliaryHeapMatchesOracle(Allocator* allocator) {
    constexpr uint64_t kCount = 2048;
    constexpr uint64_t kCapacity = 37;
    auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<max_heap, true>(
        allocator, static_cast<int64_t>(kCapacity));

    std::vector<InnerIdType> insertion_order(kCount);
    std::iota(insertion_order.begin(), insertion_order.end(), InnerIdType{0});
    std::mt19937 random(0xA17E5EEDU);
    std::shuffle(insertion_order.begin(), insertion_order.end(), random);

    std::vector<DistanceHeap::AuxiliaryDistanceRecord> oracle;
    oracle.reserve(kCount);
    for (const auto id : insertion_order) {
        const float distance = static_cast<float>(id) + 0.25F;
        DistanceHeap::AuxiliaryDistanceRecord record{{distance, id},
                                                     1000.0F + static_cast<float>(id),
                                                     static_cast<BucketIdType>(id % 11U),
                                                     id + 100U,
                                                     0,
                                                     10000U + id};
        oracle.emplace_back(record);
        heap->PushWithAuxiliary(distance,
                                id,
                                record.auxiliary,
                                record.source_bucket_id,
                                record.source_offset_id,
                                record.source_version);
    }

    auto by_distance = [](const auto& lhs, const auto& rhs) {
        return lhs.record.first < rhs.record.first;
    };
    std::sort(oracle.begin(), oracle.end(), by_distance);
    if constexpr (max_heap) {
        oracle.resize(kCapacity);
    } else {
        oracle.erase(oracle.begin(), oracle.end() - static_cast<int64_t>(kCapacity));
    }

    REQUIRE(heap->Size() == kCapacity);
    const auto* heap_data = heap->GetDataWithAuxiliary();
    std::vector<DistanceHeap::AuxiliaryDistanceRecord> actual(heap_data, heap_data + heap->Size());
    std::sort(actual.begin(), actual.end(), by_distance);
    REQUIRE(actual.size() == oracle.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i].record == oracle[i].record);
        REQUIRE(actual[i].auxiliary == oracle[i].auxiliary);
        REQUIRE(actual[i].source_bucket_id == oracle[i].source_bucket_id);
        REQUIRE(actual[i].source_offset_id == oracle[i].source_offset_id);
        REQUIRE(actual[i].source_version == oracle[i].source_version);
    }
}

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

TEST_CASE("auxiliary heap keeps payload aligned with distance records",
          "[ut][distance_heap][auxiliary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    auto require_aligned = [](const DistHeapPtr& heap) {
        REQUIRE(heap->StoresAuxiliary());
        const auto* auxiliary_data = heap->GetDataWithAuxiliary();
        const auto* distance_data = heap->GetData();
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            REQUIRE(auxiliary_data[i].record == distance_data[i]);
            if (IsNaNBits(auxiliary_data[i].auxiliary)) {
                continue;
            }
            const auto id = auxiliary_data[i].record.second;
            REQUIRE(auxiliary_data[i].auxiliary == 1000.0F + static_cast<float>(id));
            REQUIRE(auxiliary_data[i].source_bucket_id == static_cast<BucketIdType>(id % 3U));
            REQUIRE(auxiliary_data[i].source_offset_id == id + 10U);
            REQUIRE(auxiliary_data[i].source_version == 900U + id);
        }
    };

    SECTION("non-fixed push pop and equal distances retain matching payloads") {
        auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(allocator.get(), 5);
        auto cur_heap_top = std::numeric_limits<float>::max();
        for (InnerIdType id = 0; id < 100; ++id) {
            const float distance = static_cast<float>((id * 37U) % 19U);
            if (heap->Size() < 5 or distance < cur_heap_top) {
                heap->PushWithAuxiliary(distance,
                                        id,
                                        1000.0F + static_cast<float>(id),
                                        static_cast<BucketIdType>(id % 3U),
                                        id + 10U,
                                        900U + id);
            }
            if (heap->Size() > 5) {
                heap->Pop();
            }
            if (heap->Size() == 5) {
                cur_heap_top = heap->Top().first;
            }
            require_aligned(heap);
        }

        REQUIRE(heap->Size() == 5);
        while (not heap->Empty()) {
            const auto top = heap->Top();
            const auto* auxiliary_data = heap->GetDataWithAuxiliary();
            REQUIRE(auxiliary_data[0].record == top);
            REQUIRE(auxiliary_data[0].auxiliary == 1000.0F + static_cast<float>(top.second));
            heap->Pop();
        }
    }

    SECTION("fixed max heap rejects and replaces records with their payload") {
        auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, true>(allocator.get(), 3);
        for (const auto& [distance, id] : std::vector<DistanceHeap::DistanceRecord>{
                 {9.0F, 0}, {1.0F, 1}, {5.0F, 2}, {2.0F, 3}, {0.0F, 4}, {20.0F, 5}}) {
            heap->PushWithAuxiliary(distance,
                                    id,
                                    1000.0F + static_cast<float>(id),
                                    static_cast<BucketIdType>(id % 3U),
                                    id + 10U,
                                    900U + id);
        }

        require_aligned(heap);
        REQUIRE(heap->Size() == 3);
        std::vector<InnerIdType> ids;
        const auto* data = heap->GetDataWithAuxiliary();
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            ids.emplace_back(data[i].record.second);
        }
        std::sort(ids.begin(), ids.end());
        REQUIRE(ids == std::vector<InnerIdType>{1, 3, 4});
    }

    SECTION("zero-capacity fixed heap rejects auxiliary records") {
        auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, true>(allocator.get(), 0);
        heap->PushWithAuxiliary(1.0F, 7, 1007.0F, 1, 17, 907);

        REQUIRE(heap->Empty());
        REQUIRE(heap->Size() == 0);
    }

    SECTION("ordinary pushes carry an explicit missing payload") {
        auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(allocator.get(), 2);
        heap->Push(1.0F, 7);

        REQUIRE(heap->Size() == 1);
        const auto& data = heap->GetDataWithAuxiliary()[0];
        REQUIRE(IsNaNBits(data.auxiliary));
        REQUIRE(data.source_bucket_id == -1);
        REQUIRE(data.source_offset_id == std::numeric_limits<InnerIdType>::max());
        REQUIRE(data.source_version == std::numeric_limits<uint64_t>::max());
        require_aligned(heap);
    }

    SECTION("merge preserves auxiliary records") {
        auto source =
            DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(allocator.get(), 4);
        source->PushWithAuxiliary(4.0F, 0, 1000.0F, 0, 10, 900);
        source->PushWithAuxiliary(2.0F, 1, 1001.0F, 1, 11, 901);
        source->PushWithAuxiliary(3.0F, 2, 1002.0F, 2, 12, 902);

        auto destination =
            DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(allocator.get(), 4);
        destination->Merge(*source);

        REQUIRE(destination->Size() == source->Size());
        require_aligned(destination);

        auto ordinary = DistanceHeap::MakeInstanceBySize<true, false>(allocator.get(), 4);
        ordinary->Merge(*source);
        REQUIRE(ordinary->Size() == source->Size());
        REQUIRE_FALSE(ordinary->StoresAuxiliary());
        REQUIRE(ordinary->GetDataWithAuxiliary() == nullptr);
    }

    SECTION("fixed min heap keeps the largest distances and their payloads") {
        auto heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<false, true>(allocator.get(), 2);
        for (InnerIdType id = 0; id < 5; ++id) {
            heap->PushWithAuxiliary(static_cast<float>(id),
                                    id,
                                    1000.0F + static_cast<float>(id),
                                    static_cast<BucketIdType>(id % 3U),
                                    id + 10U,
                                    900U + id);
        }

        require_aligned(heap);
        REQUIRE(heap->Size() == 2);
        std::vector<InnerIdType> ids;
        const auto* data = heap->GetDataWithAuxiliary();
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            ids.emplace_back(data[i].record.second);
        }
        std::sort(ids.begin(), ids.end());
        REQUIRE(ids == std::vector<InnerIdType>{3, 4});
    }

    SECTION("fixed heaps accept equal distances and replace root provenance") {
        auto max_heap =
            DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, true>(allocator.get(), 1);
        max_heap->PushWithAuxiliary(3.0F, 1, 1001.0F, 1, 11, 901);
        max_heap->PushWithAuxiliary(3.0F, 2, 1002.0F, 2, 12, 902);
        REQUIRE(max_heap->Size() == 1);
        REQUIRE(max_heap->Top().first == 3.0F);
        REQUIRE(max_heap->Top().second == 2);
        REQUIRE(max_heap->GetDataWithAuxiliary()[0].auxiliary == 1002.0F);
        REQUIRE(max_heap->GetDataWithAuxiliary()[0].source_bucket_id == 2);
        REQUIRE(max_heap->GetDataWithAuxiliary()[0].source_offset_id == 12);
        REQUIRE(max_heap->GetDataWithAuxiliary()[0].source_version == 902);
        max_heap->PushWithAuxiliary(4.0F, 3, 1003.0F, 3, 13, 903);
        REQUIRE(max_heap->Top().first == 3.0F);
        REQUIRE(max_heap->Top().second == 2);

        auto min_heap =
            DistanceHeap::MakeInstanceBySizeWithAuxiliary<false, true>(allocator.get(), 1);
        min_heap->PushWithAuxiliary(3.0F, 4, 1004.0F, 4, 14, 904);
        min_heap->PushWithAuxiliary(3.0F, 5, 1005.0F, 5, 15, 905);
        REQUIRE(min_heap->Size() == 1);
        REQUIRE(min_heap->Top().first == 3.0F);
        REQUIRE(min_heap->Top().second == 5);
        REQUIRE(min_heap->GetDataWithAuxiliary()[0].auxiliary == 1005.0F);
        REQUIRE(min_heap->GetDataWithAuxiliary()[0].source_bucket_id == 5);
        REQUIRE(min_heap->GetDataWithAuxiliary()[0].source_offset_id == 15);
        REQUIRE(min_heap->GetDataWithAuxiliary()[0].source_version == 905);
        min_heap->PushWithAuxiliary(2.0F, 6, 1006.0F, 6, 16, 906);
        REQUIRE(min_heap->Top().first == 3.0F);
        REQUIRE(min_heap->Top().second == 5);
    }
}

TEST_CASE("fixed auxiliary heap matches randomized max and min oracles",
          "[ut][distance_heap][auxiliary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    RequireFixedAuxiliaryHeapMatchesOracle<true>(allocator.get());
    RequireFixedAuxiliaryHeapMatchesOracle<false>(allocator.get());
}
