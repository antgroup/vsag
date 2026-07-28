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

#include "code_slot_map.h"

#include <atomic>
#include <cstring>
#include <future>
#include <limits>
#include <shared_mutex>
#include <sstream>
#include <vector>

#include "code_slot_flatten_adapter.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

namespace {

class RecordingFlatten : public vsag::FlattenInterface {
public:
    RecordingFlatten() {
        this->code_size_ = sizeof(uint32_t);
        this->max_capacity_ = 16;
        this->total_count_ = 0;
    }

    void
    Query(float* result_dists,
          const vsag::ComputerInterfacePtr& computer,
          const vsag::InnerIdType* idx,
          vsag::InnerIdType id_count,
          vsag::QueryContext* ctx = nullptr) override {
        queried_ids.assign(idx, idx + id_count);
        for (vsag::InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = static_cast<float>(idx[i]);
        }
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const vsag::ComputerInterfacePtr& computer,
                            const vsag::InnerIdType* idx,
                            vsag::InnerIdType id_count,
                            float threshold,
                            vsag::QueryContext* ctx = nullptr) override {
        filtered_ids.assign(idx, idx + id_count);
        filter_threshold = threshold;
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const vsag::ComputerInterfacePtr& computer,
                                const vsag::InnerIdType* idx,
                                vsag::InnerIdType id_count,
                                vsag::QueryContext* ctx = nullptr) override {
        lower_bound_ids.assign(idx, idx + id_count);
    }

    void
    QueryWithDistanceHint(float* result_dists,
                          const float* hint_dists,
                          const vsag::ComputerInterfacePtr& computer,
                          const vsag::InnerIdType* idx,
                          vsag::InnerIdType id_count,
                          vsag::QueryContext* ctx = nullptr) override {
        hinted_ids.assign(idx, idx + id_count);
        received_hints.assign(hint_dists, hint_dists + id_count);
    }

    vsag::ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return nullptr;
    }

    void
    Train(const void* data, uint64_t count) override {
    }

    void
    InsertVector(const void* vector, vsag::InnerIdType idx) override {
        inserted_ids.push_back(idx);
        this->total_count_ = std::max(this->total_count_, idx + 1);
    }

    bool
    UpdateVector(const void* vector, vsag::InnerIdType idx) override {
        updated_ids.push_back(idx);
        return true;
    }

    void
    BatchInsertVector(const void* vectors,
                      vsag::InnerIdType count,
                      vsag::InnerIdType* idx_vec) override {
        batch_inserted_ids.assign(idx_vec, idx_vec + count);
    }

    float
    ComputePairVectors(vsag::InnerIdType id1, vsag::InnerIdType id2) override {
        pair_ids = {id1, id2};
        return static_cast<float>(id1 * 10 + id2);
    }

    void
    Prefetch(vsag::InnerIdType id) override {
        prefetched_ids.push_back(id);
    }

    std::string
    GetQuantizerName() override {
        return "recording";
    }

    vsag::MetricType
    GetMetricType() override {
        return vsag::MetricType::METRIC_TYPE_L2SQR;
    }

    void
    Resize(vsag::InnerIdType capacity) override {
        this->max_capacity_ = capacity;
    }

    void
    ExportModel(const vsag::FlattenInterfacePtr& other) const override {
        exported_models.push_back(other);
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        return false;
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        if (vector == nullptr || codes == nullptr) {
            return false;
        }
        auto encoded = static_cast<uint32_t>(*vector);
        std::memcpy(codes, &encoded, sizeof(encoded));
        return true;
    }

    const uint8_t*
    GetCodesById(vsag::InnerIdType id, bool& need_release) const override {
        need_release = false;
        last_code = static_cast<uint32_t>(id);
        return reinterpret_cast<const uint8_t*>(&last_code);
    }

    void
    Release(const uint8_t* data) const override {
    }

    bool
    GetCodesById(vsag::InnerIdType id, uint8_t* codes) const override {
        copied_ids.push_back(id);
        auto encoded = static_cast<uint32_t>(id);
        std::memcpy(codes, &encoded, sizeof(encoded));
        return true;
    }

    void
    MergeOther(const vsag::FlattenInterfacePtr& other, vsag::InnerIdType bias) override {
        merged_bias = bias;
    }

    std::vector<vsag::InnerIdType> queried_ids;
    std::vector<vsag::InnerIdType> filtered_ids;
    std::vector<vsag::InnerIdType> lower_bound_ids;
    std::vector<vsag::InnerIdType> hinted_ids;
    std::vector<float> received_hints;
    std::vector<vsag::InnerIdType> inserted_ids;
    std::vector<vsag::InnerIdType> updated_ids;
    std::vector<vsag::InnerIdType> batch_inserted_ids;
    std::vector<vsag::InnerIdType> pair_ids;
    std::vector<vsag::InnerIdType> prefetched_ids;
    mutable std::vector<vsag::InnerIdType> copied_ids;
    mutable std::vector<vsag::FlattenInterfacePtr> exported_models;
    mutable uint32_t last_code{0};
    float filter_threshold{0.0F};
    vsag::InnerIdType merged_bias{std::numeric_limits<vsag::InnerIdType>::max()};
};

}  // namespace

TEST_CASE("CodeSlotMap binds logical ids to physical slots", "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::CodeSlotMap mapping(allocator.get());

    mapping.ReserveLogicalSize(4);
    auto slot0 = mapping.AllocateSlot();
    mapping.PublishSlot(0, slot0);
    auto slot1 = mapping.AllocateSlot();
    mapping.PublishSlot(1, slot1);
    mapping.PublishSlot(2, slot1);
    auto slot2 = mapping.AllocateSlot();
    mapping.PublishSlot(3, slot2);

    REQUIRE(mapping.PhysicalCount() == 3);
    REQUIRE(mapping.PublishedLogicalCount() == 4);
    REQUIRE(slot0 == 0);
    REQUIRE(slot1 == 1);
    REQUIRE(slot2 == 2);
    REQUIRE(mapping.Resolve(0) == 0);
    REQUIRE(mapping.Resolve(1) == 1);
    REQUIRE(mapping.Resolve(2) == 1);
    REQUIRE(mapping.Resolve(3) == 2);
}

TEST_CASE("CodeSlotMap rejects invalid mappings", "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::CodeSlotMap mapping(allocator.get());

    REQUIRE_THROWS(mapping.Resolve(0));
    REQUIRE_THROWS(mapping.PublishSlot(1, 0));

    mapping.ReserveLogicalSize(2);
    auto slot = mapping.AllocateSlot();
    mapping.PublishSlot(0, slot);
    REQUIRE_THROWS(mapping.PublishSlot(0, slot));
    REQUIRE_THROWS(mapping.PublishSlot(1, slot + 1));
}

TEST_CASE("CodeSlotMap serializes logical to physical slots", "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::CodeSlotMap mapping(allocator.get());

    mapping.ReserveLogicalSize(5);
    mapping.PublishSlot(0, mapping.AllocateSlot());
    auto slot1 = mapping.AllocateSlot();
    mapping.PublishSlot(1, slot1);
    mapping.PublishSlot(2, slot1);
    mapping.PublishSlot(4, mapping.AllocateSlot());

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    mapping.Serialize(writer);

    vsag::CodeSlotMap restored(allocator.get());
    vsag::IOStreamReader reader(stream);
    restored.Deserialize(reader);

    REQUIRE(restored.PhysicalCount() == 3);
    REQUIRE(restored.PublishedLogicalCount() == 4);
    REQUIRE(restored.Resolve(0) == 0);
    REQUIRE(restored.Resolve(1) == 1);
    REQUIRE(restored.Resolve(2) == 1);
    REQUIRE(restored.Resolve(4) == 2);
    REQUIRE_THROWS(restored.Resolve(3));
}

TEST_CASE("CodeSlotMap rejects an impossible serialized physical count",
          "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    constexpr vsag::CodeSlotIdType physical_count = 2;
    std::vector<vsag::CodeSlotIdType> slots = {0};
    vsag::StreamWriter::WriteObj(writer, physical_count);
    vsag::StreamWriter::WriteVector(writer, slots);

    vsag::CodeSlotMap mapping(allocator.get());
    vsag::IOStreamReader reader(stream);
    REQUIRE_THROWS(mapping.Deserialize(reader));
}

TEST_CASE("CodeSlotMap supports concurrent independent binds", "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::CodeSlotMap mapping(allocator.get());

    constexpr vsag::InnerIdType count = 64;
    mapping.ReserveLogicalSize(count);
    std::vector<std::future<vsag::InnerIdType>> futures;
    futures.reserve(count);
    for (vsag::InnerIdType id = 0; id < count; ++id) {
        futures.emplace_back(std::async(std::launch::async, [&mapping, id]() {
            auto slot = mapping.AllocateSlot();
            mapping.PublishSlot(id, slot);
            return slot;
        }));
    }
    for (auto& future : futures) {
        REQUIRE(future.get() < count);
    }

    REQUIRE(mapping.PhysicalCount() == count);
    for (vsag::InnerIdType id = 0; id < count; ++id) {
        REQUIRE(mapping.Resolve(id) < count);
    }
}

TEST_CASE("CodeSlotFlattenAdapter maps logical ids before calling flatten",
          "[ut][datacell][code_slot_map]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto mapping = std::make_shared<vsag::CodeSlotMap>(allocator.get());
    mapping->ReserveLogicalSize(4);
    mapping->PublishSlot(0, mapping->AllocateSlot());
    auto slot1 = mapping->AllocateSlot();
    mapping->PublishSlot(1, slot1);
    mapping->PublishSlot(2, slot1);
    mapping->PublishSlot(3, mapping->AllocateSlot());

    auto physical_codes = std::make_shared<RecordingFlatten>();
    std::atomic<uint64_t> logical_total_count{4};
    auto adapter = vsag::MakeCodeSlotFlattenAdapter(
        physical_codes, mapping, allocator.get(), &logical_total_count);

    REQUIRE(mapping->PhysicalCount() == 3);
    REQUIRE(adapter->TotalCount() == 4);
    REQUIRE(vsag::GetCodeSlotPhysicalFlatten(adapter) == physical_codes);
    REQUIRE(vsag::GetCodeSlotPhysicalFlatten(physical_codes) == physical_codes);
    REQUIRE_THROWS(adapter->Resize(8));
    REQUIRE_THROWS(adapter->Move(0, 1));
    REQUIRE_THROWS(adapter->MergeOther(adapter, 0));
    REQUIRE_THROWS(adapter->ShrinkToFit(2));
    vsag::GetCodeSlotPhysicalFlatten(adapter)->Resize(8);
    REQUIRE(physical_codes->max_capacity_ == 8);

    vsag::InnerIdType logical_ids[] = {0, 2, 3};
    float result_dists[] = {0.0F, 0.0F, 0.0F};
    adapter->Query(result_dists, nullptr, logical_ids, 3);
    REQUIRE(physical_codes->queried_ids == std::vector<vsag::InnerIdType>{0, 1, 2});
    REQUIRE(result_dists[0] == 0.0F);
    REQUIRE(result_dists[1] == 1.0F);
    REQUIRE(result_dists[2] == 2.0F);

    adapter->QueryWithDistanceFilter(result_dists, nullptr, logical_ids, 3, 1.5F);
    REQUIRE(physical_codes->filtered_ids == std::vector<vsag::InnerIdType>{0, 1, 2});
    REQUIRE(physical_codes->filter_threshold == 1.5F);

    float lower_bounds[] = {0.0F, 0.0F, 0.0F};
    adapter->QueryWithDistanceLowerBound(result_dists, lower_bounds, nullptr, logical_ids, 3);
    REQUIRE(physical_codes->lower_bound_ids == std::vector<vsag::InnerIdType>{0, 1, 2});

    float hints[] = {3.0F, 2.0F, 1.0F};
    adapter->QueryWithDistanceHint(result_dists, hints, nullptr, logical_ids, 3);
    REQUIRE(physical_codes->hinted_ids == std::vector<vsag::InnerIdType>{0, 1, 2});
    REQUIRE(physical_codes->received_hints == std::vector<float>{3.0F, 2.0F, 1.0F});

    adapter->InsertVector(nullptr, 2);
    REQUIRE(physical_codes->inserted_ids == std::vector<vsag::InnerIdType>{1});

    REQUIRE(adapter->UpdateVector(nullptr, 3));
    REQUIRE(physical_codes->updated_ids == std::vector<vsag::InnerIdType>{2});

    vsag::InnerIdType batch_ids[] = {2, 1, 3};
    adapter->BatchInsertVector(nullptr, 3, batch_ids);
    REQUIRE(physical_codes->batch_inserted_ids == std::vector<vsag::InnerIdType>{1, 1, 2});

    REQUIRE(adapter->ComputePairVectors(2, 3) == 12.0F);
    REQUIRE(physical_codes->pair_ids == std::vector<vsag::InnerIdType>{1, 2});

    adapter->Prefetch(2);
    REQUIRE(physical_codes->prefetched_ids == std::vector<vsag::InnerIdType>{1});

    bool need_release = false;
    auto* codes = adapter->GetCodesById(2, need_release);
    REQUIRE_FALSE(need_release);
    uint32_t decoded = 0;
    std::memcpy(&decoded, codes, sizeof(decoded));
    REQUIRE(decoded == 1);
    REQUIRE(physical_codes->copied_ids.empty());
    adapter->Release(codes);

    float duplicate_query = 1.0F;
    REQUIRE(adapter->CompareRawVectorWithId(&duplicate_query, 2));
    float different_query = 2.0F;
    REQUIRE_FALSE(adapter->CompareRawVectorWithId(&different_query, 2));

    physical_codes->total_count_ = 2;
    physical_codes->max_capacity_ = 8;
    physical_codes->code_size_ = sizeof(uint32_t);
    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    adapter->Serialize(writer);

    auto restored_physical_codes = std::make_shared<RecordingFlatten>();
    restored_physical_codes->total_count_ = 0;
    restored_physical_codes->max_capacity_ = 0;
    restored_physical_codes->code_size_ = 0;
    auto restored_adapter = vsag::MakeCodeSlotFlattenAdapter(
        restored_physical_codes, mapping, allocator.get(), &logical_total_count);
    vsag::IOStreamReader reader(stream);
    restored_adapter->Deserialize(reader);
    REQUIRE(restored_physical_codes->total_count_ == 2);
    REQUIRE(restored_physical_codes->max_capacity_ == 8);
    REQUIRE(restored_physical_codes->code_size_ == sizeof(uint32_t));
    REQUIRE(restored_adapter->TotalCount() == 4);

    auto model_physical_codes = std::make_shared<RecordingFlatten>();
    auto model_adapter = vsag::MakeCodeSlotFlattenAdapter(
        model_physical_codes, mapping, allocator.get(), &logical_total_count);
    adapter->ExportModel(model_adapter);
    REQUIRE(physical_codes->exported_models.size() == 1);
    REQUIRE(physical_codes->exported_models[0] == model_physical_codes);
}
