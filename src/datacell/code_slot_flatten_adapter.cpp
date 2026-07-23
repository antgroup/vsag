// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "code_slot_flatten_adapter.h"

#include <fmt/format.h>

#include <array>
#include <utility>

#include "common.h"
#include "index_common_param.h"
#include "typing.h"

namespace vsag {

namespace {

class CodeSlotFlattenAdapter : public FlattenInterface {
public:
    CodeSlotFlattenAdapter(FlattenInterfacePtr base,
                           std::shared_ptr<const CodeSlotMap> mapping,
                           Allocator* allocator,
                           const std::atomic<uint64_t>* logical_total_count)
        : base_(std::move(base)),
          mapping_(std::move(mapping)),
          allocator_(allocator),
          logical_total_count_(logical_total_count) {
        this->refresh_metadata();
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            base_->Query(result_dists, computer, mapped_ids, id_count, ctx);
        });
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            base_->QueryWithDistanceFilter(
                result_dists, computer, mapped_ids, id_count, threshold, ctx);
        });
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            base_->QueryWithDistanceLowerBound(
                result_dists, lower_bounds, computer, mapped_ids, id_count, ctx);
        });
    }

    void
    QueryWithDistanceHint(float* result_dists,
                          const float* hint_dists,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            base_->QueryWithDistanceHint(
                result_dists, hint_dists, computer, mapped_ids, id_count, ctx);
        });
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return base_->FactoryComputer(query);
    }

    void
    Train(const void* data, uint64_t count) override {
        base_->Train(data, count);
        this->refresh_metadata();
    }

    void
    InsertVector(const void* vector, InnerIdType idx) override {
        base_->InsertVector(vector, mapping_->Resolve(idx));
    }

    void
    InsertVectorToSlot(const void* vector, CodeSlotIdType code_slot_id) {
        base_->InsertVector(vector, code_slot_id);
    }

    bool
    UpdateVector(const void* vector, InnerIdType idx) override {
        return base_->UpdateVector(vector, mapping_->Resolve(idx));
    }

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override {
        if (idx_vec == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "code-slot adapter requires explicit logical ids");
        }
        Vector<CodeSlotIdType> code_slot_ids(static_cast<uint64_t>(count), allocator_);
        mapping_->ResolveBatch(idx_vec, count, code_slot_ids.data());
        base_->BatchInsertVector(vectors, count, code_slot_ids.data());
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        CodeSlotIdType code_slot_id1 = 0;
        CodeSlotIdType code_slot_id2 = 0;
        mapping_->ResolvePair(id1, id2, code_slot_id1, code_slot_id2);
        return base_->ComputePairVectors(code_slot_id1, code_slot_id2);
    }

    void
    Prefetch(InnerIdType id) override {
        base_->Prefetch(mapping_->Resolve(id));
    }

    std::string
    GetQuantizerName() override {
        return base_->GetQuantizerName();
    }

    MetricType
    GetMetricType() override {
        return base_->GetMetricType();
    }

    void
    Resize(InnerIdType capacity) override {
        (void)capacity;
        reject_physical_operation("Resize");
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        auto other_adapter = std::dynamic_pointer_cast<CodeSlotFlattenAdapter>(other);
        if (other_adapter != nullptr) {
            base_->ExportModel(other_adapter->base_);
            return;
        }
        base_->ExportModel(other);
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        base_->InitIO(io_param);
        this->refresh_metadata();
    }

    uint64_t
    GetMemoryUsage() const override {
        // The adapter does not own the slot map. Its caller accounts for a shared map once.
        return base_->GetMemoryUsage();
    }

    IndexCommonParam
    ExportCommonParam() override {
        return base_->ExportCommonParam();
    }

    bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) override {
        auto ret = base_->SetRuntimeParameters(new_params);
        this->refresh_metadata();
        return ret;
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        return base_->Decode(codes, vector);
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        return base_->Encode(vector, codes);
    }

    bool
    CompareRawVectorWithId(const void* vector, InnerIdType id) override {
        return base_->CompareRawVectorWithId(vector, mapping_->Resolve(id));
    }

    const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        return base_->GetCodesById(mapping_->Resolve(id), need_release);
    }

    void
    Release(const uint8_t* data) const override {
        base_->Release(data);
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        return base_->GetCodesById(mapping_->Resolve(id), codes);
    }

    InnerIdType
    TotalCount() const override {
        return static_cast<InnerIdType>(logical_total_count_->load(std::memory_order_acquire));
    }

    void
    Serialize(StreamWriter& writer) override {
        base_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        base_->Deserialize(std::move(reader));
        this->refresh_metadata();
    }

    bool
    InMemory() const override {
        return base_->InMemory();
    }

    bool
    HoldMolds() const override {
        return base_->HoldMolds();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        (void)other;
        (void)bias;
        reject_physical_operation("MergeOther");
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        (void)from;
        (void)to;
        reject_physical_operation("Move");
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        (void)capacity;
        reject_physical_operation("ShrinkToFit");
    }

    [[nodiscard]] FlattenInterfacePtr
    PhysicalFlatten() const {
        return this->base_;
    }

private:
    [[noreturn]] static void
    reject_physical_operation(const char* operation) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("{} requires the physical flatten behind CodeSlotFlattenAdapter",
                        operation));
    }

    void
    refresh_metadata() {
        this->code_size_ = base_->code_size_;
        this->prefetch_stride_code_ = base_->prefetch_stride_code_;
        this->prefetch_depth_code_ = base_->prefetch_depth_code_;
    }

    template <typename QueryFunc>
    void
    with_mapped_ids(const InnerIdType* idx,
                    InnerIdType id_count,
                    QueryContext* ctx,
                    QueryFunc&& query_func) const {
        if (id_count == 0) {
            query_func(idx);
            return;
        }
        if (id_count == 1) {
            CodeSlotIdType mapped_id = 0;
            mapped_id = mapping_->Resolve(idx[0]);
            query_func(&mapped_id);
            return;
        }
        constexpr InnerIdType stack_mapped_id_count = 128;
        if (id_count <= stack_mapped_id_count) {
            std::array<CodeSlotIdType, stack_mapped_id_count> mapped_ids;
            mapping_->ResolveBatch(idx, id_count, mapped_ids.data());
            query_func(mapped_ids.data());
            return;
        }
        Allocator* allocator = select_query_allocator(ctx, allocator_);
        Vector<CodeSlotIdType> mapped_ids(static_cast<uint64_t>(id_count), allocator);
        mapping_->ResolveBatch(idx, id_count, mapped_ids.data());
        query_func(mapped_ids.data());
    }

    FlattenInterfacePtr base_{nullptr};
    std::shared_ptr<const CodeSlotMap> mapping_{nullptr};
    Allocator* allocator_{nullptr};
    const std::atomic<uint64_t>* logical_total_count_{nullptr};
};

}  // namespace

FlattenInterfacePtr
MakeCodeSlotFlattenAdapter(FlattenInterfacePtr base,
                           std::shared_ptr<const CodeSlotMap> mapping,
                           Allocator* allocator,
                           const std::atomic<uint64_t>* logical_total_count) {
    return std::make_shared<CodeSlotFlattenAdapter>(
        std::move(base), std::move(mapping), allocator, logical_total_count);
}

FlattenInterfacePtr
GetCodeSlotPhysicalFlatten(const FlattenInterfacePtr& flatten) {
    auto adapter = std::dynamic_pointer_cast<CodeSlotFlattenAdapter>(flatten);
    return adapter == nullptr ? flatten : adapter->PhysicalFlatten();
}

void
InsertVectorToCodeSlot(const FlattenInterfacePtr& flatten,
                       const void* vector,
                       CodeSlotIdType code_slot_id) {
    auto adapter = std::dynamic_pointer_cast<CodeSlotFlattenAdapter>(flatten);
    if (adapter != nullptr) {
        adapter->InsertVectorToSlot(vector, code_slot_id);
        return;
    }
    flatten->InsertVector(vector, code_slot_id);
}

}  // namespace vsag
