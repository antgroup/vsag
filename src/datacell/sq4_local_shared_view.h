#pragma once

#include "flatten_interface.h"
#include "index_common_param.h"

namespace vsag {

class SQ4LocalSharedView final : public FlattenInterface {
public:
    SQ4LocalSharedView(FlattenInterfacePtr precise, const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override;

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx) override;

    bool
    UpdateVector(const void* vector, InnerIdType idx) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override;

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override;

    void
    Prefetch(InnerIdType id) override;

    std::string
    GetQuantizerName() override;

    MetricType
    GetMetricType() override;

    void
    Resize(InnerIdType capacity) override;

    void
    ExportModel(const FlattenInterfacePtr& other) const override;

    bool
    Decode(const uint8_t* codes, float* vector) override;

    bool
    Encode(const float* vector, uint8_t* codes) override;

    const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    InnerIdType
    TotalCount() const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override;

    bool
    InMemory() const override;

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override;

    void
    Move(InnerIdType from, InnerIdType to) override;

    void
    ShrinkToFit(InnerIdType capacity) override;

    IndexCommonParam
    ExportCommonParam() override;

private:
    FlattenInterfacePtr precise_;
    IndexCommonParam common_param_;
};

}  // namespace vsag
