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

#pragma once

#include <memory>

#include "algorithm/sindi/sindi_sparse_dmq.h"
#include "algorithm/sparse_index/sparse_index.h"
#include "hash_types.h"
#include "json_types.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/dataset.h"

namespace vsag {

class LabelTable;

class SINDIRerankQueryContext {
public:
    virtual ~SINDIRerankQueryContext() = default;
};

class SINDIRerankBackend {
public:
    virtual ~SINDIRerankBackend() = default;

    virtual void
    Add(const DatasetPtr& base) = 0;

    virtual std::unique_ptr<SINDIRerankQueryContext>
    PrepareQuery(const SparseVector& query) const = 0;

    virtual float
    CalDistanceByInnerId(const SINDIRerankQueryContext& query_context,
                         InnerIdType inner_id) const = 0;

    virtual float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const = 0;

    virtual DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const = 0;

    virtual void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const = 0;

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    [[nodiscard]] virtual int64_t
    GetMemoryUsage() const = 0;

    virtual void
    GetMemoryUsageDetail(JsonType& memory_usage) const {
        memory_usage["total"].SetInt(static_cast<uint64_t>(GetMemoryUsage()));
    }
};

class SINDIFp32RerankBackend : public SINDIRerankBackend {
public:
    explicit SINDIFp32RerankBackend(const IndexCommonParam& common_param);

    void
    Add(const DatasetPtr& base) override;

    std::unique_ptr<SINDIRerankQueryContext>
    PrepareQuery(const SparseVector& query) const override;

    float
    CalDistanceByInnerId(const SINDIRerankQueryContext& query_context,
                         InnerIdType inner_id) const override;

    float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const override;

    DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

private:
    std::shared_ptr<SparseIndex> sparse_index_{nullptr};
};

class SINDIDmqRerankBackend : public SINDIRerankBackend {
public:
    SINDIDmqRerankBackend(uint32_t bits,
                          uint32_t term_id_limit,
                          std::shared_ptr<LabelTable> label_table,
                          const IndexCommonParam& common_param);

    void
    Add(const DatasetPtr& base) override;

    std::unique_ptr<SINDIRerankQueryContext>
    PrepareQuery(const SparseVector& query) const override;

    float
    CalDistanceByInnerId(const SINDIRerankQueryContext& query_context,
                         InnerIdType inner_id) const override;

    float
    CalcDistanceById(const DatasetPtr& vector, int64_t id) const override;

    DatasetPtr
    CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count) const override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

    void
    GetMemoryUsageDetail(JsonType& memory_usage) const override;

private:
    struct EncodedVector {
        uint64_t term_offset{0};
        uint32_t len{0};
        sindi_dmq::DirectDmqVectorFactors factors;
    };

private:
    [[nodiscard]] uint8_t
    LoadCode(uint64_t term_offset, uint32_t term_index) const;

    [[nodiscard]] uint32_t
    LoadId(uint64_t term_offset, uint32_t term_index) const;

    void
    LoadIdBlock(uint64_t term_offset, uint32_t term_index, uint32_t count, uint32_t* ids) const;

    void
    StoreCode(uint64_t term_offset, uint32_t term_index, uint8_t code);

    void
    StoreId(uint64_t term_offset, uint32_t term_index, uint32_t id);

    [[nodiscard]] uint32_t
    GetCodebookIndex(uint32_t term_id) const;

    void
    RebuildCodebookLookup();

    void
    TrainCodebooks(const DatasetPtr& base, bool train_missing_only);

private:
    Allocator* allocator_{nullptr};
    std::shared_ptr<LabelTable> label_table_{nullptr};
    Vector<EncodedVector> encoded_vectors_;
    Vector<uint8_t> id_codes_;
    Vector<uint8_t> value_codes_;
    Vector<uint32_t> codebook_term_ids_;
    Vector<sindi_dmq::DirectDmqCodebook> codebooks_;
    UnorderedMap<uint32_t, uint32_t> codebook_index_by_term_id_;
    Vector<uint32_t> codebook_index_lookup_;
    uint32_t total_bits_{8};
    uint32_t id_bits_{32};
    uint64_t total_term_count_{0};
    int64_t cur_element_count_{0};
};

}  // namespace vsag
