
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

#include <shared_mutex>

#include "bucket_interface.h"
#include "byte_buffer.h"
#include "quantization/product_quantization/pq_fastscan_quantizer.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
class BucketDataCell : public BucketInterface {
public:
    explicit BucketDataCell(const QuantizerParamPtr& quantization_param,
                            const IOParamPtr& io_param,
                            const IndexCommonParam& common_param,
                            BucketIdType bucket_count);

    void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        return this->scan_bucket_by_id(result_dists, comp, bucket_id);
    }

    float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        return this->query_one_by_id(comp, bucket_id, offset_id);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) override;

    InnerIdType*
    GetInnerIds(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        return this->inner_ids_[bucket_id].data();
    }

    void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) override {
        this->check_valid_bucket_id(bucket_id);
        this->datas_[bucket_id]->Prefetch(offset_id * code_size_, code_size_);
    }

    void
    Package() override {
        if (GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
            this->package_fastscan();
        }
    }

    void
    ExportModel(const BucketInterfacePtr& other) const override;

    void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return this->quantizer_->Name();
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return this->quantizer_->Metric();
    }

    [[nodiscard]] InnerIdType
    GetBucketSize(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        return this->bucket_sizes_[bucket_id];
    }

private:
    inline void
    check_valid_bucket_id(BucketIdType bucket_id) {
        if (bucket_id >= this->bucket_count_ or bucket_id < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "visited invalid bucket id");
        }
    }

    inline void
    scan_bucket_by_id(float* result_dists,
                      const std::shared_ptr<Computer<QuantTmpl>>& computer,
                      const BucketIdType& bucket_id);

    inline float
    query_one_by_id(const std::shared_ptr<Computer<QuantTmpl>>& computer,
                    const BucketIdType& bucket_id,
                    const InnerIdType& offset_id);
    inline void
    insert_vector_with_locate(const float* vector,
                              const BucketIdType& bucket_id,
                              const InnerIdType& offset_id);

    inline void
    package_fastscan();

private:
    std::shared_ptr<QuantTmpl> quantizer_{nullptr};

    Vector<std::shared_ptr<IOTmpl>> datas_;

    Vector<InnerIdType> bucket_sizes_;

    Vector<std::shared_mutex> bucket_mutexes_;

    Vector<Vector<InnerIdType>> inner_ids_;

    Allocator* const allocator_{nullptr};
};

template <typename QuantTmpl, typename IOTmpl>
BucketDataCell<QuantTmpl, IOTmpl>::BucketDataCell(const QuantizerParamPtr& quantization_param,
                                                  const IOParamPtr& io_param,
                                                  const IndexCommonParam& common_param,
                                                  BucketIdType bucket_count)
    : BucketInterface(),
      datas_(common_param.allocator_.get()),
      bucket_sizes_(bucket_count, 0, common_param.allocator_.get()),
      inner_ids_(bucket_count,
                 Vector<InnerIdType>(common_param.allocator_.get()),
                 common_param.allocator_.get()),
      bucket_mutexes_(bucket_count, common_param.allocator_.get()),
      allocator_(common_param.allocator_.get()) {
    this->bucket_count_ = bucket_count;
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();

    for (int i = 0; i < bucket_count; ++i) {
        this->datas_.emplace_back(std::make_shared<IOTmpl>(io_param, common_param));
    }
}

template <typename QuantTmpl, typename IOTmpl>
float
BucketDataCell<QuantTmpl, IOTmpl>::query_one_by_id(
    const std::shared_ptr<Computer<QuantTmpl>>& computer,
    const BucketIdType& bucket_id,
    const InnerIdType& offset_id) {
    this->check_valid_bucket_id(bucket_id);
    if (offset_id >= this->bucket_sizes_[bucket_id]) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "invalid offset id for bucket");
    }
    float ret;
    bool need_release = false;
    const auto* codes =
        this->datas_[bucket_id]->Read(code_size_, offset_id * code_size_, need_release);
    computer->ComputeDist(codes, &ret);
    if (need_release) {
        this->datas_[bucket_id]->Release(codes);
    }
    return ret;
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::scan_bucket_by_id(
    float* result_dists,
    const std::shared_ptr<Computer<QuantTmpl>>& computer,
    const BucketIdType& bucket_id) {
    constexpr InnerIdType scan_block_size = 32;
    InnerIdType offset = 0;
    this->check_valid_bucket_id(bucket_id);
    auto data_count = this->bucket_sizes_[bucket_id];
    while (data_count > 0) {
        auto compute_count = std::min(data_count, scan_block_size);
        bool need_release = false;
        const auto* codes = this->datas_[bucket_id]->Read(
            code_size_ * compute_count, offset * code_size_, need_release);
        computer->ScanBatchDists(compute_count, codes, result_dists + offset);
        if (need_release) {
            this->datas_[bucket_id]->Release(codes);
        }
        data_count -= compute_count;
        offset += compute_count;
    }
}

template <typename QuantTmpl, typename IOTmpl>
ComputerInterfacePtr
BucketDataCell<QuantTmpl, IOTmpl>::FactoryComputer(const void* query) {
    const auto* float_query = reinterpret_cast<const float*>(query);
    auto computer = this->quantizer_->FactoryComputer();
    computer->SetQuery(float_query);
    return computer;
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    this->quantizer_->Train(reinterpret_cast<const float*>(data), count);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector,
                                                BucketIdType bucket_id,
                                                InnerIdType inner_id) {
    check_valid_bucket_id(bucket_id);
    InnerIdType locate;
    {
        std::lock_guard lock(this->bucket_mutexes_[bucket_id]);
        locate = this->bucket_sizes_[bucket_id];
        this->bucket_sizes_[bucket_id]++;
        inner_ids_[bucket_id].emplace_back(inner_id);
    }
    this->insert_vector_with_locate(reinterpret_cast<const float*>(vector), bucket_id, locate);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::insert_vector_with_locate(const float* vector,
                                                             const BucketIdType& bucket_id,
                                                             const InnerIdType& offset_id) {
    ByteBuffer codes(static_cast<uint64_t>(code_size_), this->allocator_);
    this->quantizer_->EncodeOne(vector, codes.data);
    this->datas_[bucket_id]->Write(
        codes.data,
        code_size_,
        static_cast<uint64_t>(offset_id) * static_cast<uint64_t>(code_size_));
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    BucketInterface::Serialize(writer);
    quantizer_->Serialize(writer);
    for (BucketIdType i = 0; i < this->bucket_count_; ++i) {
        datas_[i]->Serialize(writer);
        StreamWriter::WriteVector(writer, inner_ids_[i]);
    }
    StreamWriter::WriteVector(writer, this->bucket_sizes_);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Deserialize(StreamReader& reader) {
    BucketInterface::Deserialize(reader);
    quantizer_->Deserialize(reader);
    for (BucketIdType i = 0; i < this->bucket_count_; ++i) {
        datas_[i]->Deserialize(reader);
        StreamReader::ReadVector(reader, inner_ids_[i]);
    }
    StreamReader::ReadVector(reader, this->bucket_sizes_);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::package_fastscan() {
    ByteBuffer buffer(code_size_ * 32, this->allocator_);
    for (int64_t i = 0; i < this->bucket_count_; ++i) {
        auto bucket_size = this->bucket_sizes_[i];
        if (bucket_size == 0) {
            continue;
        }
        bool need_release = false;
        const auto* codes = this->datas_[i]->Read(code_size_ * bucket_size, 0, need_release);
        InnerIdType begin = 0;
        while (begin < bucket_size) {
            quantizer_->Package32(codes + begin * code_size_, buffer.data);
            this->datas_[i]->Write(buffer.data, code_size_ * 32, begin * code_size_);
            begin += 32;
        }
        if (need_release) {
            this->datas_[i]->Release(codes);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::ExportModel(const BucketInterfacePtr& other) const {
    std::stringstream ss;
    IOStreamWriter writer(ss);
    this->quantizer_->Serialize(writer);
    ss.seekg(0, std::ios::beg);
    IOStreamReader reader(ss);
    auto ptr = std::dynamic_pointer_cast<BucketDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Export model's bucket datacell failed");
    }
    ptr->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::MergeOther(const BucketInterfacePtr& other, InnerIdType bias) {
    auto ptr = std::dynamic_pointer_cast<BucketDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Merge other's bucket datacell failed");
    }
    for (int i = 0; i < ptr->bucket_count_; ++i) {
        bool need_release = false;
        if (ptr->bucket_sizes_[i] == 0) {
            continue;
        }
        auto* other_data =
            ptr->datas_[i]->Read(ptr->bucket_sizes_[i] * this->code_size_, 0, need_release);
        this->datas_[i]->Write(other_data,
                               ptr->bucket_sizes_[i] * this->code_size_,
                               this->bucket_sizes_[i] * this->code_size_);
        if (need_release) {
            ptr->datas_[i]->Release(other_data);
        }
        this->bucket_sizes_[i] += ptr->bucket_sizes_[i];
        this->inner_ids_[i].reserve(this->bucket_sizes_[i]);
        for (auto id : ptr->inner_ids_[i]) {
            this->inner_ids_[i].emplace_back(id + bias);
        }
    }
}

}  // namespace vsag
