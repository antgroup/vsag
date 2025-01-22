
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
    InsertVector(const void* vector, BucketIdType bucket_id, LabelType label) override;

    LabelType*
    GetLabel(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        return this->labels_[bucket_id].data();
    }

    void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) override {
        this->check_valid_bucket_id(bucket_id);
        this->datas_[bucket_id]->Prefetch(offset_id * code_size_, code_size_);
    }

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
            throw std::runtime_error("visited invalid bucket id");
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

private:
    std::shared_ptr<QuantTmpl> quantizer_{nullptr};

    Vector<std::shared_ptr<IOTmpl>> datas_;

    Vector<InnerIdType> bucket_sizes_;

    Vector<std::shared_mutex> bucket_mutexes_;

    Vector<Vector<LabelType>> labels_;

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
      labels_(bucket_count,
              Vector<LabelType>(common_param.allocator_.get()),
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
        throw std::runtime_error("invalid offset id for bucket");
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
        computer->ComputeBatchDists(compute_count, codes, result_dists + offset);
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
                                                LabelType label) {
    check_valid_bucket_id(bucket_id);
    InnerIdType locate;
    {
        std::lock_guard lock(this->bucket_mutexes_[bucket_id]);
        locate = this->bucket_sizes_[bucket_id];
        this->bucket_sizes_[bucket_id]++;
        labels_[bucket_id].emplace_back(label);
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
        StreamWriter::WriteVector(writer, labels_[i]);
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
        StreamReader::ReadVector(reader, labels_[i]);
    }
    StreamReader::ReadVector(reader, this->bucket_sizes_);
}

}  // namespace vsag
