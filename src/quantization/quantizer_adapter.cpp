
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

#include "quantization/quantizer_adapter.h"

#include <cmath>
#include <memory>
#include <type_traits>

#include "quantization/computer.h"
#include "quantization/quantizer.h"

namespace vsag {

template <typename QuantT, typename DataT>
QuantizerAdapter<QuantT, DataT>::QuantizerAdapter(const QuantizerParamPtr& param,
                                                  const IndexCommonParam& common_param)
    : Quantizer<QuantizerAdapter<QuantT, DataT>>(common_param.dim_, common_param.allocator_.get()) {
    this->inner_quantizer_ = std::make_shared<QuantT>(param, common_param);
    this->code_size_ = this->inner_quantizer_->GetCodeSize();
    this->query_code_size_ = this->inner_quantizer_->GetQueryCodeSize();
    this->metric_ = common_param.metric_;
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::TrainImpl(const DataType* data, size_t count) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<DataType> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            vec[i] = static_cast<DataType>(data_int8[i]);
        }
        return this->inner_quantizer_->TrainImpl(vec.data(), count);
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::TrainImpl only supports int8_t data type");
        return false;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<DataType> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            vec[i] = static_cast<DataType>(data_int8[i]);
        }
        return this->inner_quantizer_->EncodeOneImpl(vec.data(), codes);
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::EncodeOneImpl only supports int8_t data type");
        return false;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeBatchImpl(const DataType* data,
                                                 uint8_t* codes,
                                                 uint64_t count) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<DataType> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            vec[i] = static_cast<DataType>(data_int8[i]);
        }
        return this->inner_quantizer_->EncodeBatchImpl(vec.data(), codes, count);
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::EncodeBatchImpl only supports int8_t data type");
        return false;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        Vector<DataType> vec(this->dim_, this->allocator_);
        if (!this->inner_quantizer_->DecodeOneImpl(codes, vec.data())) {
            return false;
        }
        for (int64_t i = 0; i < this->dim_; i++) {
            reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
        }
        return true;
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::DecodeOneImpl only supports int8_t data type");
        return false;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeBatchImpl(const uint8_t* codes,
                                                 DataType* data,
                                                 uint64_t count) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        Vector<DataType> vec(this->dim_ * count, this->allocator_);
        if (!this->inner_quantizer_->DecodeBatchImpl(codes, vec.data(), count)) {
            return false;
        }
        for (int64_t i = 0; i < this->dim_ * count; i++) {
            reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
        }
        return true;
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::DecodeBatchImpl only supports int8_t data type");
        return false;
    }
}
template <typename QuantT, typename DataT>
float
QuantizerAdapter<QuantT, DataT>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    return this->inner_quantizer_->ComputeImpl(codes1, codes2);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::SerializeImpl(StreamWriter& writer) {
    this->inner_quantizer_->SerializeImpl(writer);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::DeserializeImpl(StreamReader& reader) {
    this->inner_quantizer_->DeserializeImpl(reader);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ProcessQueryImpl(
    const DataType* query, Computer<QuantizerAdapter<QuantT, DataT>>& computer) const {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* query_int8 = reinterpret_cast<const int8_t*>(query);
        Vector<DataType> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            vec[i] = static_cast<DataType>(query_int8[i]);
        }
        auto& inner_computer = reinterpret_cast<Computer<QuantT>&>(computer);
        this->inner_quantizer_->ProcessQueryImpl(vec.data(), inner_computer);
    } else {
        static_assert(std::is_same_v<DataT, int8_t>,
                      "QuantizerAdapter::ProcessQueryImpl only supports int8_t data type");
    }
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ComputeDistImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer, const uint8_t* codes, float* dists) const {
    auto& inner_computer = reinterpret_cast<Computer<QuantT>&>(computer);
    this->inner_quantizer_->ComputeDistImpl(inner_computer, codes, dists);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ScanBatchDistImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    uint64_t count,
    const uint8_t* codes,
    float* dists) const {
    auto& inner_computer = reinterpret_cast<Computer<QuantT>&>(computer);
    for (uint64_t i = 0; i < count; ++i) {
        this->inner_quantizer_->ComputeDistImpl(
            inner_computer, codes + i * this->code_size_, dists + i);
    }
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ReleaseComputerImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

TEMPLATE_QUANTIZER_ADAPTER(ProductQuantizer, int8_t);
}  // namespace vsag
