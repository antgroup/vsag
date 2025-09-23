
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
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "quantization/computer.h"
#include "typing.h"

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
    Vector<DataType> vec(this->dim_ * count, this->allocator_);
    for (int64_t i = 0; i < this->dim_ * count; ++i) {
        vec[i] = static_cast<DataType>(reinterpret_cast<const DataT*>(data)[i]);
    }
    return this->inner_quantizer_->Train(vec.data(), count);
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    auto data_int8 = reinterpret_cast<const int8_t*>(data);
    Vector<DataType> vec(this->dim_, this->allocator_);
    for (int64_t i = 0; i < this->dim_; i++) {
        vec[i] = static_cast<DataType>(data_int8[i]);
    }
    this->inner_quantizer_->EncodeOne(vec.data(), codes);
    return true;
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeBatchImpl(const DataType* data,
                                                 uint8_t* codes,
                                                 uint64_t count) {
    auto data_int8 = reinterpret_cast<const int8_t*>(data);
    Vector<DataType> vec(this->dim_ * count, this->allocator_);
    for (int64_t i = 0; i < this->dim_ * count; ++i) {
        vec[i] = static_cast<DataType>(data_int8[i]);
    }
    return this->inner_quantizer_->EncodeBatch(vec.data(), codes, count);
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    Vector<DataType> vec(this->dim_, this->allocator_);
    if (!this->inner_quantizer_->DecodeOne(codes, vec.data())) {
        return false;
    }
    for (int64_t i = 0; i < this->dim_; i++) {
        reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
    }
    return true;
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeBatchImpl(const uint8_t* codes,
                                                 DataType* data,
                                                 uint64_t count) {
    Vector<DataType> vec(this->dim_ * count, this->allocator_);
    if (!this->inner_quantizer_->DecodeBatch(codes, vec.data(), count)) {
        return false;
    }
    for (int64_t i = 0; i < this->dim_ * count; i++) {
        reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
    }
    return true;
}

}  // namespace vsag
