
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

#include "multi_vector_datacell.h"
#include "vsag/options.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
MultiVectorDataCell<QuantTmpl, IOTmpl>::MultiVectorDataCell(
    const QuantizerParamPtr& quantization_param,
    const IOParamPtr& io_param,
    const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      multi_vector_dim_(static_cast<uint32_t>(common_param.dim_)),
      metric_(common_param.metric_) {
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->io_ = std::make_shared<IOTmpl>(io_param, common_param);
    this->offset_io_ =
        std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(), allocator_);
    this->max_capacity_ = 0;
    this->code_size_ = 0;
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    this->quantizer_->Train(static_cast<const float*>(data), count);
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector, InnerIdType idx) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "InsertVector is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                          InnerIdType count,
                                                          InnerIdType* idx_vec) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "BatchInsertVector is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Resize(InnerIdType new_capacity) {
    if (new_capacity <= this->max_capacity_) {
        return;
    }
    this->offset_io_->Resize(static_cast<uint64_t>(new_capacity) * sizeof(uint64_t));
    this->max_capacity_ = new_capacity;
}

template <typename QuantTmpl, typename IOTmpl>
std::string
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetQuantizerName() {
    return this->quantizer_->Name();
}

template <typename QuantTmpl, typename IOTmpl>
MetricType
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMetricType() {
    return this->metric_;
}

template <typename QuantTmpl, typename IOTmpl>
const uint8_t*
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetCodesById(InnerIdType id, bool& need_release) const {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "GetCodesById is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Release(const uint8_t* data) const {
    allocator_->Deallocate(const_cast<uint8_t*>(data));
}

template <typename QuantTmpl, typename IOTmpl>
bool
MultiVectorDataCell<QuantTmpl, IOTmpl>::InMemory() const {
    return FlattenInterface::InMemory();
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "Serialize is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "Deserialize is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
ComputerInterfacePtr
MultiVectorDataCell<QuantTmpl, IOTmpl>::FactoryComputer(const void* query) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "FactoryComputer is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
void
MultiVectorDataCell<QuantTmpl, IOTmpl>::Query(float* result_dists,
                                              const ComputerInterfacePtr& computer,
                                              const InnerIdType* idx,
                                              InnerIdType id_count,
                                              QueryContext* ctx) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "Query is not yet implemented for MultiVectorDataCell");
}

template <typename QuantTmpl, typename IOTmpl>
int64_t
MultiVectorDataCell<QuantTmpl, IOTmpl>::GetMemoryUsage() const {
    int64_t memory = sizeof(MultiVectorDataCell<QuantTmpl, IOTmpl>);
    memory += this->offset_io_->size_;
    if (IOTmpl::InMemory) {
        memory += this->io_->GetMemoryUsage();
    }
    memory += sizeof(QuantTmpl);
    return memory;
}

}  // namespace vsag
