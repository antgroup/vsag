
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

#include "flatten_datacell.h"
#include "flatten_interface.h"
#include "sparse_vector_datacell.h"

namespace vsag {
namespace creator {
#define FOR_EACH_IO_TYPE(MACRO, ...)  \
    MACRO(MemoryBlockIO, __VA_ARGS__) \
    MACRO(MemoryIO, __VA_ARGS__)      \
    MACRO(BufferIO, __VA_ARGS__)      \
    MACRO(AsyncIO, __VA_ARGS__)       \
    MACRO(MMapIO, __VA_ARGS__)        \
    MACRO(ReaderIO, __VA_ARGS__)

#define FOR_EACH_DENSE_QUANTIZER(MACRO)     \
    MACRO(SQ8, SQ8Quantizer)                \
    MACRO(FP32, FP32Quantizer)              \
    MACRO(SQ4, SQ4Quantizer)                \
    MACRO(SQ4_UNIFORM, SQ4UniformQuantizer) \
    MACRO(SQ8_UNIFORM, SQ8UniformQuantizer) \
    MACRO(BF16, BF16Quantizer)              \
    MACRO(FP16, FP16Quantizer)              \
    MACRO(PQ, ProductQuantizer)             \
    MACRO(PQFS, PQFastScanQuantizer)        \
    MACRO(RABITQ, RaBitQuantizer)

#define FOR_EACH_ALL_QUANTIZER(MACRO) \
    FOR_EACH_DENSE_QUANTIZER(MACRO)   \
    MACRO(INT8, INT8Quantizer)        \
    MACRO(SPARSE, SparseQuantizer)

template <typename QuantTemp, typename IOTemp>
FlattenInterfacePtr
create_flatten_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

template <typename QuantTemp, typename IOTemp>
FlattenInterfacePtr
create_quantizer_adapter_cell(const FlattenInterfaceParamPtr& param,
                              const IndexCommonParam& common_param);

template <typename BottomQuantTemp, MetricType metric, typename IOTemp>
FlattenInterfacePtr
create_tq_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

template <typename BottomQuantTemp, MetricType metric, typename IOTemp>
FlattenInterfacePtr
create_rq_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

template <typename BottomQuantTemp, MetricType metric, typename IOTemp>
FlattenInterfacePtr
create_trq_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param);

template <typename QuantTemp, typename IOTemp>
inline FlattenInterfacePtr
make_instance_impl(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    auto& io_param = param->io_parameter;
    auto& quantizer_param = param->quantizer_parameter;
    if (param->name == SPARSE_VECTOR_DATA_CELL) {
        return std::make_shared<SparseVectorDataCell<QuantTemp, IOTemp>>(
            quantizer_param, io_param, common_param);
    }
    if (param->name == FLATTEN_DATA_CELL) {
        return std::make_shared<FlattenDataCell<QuantTemp, IOTemp>>(
            quantizer_param, io_param, common_param);
    }
    return nullptr;
}

}  // namespace creator
}  // namespace vsag
