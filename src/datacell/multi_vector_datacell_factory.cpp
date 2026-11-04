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

#include "flatten_interface_factory.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "multi_vector_datacell.h"
#include "quantization/int8_quantizer.h"
#include "quantization/quantizer_headers.h"

namespace vsag {

template <typename QuantizerT, typename IOTmpl>
FlattenInterfacePtr
make_multi_vector_data_cell_instance(const FlattenInterfaceParamPtr& param,
                                     const IndexCommonParam& common_param) {
    if (param->name == MULTI_VECTOR_DATA_CELL) {
        return std::make_shared<MultiVectorDataCell<QuantizerT, IOTmpl>>(
            param->quantizer_parameter, param->io_parameter, common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("Unknown flatten interface name: {}", param->name));
}

template <MetricType metric, typename IOTmpl>
FlattenInterfacePtr
make_multi_vector_data_cell_instance(const FlattenInterfaceParamPtr& param,
                                     const IndexCommonParam& common_param) {
    if (param->name != MULTI_VECTOR_DATA_CELL) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Unknown flatten interface name: {}", param->name));
    }
    const auto& quantization = param->quantizer_parameter->GetTypeName();
    if (quantization == QUANTIZATION_TYPE_VALUE_FP32) {
        return make_multi_vector_data_cell_instance<FP32Quantizer<metric>, IOTmpl>(param,
                                                                                   common_param);
    }
    if (quantization == QUANTIZATION_TYPE_VALUE_FP16) {
        return make_multi_vector_data_cell_instance<FP16Quantizer<metric>, IOTmpl>(param,
                                                                                   common_param);
    }
    if (quantization == QUANTIZATION_TYPE_VALUE_BF16) {
        return make_multi_vector_data_cell_instance<BF16Quantizer<metric>, IOTmpl>(param,
                                                                                   common_param);
    }
    if (quantization == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        return make_multi_vector_data_cell_instance<SQ8UniformQuantizer<metric>, IOTmpl>(
            param, common_param);
    }
    if (quantization == QUANTIZATION_TYPE_VALUE_INT8) {
        return make_multi_vector_data_cell_instance<INT8Quantizer<metric>, IOTmpl>(param,
                                                                                   common_param);
    }
    throw VsagException(
        ErrorType::INVALID_ARGUMENT,
        fmt::format("multi-vector datacell does not support quantization type: {}", quantization));
}

template <typename IOTmpl>
FlattenInterfacePtr
make_multi_vector_data_cell_for_io(const FlattenInterfaceParamPtr& param,
                                   const IndexCommonParam& common_param) {
    if (common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        return make_multi_vector_data_cell_instance<MetricType::METRIC_TYPE_L2SQR, IOTmpl>(
            param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        return make_multi_vector_data_cell_instance<MetricType::METRIC_TYPE_IP, IOTmpl>(
            param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        return make_multi_vector_data_cell_instance<MetricType::METRIC_TYPE_COSINE, IOTmpl>(
            param, common_param);
    }
    return nullptr;
}

FlattenInterfacePtr
MakeMultiVectorDataCell(const FlattenInterfaceParamPtr& param,
                        const IndexCommonParam& common_param) {
    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_multi_vector_data_cell_for_io<MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return make_multi_vector_data_cell_for_io<MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return make_multi_vector_data_cell_for_io<BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return make_multi_vector_data_cell_for_io<AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_URING_IO) {
        return make_multi_vector_data_cell_for_io<UringIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return make_multi_vector_data_cell_for_io<MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return make_multi_vector_data_cell_for_io<ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
