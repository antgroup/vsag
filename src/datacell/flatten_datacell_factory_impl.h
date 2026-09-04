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
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/int8_quantizer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/quantizer_headers.h"
#include "quantization/transform_quantization/transform_quantizer_parameter.h"
#include "rabitq_split_datacell_factory.h"

namespace vsag {

template <typename QuantTmpl, typename IOTmpl>
FlattenInterfacePtr
MakeFlattenDataCellInstance(const FlattenInterfaceParamPtr& param,
                            const IndexCommonParam& common_param) {
    if (param->name == FLATTEN_DATA_CELL) {
        return std::make_shared<FlattenDataCell<QuantTmpl, FixedLayout<IOTmpl>>>(
            param->quantizer_parameter, param->io_parameter, common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("Unknown flatten interface name: {}", param->name));
}

template <typename QuantizerT, typename IOTmpl, MetricType metric>
FlattenInterfacePtr
MakeFlattenDataCellInstanceWithTQ(const FlattenInterfaceParamPtr& param,
                                  const IndexCommonParam& common_param,
                                  bool is_transform_quantizer) {
    if (is_transform_quantizer) {
        return MakeFlattenDataCellInstance<TransformQuantizer<QuantizerT, metric>, IOTmpl>(
            param, common_param);
    }
    return MakeFlattenDataCellInstance<QuantizerT, IOTmpl>(param, common_param);
}

template <MetricType metric, typename IOTmpl>
FlattenInterfacePtr
MakeFlattenDataCellInstance(const FlattenInterfaceParamPtr& param,
                            const IndexCommonParam& common_param) {
    const std::string quantization_string = param->quantizer_parameter->GetTypeName();

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        if (quantization_string == QUANTIZATION_TYPE_VALUE_INT8) {
            return MakeFlattenDataCellInstance<INT8Quantizer<metric>, IOTmpl>(param, common_param);
        }
        if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
            return MakeFlattenDataCellInstance<QuantizerAdapter<ProductQuantizer<metric>, int8_t>,
                                               IOTmpl>(param, common_param);
        }
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("INT8 data type does not support {} quantization", quantization_string));
    }

    if (common_param.data_type_ == DataTypes::DATA_TYPE_FP16 ||
        common_param.data_type_ == DataTypes::DATA_TYPE_BF16) {
        if (quantization_string == QUANTIZATION_TYPE_VALUE_FP16) {
            return MakeFlattenDataCellInstance<QuantizerAdapter<FP16Quantizer<metric>, uint16_t>,
                                               IOTmpl>(param, common_param);
        }
        if (quantization_string == QUANTIZATION_TYPE_VALUE_BF16) {
            return MakeFlattenDataCellInstance<QuantizerAdapter<BF16Quantizer<metric>, uint16_t>,
                                               IOTmpl>(param, common_param);
        }
        if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
            return MakeFlattenDataCellInstance<QuantizerAdapter<ProductQuantizer<metric>, uint16_t>,
                                               IOTmpl>(param, common_param);
        }
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("FP16/BF16 data type does not support {} quantization",
                                        quantization_string));
    }

    auto actual_quantization = quantization_string;
    const bool is_transform_quantizer = quantization_string == QUANTIZATION_TYPE_VALUE_TQ;
    if (is_transform_quantizer) {
        const auto tq_param =
            std::dynamic_pointer_cast<TransformQuantizerParameter>(param->quantizer_parameter);
        if (not tq_param) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Expected TransformQuantizerParameter for TQ quantization");
        }
        actual_quantization = tq_param->GetBottomQuantizationName();
    }

    if (actual_quantization == QUANTIZATION_TYPE_VALUE_SQ8) {
        return MakeFlattenDataCellInstanceWithTQ<SQ8Quantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_FP32) {
        return MakeFlattenDataCellInstanceWithTQ<FP32Quantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_SQ4) {
        return MakeFlattenDataCellInstanceWithTQ<SQ4Quantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        return MakeFlattenDataCellInstanceWithTQ<SQ4UniformQuantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        return MakeFlattenDataCellInstanceWithTQ<SQ8UniformQuantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_BF16) {
        return MakeFlattenDataCellInstanceWithTQ<BF16Quantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_FP16) {
        return MakeFlattenDataCellInstanceWithTQ<FP16Quantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_PQ) {
        return MakeFlattenDataCellInstanceWithTQ<ProductQuantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_PQFS) {
        return MakeFlattenDataCellInstanceWithTQ<PQFastScanQuantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }
    if (actual_quantization == QUANTIZATION_TYPE_VALUE_RABITQ) {
        if (param->name == RABITQ_SPLIT_DATA_CELL) {
            return MakeRaBitQSplitDataCell(param, common_param, is_transform_quantizer);
        }
        return MakeFlattenDataCellInstanceWithTQ<RaBitQuantizer<metric>, IOTmpl, metric>(
            param, common_param, is_transform_quantizer);
    }

    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("Unsupported quantization type: {}", actual_quantization));
}

template <MetricType metric>
FlattenInterfacePtr
MakeFlattenDataCellForMetric(const FlattenInterfaceParamPtr& param,
                             const IndexCommonParam& common_param) {
    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return MakeFlattenDataCellInstance<metric, MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return MakeFlattenDataCellInstance<metric, MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return MakeFlattenDataCellInstance<metric, BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return MakeFlattenDataCellInstance<metric, AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_URING_IO) {
        return MakeFlattenDataCellInstance<metric, UringIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return MakeFlattenDataCellInstance<metric, MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return MakeFlattenDataCellInstance<metric, ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
