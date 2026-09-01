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

#include "inner_string_params.h"
#include "io/common/io_type_dispatch.h"
#include "rabitq_split_datacell.h"

namespace vsag {

inline IOParamPtr
ConvertRaBitQSplitIOParamType(const IOParamPtr& io_param, const std::string& type_name) {
    if (io_param == nullptr) {
        return nullptr;
    }
    auto json = io_param->ToJson();
    json[TYPE_KEY].SetString(type_name);
    return IOParameter::GetIOParameterByJson(json);
}

template <MetricType metric, typename IOTmpl, typename QuantizerT>
FlattenInterfacePtr
MakeHomogeneousRaBitQSplitDataCell(const FlattenInterfaceParamPtr& param,
                                   const IndexCommonParam& common_param) {
    return std::make_shared<RaBitQSplitDataCell<metric, IOTmpl, IOTmpl, QuantizerT>>(
        param->quantizer_parameter,
        param->io_parameter,
        param->supplement_io_parameter,
        common_param);
}

template <MetricType metric, typename QuantizerT>
FlattenInterfacePtr
MakeRaBitQSplitDataCellForMetricImpl(const FlattenInterfaceParamPtr& param,
                                     const IndexCommonParam& common_param) {
    if (param->supplement_io_parameter != nullptr) {
        const auto& supplement_type = param->supplement_io_parameter->GetTypeName();
        const auto& base_type = param->io_parameter->GetTypeName();
        const auto supplement_kind = param->supplement_io_parameter->Kind();
        const auto base_kind = param->io_parameter->Kind();
        if (base_kind == IOKind::BLOCK_MEMORY and supplement_kind == IOKind::ASYNC) {
#if HAVE_LIBAIO
            return std::make_shared<
                RaBitQSplitDataCell<metric, MemoryBlockIO, AsyncIO, QuantizerT>>(
                param->quantizer_parameter,
                param->io_parameter,
                param->supplement_io_parameter,
                common_param);
#else
            auto buffer_supplement_io_param = ConvertRaBitQSplitIOParamType(
                param->supplement_io_parameter, IO_TYPE_VALUE_BUFFER_IO);
            return std::make_shared<
                RaBitQSplitDataCell<metric, MemoryBlockIO, BufferIO, QuantizerT>>(
                param->quantizer_parameter,
                param->io_parameter,
                buffer_supplement_io_param,
                common_param);
#endif
        }
#if !HAVE_LIBAIO
        if (base_kind == IOKind::BLOCK_MEMORY and supplement_kind == IOKind::BUFFER) {
            return std::make_shared<
                RaBitQSplitDataCell<metric, MemoryBlockIO, BufferIO, QuantizerT>>(
                param->quantizer_parameter,
                param->io_parameter,
                param->supplement_io_parameter,
                common_param);
        }
#endif
        if (base_kind != supplement_kind) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("rabitq split data cell does not support hybrid IO "
                                            "combination: one-bit={}, supplement={}. Supported "
                                            "hybrid: one-bit=block_memory_io, supplement=async_io.",
                                            base_type,
                                            supplement_type));
        }
    }

    return VisitIOKind(param->io_parameter->Kind(), [&](auto tag) -> FlattenInterfacePtr {
        using IO = typename decltype(tag)::Type;
        if constexpr (std::is_void_v<IO> or std::is_same_v<IO, UringIO>) {
            return nullptr;
        } else {
            return MakeHomogeneousRaBitQSplitDataCell<metric, IO, QuantizerT>(param, common_param);
        }
    });
}

template <MetricType metric>
FlattenInterfacePtr
MakeRaBitQSplitDataCellForMetric(const FlattenInterfaceParamPtr& param,
                                 const IndexCommonParam& common_param,
                                 bool is_transform_quantizer) {
    if (is_transform_quantizer) {
        auto tq_param =
            std::dynamic_pointer_cast<TransformQuantizerParameter>(param->quantizer_parameter);
        CHECK_ARGUMENT(tq_param != nullptr and tq_param->tq_chain_.size() == 1 and
                           tq_param->tq_chain_.front() == TRANSFORMER_TYPE_VALUE_MRLE,
                       "rabitq split transform quantizer requires tq_chain=\"mrle, rabitq\"");
        using QuantizerT = TransformQuantizer<RaBitQuantizer<metric>, metric>;
        return MakeRaBitQSplitDataCellForMetricImpl<metric, QuantizerT>(param, common_param);
    }
    return MakeRaBitQSplitDataCellForMetricImpl<metric, RaBitQuantizer<metric>>(param,
                                                                                common_param);
}

}  // namespace vsag
