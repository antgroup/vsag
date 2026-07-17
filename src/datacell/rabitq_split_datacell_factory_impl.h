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
#include "io/io_headers.h"
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

template <MetricType metric, typename IOTmpl>
FlattenInterfacePtr
MakeHomogeneousRaBitQSplitDataCell(const FlattenInterfaceParamPtr& param,
                                   const IndexCommonParam& common_param) {
    return std::make_shared<RaBitQSplitDataCell<metric, IOTmpl, IOTmpl>>(
        param->quantizer_parameter,
        param->io_parameter,
        param->supplement_io_parameter,
        common_param);
}

template <MetricType metric>
FlattenInterfacePtr
MakeRaBitQSplitDataCellForMetric(const FlattenInterfaceParamPtr& param,
                                 const IndexCommonParam& common_param) {
    if (param->supplement_io_parameter != nullptr) {
        const auto& supplement_type = param->supplement_io_parameter->GetTypeName();
        const auto& base_type = param->io_parameter->GetTypeName();
        if (base_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO and
            supplement_type == IO_TYPE_VALUE_ASYNC_IO) {
#if HAVE_LIBAIO
            return std::make_shared<RaBitQSplitDataCell<metric, MemoryBlockIO, AsyncIO>>(
                param->quantizer_parameter,
                param->io_parameter,
                param->supplement_io_parameter,
                common_param);
#else
            auto buffer_supplement_io_param = ConvertRaBitQSplitIOParamType(
                param->supplement_io_parameter, IO_TYPE_VALUE_BUFFER_IO);
            return std::make_shared<RaBitQSplitDataCell<metric, MemoryBlockIO, BufferIO>>(
                param->quantizer_parameter,
                param->io_parameter,
                buffer_supplement_io_param,
                common_param);
#endif
        }
#if !HAVE_LIBAIO
        if (base_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO and
            supplement_type == IO_TYPE_VALUE_BUFFER_IO) {
            return std::make_shared<RaBitQSplitDataCell<metric, MemoryBlockIO, BufferIO>>(
                param->quantizer_parameter,
                param->io_parameter,
                param->supplement_io_parameter,
                common_param);
        }
#endif
        if (base_type != supplement_type) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("rabitq split data cell does not support hybrid IO "
                                            "combination: one-bit={}, supplement={}. Supported "
                                            "hybrid: one-bit=block_memory_io, supplement=async_io.",
                                            base_type,
                                            supplement_type));
        }
    }

    const auto& io_type = param->io_parameter->GetTypeName();
    if (io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, MemoryBlockIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MEMORY_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, MemoryIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_BUFFER_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, BufferIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_ASYNC_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, AsyncIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_MMAP_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, MMapIO>(param, common_param);
    }
    if (io_type == IO_TYPE_VALUE_READER_IO) {
        return MakeHomogeneousRaBitQSplitDataCell<metric, ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
