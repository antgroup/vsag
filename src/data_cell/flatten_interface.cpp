
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

#include "flatten_interface.h"

#include <fmt/format-inl.h>

#include "common.h"
#include "flatten_datacell.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/quantizer_headers.h"

namespace vsag {
template <typename QuantTemp, typename IOTemp>
static FlattenInterfacePtr
make_instance(const JsonType& flatten_interface_param, const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(
        flatten_interface_param.contains(QUANTIZATION_PARAMS_KEY),
        fmt::format("flatten interface parameters must contains {}", QUANTIZATION_PARAMS_KEY));
    CHECK_ARGUMENT(flatten_interface_param.contains(IO_PARAMS_KEY),
                   fmt::format("flatten interface parameters must contains {}", IO_PARAMS_KEY));
    return std::make_shared<FlattenDataCell<QuantTemp, IOTemp>>(
        flatten_interface_param[QUANTIZATION_PARAMS_KEY],
        flatten_interface_param[IO_PARAMS_KEY],
        common_param);
}

template <MetricType metric, typename IOTemp>
static FlattenInterfacePtr
make_instance(const JsonType& flatten_interface_param, const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(
        flatten_interface_param.contains(QUANTIZATION_TYPE_KEY),
        fmt::format("flatten interface parameters must contains {}", QUANTIZATION_TYPE_KEY));

    std::string quantization_string = flatten_interface_param[QUANTIZATION_TYPE_KEY];
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8) {
        return make_instance<SQ8Quantizer<metric>, IOTemp>(flatten_interface_param, common_param);
    } else if (quantization_string == QUANTIZATION_TYPE_VALUE_FP32) {
        return make_instance<FP32Quantizer<metric>, IOTemp>(flatten_interface_param, common_param);
    } else if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4) {
        return make_instance<SQ4Quantizer<metric>, IOTemp>(flatten_interface_param, common_param);
    } else if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        return make_instance<SQ4UniformQuantizer<metric>, IOTemp>(flatten_interface_param,
                                                                  common_param);
    }
    return nullptr;
}

template <typename IOTemp>
static FlattenInterfacePtr
make_instance(const JsonType& flatten_interface_param, const IndexCommonParam& common_param) {
    auto metric = common_param.metric_;
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return make_instance<MetricType::METRIC_TYPE_L2SQR, IOTemp>(flatten_interface_param,
                                                                    common_param);
    }
    if (metric == MetricType::METRIC_TYPE_IP) {
        return make_instance<MetricType::METRIC_TYPE_IP, IOTemp>(flatten_interface_param,
                                                                 common_param);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        return make_instance<MetricType::METRIC_TYPE_COSINE, IOTemp>(flatten_interface_param,
                                                                     common_param);
    }
    return nullptr;
}

FlattenInterfacePtr
FlattenInterface::MakeInstance(const JsonType& flatten_interface_param,
                               const IndexCommonParam& common_param) {
    CHECK_ARGUMENT(flatten_interface_param.contains(IO_TYPE_KEY),
                   fmt::format("flatten interface parameters must contains {}", IO_TYPE_KEY));
    std::string io_string = flatten_interface_param[IO_TYPE_KEY];
    if (io_string == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_instance<MemoryBlockIO>(flatten_interface_param, common_param);
    }
    if (io_string == IO_TYPE_VALUE_MEMORY_IO) {
        return make_instance<MemoryIO>(flatten_interface_param, common_param);
    }
    return nullptr;
}
}  // namespace vsag
