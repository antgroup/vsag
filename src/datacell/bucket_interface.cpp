
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

#include "bucket_interface.h"

#include <fmt/format.h>

#include "bucket_datacell.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/quantizer_headers.h"

namespace vsag {
template <typename QuantTemp, typename IOTemp>
static BucketInterfacePtr
make_instance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param) {
    auto& io_param = param->io_parameter;
    auto& quantizer_param = param->quantizer_parameter;

    return std::make_shared<BucketDataCell<QuantTemp, IOTemp>>(
        quantizer_param,
        io_param,
        common_param,
        static_cast<BucketIdType>(param->buckets_count),
        param->use_residual_);
}

template <MetricType metric, typename IOTemp>
static BucketInterfacePtr
make_instance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param) {
    std::string quantization_string = param->quantizer_parameter->GetTypeName();
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8) {
        return make_instance<SQ8Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP32) {
        return make_instance<FP32Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4) {
        return make_instance<SQ4Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        return make_instance<SQ4UniformQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        return make_instance<SQ8UniformQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
        return make_instance<ProductQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_PQFS) {
        return make_instance<PQFastScanQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_BF16) {
        return make_instance<BF16Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP16) {
        return make_instance<FP16Quantizer<metric>, IOTemp>(param, common_param);
    }
    return nullptr;
}

template <typename IOTemp>
static BucketInterfacePtr
make_instance(const BucketDataCellParamPtr& param, const IndexCommonParam& common_param) {
    auto metric = common_param.metric_;
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return make_instance<MetricType::METRIC_TYPE_L2SQR, IOTemp>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_IP) {
        return make_instance<MetricType::METRIC_TYPE_IP, IOTemp>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        if (param->use_residual_) {
            return make_instance<MetricType::METRIC_TYPE_IP, IOTemp>(param, common_param);
        }
        return make_instance<MetricType::METRIC_TYPE_COSINE, IOTemp>(param, common_param);
    }
    return nullptr;
}

BucketInterfacePtr
BucketInterface::MakeInstance(const BucketDataCellParamPtr& param,
                              const IndexCommonParam& common_param) {
    auto io_type_name = param->io_parameter->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_instance<MemoryBlockIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_MEMORY_IO) {
        return make_instance<MemoryIO>(param, common_param);
    }
    return nullptr;
}
}  // namespace vsag
