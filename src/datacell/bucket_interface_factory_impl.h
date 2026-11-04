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

#include "bucket_datacell.h"
#include "inner_string_params.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/product_quantization/pq_fastscan_quantizer.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "quantization/scalar_quantization/sq_headers.h"

namespace vsag {

template <typename QuantTemp, typename IOTemp>
BucketInterfacePtr
MakeBucketDataCellInstance(const BucketDataCellParamPtr& param,
                           const IndexCommonParam& common_param) {
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
BucketInterfacePtr
MakeBucketDataCellInstance(const BucketDataCellParamPtr& param,
                           const IndexCommonParam& common_param) {
    std::string quantization_string = param->quantizer_parameter->GetTypeName();
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8) {
        return MakeBucketDataCellInstance<SQ8Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP32) {
        return MakeBucketDataCellInstance<FP32Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4) {
        return MakeBucketDataCellInstance<SQ4Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        return MakeBucketDataCellInstance<SQ4UniformQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        return MakeBucketDataCellInstance<SQ8UniformQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
        return MakeBucketDataCellInstance<ProductQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_PQFS) {
        return MakeBucketDataCellInstance<PQFastScanQuantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_BF16) {
        return MakeBucketDataCellInstance<BF16Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP16) {
        return MakeBucketDataCellInstance<FP16Quantizer<metric>, IOTemp>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_RABITQ) {
        return MakeBucketDataCellInstance<RaBitQuantizer<metric>, IOTemp>(param, common_param);
    }
    return nullptr;
}

template <typename IOTemp>
BucketInterfacePtr
MakeBucketDataCellInstance(const BucketDataCellParamPtr& param,
                           const IndexCommonParam& common_param) {
    auto metric = common_param.metric_;
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return MakeBucketDataCellInstance<MetricType::METRIC_TYPE_L2SQR, IOTemp>(param,
                                                                                 common_param);
    }
    if (metric == MetricType::METRIC_TYPE_IP) {
        return MakeBucketDataCellInstance<MetricType::METRIC_TYPE_IP, IOTemp>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        if (param->use_residual_) {
            return MakeBucketDataCellInstance<MetricType::METRIC_TYPE_IP, IOTemp>(param,
                                                                                  common_param);
        }
        return MakeBucketDataCellInstance<MetricType::METRIC_TYPE_COSINE, IOTemp>(param,
                                                                                  common_param);
    }
    return nullptr;
}

}  // namespace vsag
