
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

#include "quantizer_interface.h"

#include "inner_string_params.h"
#include "quantization/quantizer_headers.h"

namespace vsag {
template <MetricType metric>
static QuantizerInterfacePtr
make_instance(const QuantizerParamPtr& param, const IndexCommonParam& common_param) {
    std::string quantization_string = param->GetTypeName();
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8) {
        return std::make_shared<SQ8Quantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP32) {
        return std::make_shared<FP32Quantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4) {
        return std::make_shared<SQ4Quantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        return std::make_shared<SQ4UniformQuantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        return std::make_shared<SQ8UniformQuantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_BF16) {
        return std::make_shared<BF16Quantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_FP16) {
        return std::make_shared<FP16Quantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
        return std::make_shared<ProductQuantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_RABITQ) {
        return std::make_shared<RaBitQuantizer<metric>>(param, common_param);
    }
    if (quantization_string == QUANTIZATION_TYPE_VALUE_SPARSE) {
        return std::make_shared<SparseQuantizer<metric>>(param, common_param);
    }
    return nullptr;
}

QuantizerInterfacePtr
QuantizerInterface::MakeInstance(const QuantizerParamPtr& param,
                                 const IndexCommonParam& common_param) {
    auto metric = common_param.metric_;
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return make_instance<MetricType::METRIC_TYPE_L2SQR>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_IP) {
        return make_instance<MetricType::METRIC_TYPE_IP>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        return make_instance<MetricType::METRIC_TYPE_COSINE>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
