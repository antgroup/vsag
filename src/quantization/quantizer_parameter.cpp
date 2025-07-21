
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

#include "quantizer_parameter.h"

#include <fmt/format.h>

#include "fp32_quantizer_parameter.h"
#include "inner_string_params.h"
#include "product_quantization/pq_fastscan_quantizer_parameter.h"
#include "product_quantization/product_quantizer_parameter.h"
#include "rabitq_quantization/rabitq_quantizer_parameter.h"
#include "scalar_quantization/sq_parameter_headers.h"
#include "sparse_quantization/sparse_quantizer_parameter.h"

namespace vsag {
QuantizerParamPtr
QuantizerParameter::GetQuantizerParameterByJson(const JsonType& json) {
    std::shared_ptr<QuantizerParameter> quantizer_param = nullptr;

    auto type_name = Parameter::TryToParseType(json);
    if (type_name == QUANTIZATION_TYPE_VALUE_FP32) {
        quantizer_param = std::make_shared<FP32QuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_SQ8) {
        quantizer_param = std::make_shared<SQ8QuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM) {
        quantizer_param = std::make_shared<SQ8UniformQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_PQ) {
        quantizer_param = std::make_shared<ProductQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_SQ4) {
        quantizer_param = std::make_shared<SQ4QuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM) {
        quantizer_param = std::make_shared<SQ4UniformQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_BF16) {
        quantizer_param = std::make_shared<BF16QuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_FP16) {
        quantizer_param = std::make_shared<FP16QuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_RABITQ) {
        quantizer_param = std::make_shared<RaBitQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_SPARSE) {
        quantizer_param = std::make_shared<SparseQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else if (type_name == QUANTIZATION_TYPE_VALUE_PQFS) {
        quantizer_param = std::make_shared<PQFastScanQuantizerParameter>();
        quantizer_param->FromJson(json);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("invalid quantizer name {}", type_name));
    }

    return quantizer_param;
}
QuantizerParameter::QuantizerParameter(std::string name) : name_(std::move(name)) {
}
}  // namespace vsag
