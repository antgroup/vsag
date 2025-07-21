
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

#include "product_quantizer_parameter.h"

#include "inner_string_params.h"
#include "logger.h"

namespace vsag {

ProductQuantizerParameter::ProductQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_PQ) {
}

void
ProductQuantizerParameter::FromJson(const JsonType& json) {
    if (json.contains(PRODUCT_QUANTIZATION_DIM) &&
        json[PRODUCT_QUANTIZATION_DIM].is_number_integer()) {
        this->pq_dim_ = json[PRODUCT_QUANTIZATION_DIM];
    }

    if (json.contains(PRODUCT_QUANTIZATION_BITS) &&
        json[PRODUCT_QUANTIZATION_BITS].is_number_integer()) {
        this->pq_bits_ = json[PRODUCT_QUANTIZATION_BITS];
    }
}

JsonType
ProductQuantizerParameter::ToJson() const {
    JsonType json;
    json[QUANTIZATION_TYPE_KEY] = QUANTIZATION_TYPE_VALUE_PQ;
    json[PRODUCT_QUANTIZATION_DIM] = this->pq_dim_;
    json[PRODUCT_QUANTIZATION_BITS] = this->pq_bits_;
    return json;
}

bool
ProductQuantizerParameter::CheckCompatibility(const ParamPtr& other) const {
    auto pq_other = std::dynamic_pointer_cast<ProductQuantizerParameter>(other);
    if (not pq_other) {
        logger::error(
            "ProductQuantizerParameter::CheckCompatibility: "
            "other parameter is not a ProductQuantizerParameter");
        return false;
    }
    if (this->pq_dim_ != pq_other->pq_dim_) {
        logger::error(
            "ProductQuantizerParameter::CheckCompatibility: "
            "pq_dim mismatch: {} vs {}",
            this->pq_dim_,
            pq_other->pq_dim_);
        return false;
    }
    if (this->pq_bits_ != pq_other->pq_bits_) {
        logger::error(
            "ProductQuantizerParameter::CheckCompatibility: "
            "pq_bits mismatch: {} vs {}",
            this->pq_bits_,
            pq_other->pq_bits_);
        return false;
    }
    return true;
}
}  // namespace vsag
