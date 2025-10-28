
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

#include "impl/logger/logger.h"
#include "inner_string_params.h"

namespace vsag {

ProductQuantizerParameter::ProductQuantizerParameter()
    : QuantizerParameter(QUANTIZATION_TYPE_VALUE_PQ) {
}

void
ProductQuantizerParameter::FromJson(const JsonType& json) {
    if (json.Contains(PRODUCT_QUANTIZATION_DIM) &&
        json[PRODUCT_QUANTIZATION_DIM].IsNumberInteger()) {
        int pq_dim = json[PRODUCT_QUANTIZATION_DIM].GetInt();
        if (pq_dim > 0) {
            this->pq_dim_ = pq_dim;
        } else {
            logger::warn("Invalid pq_dim value: {}, using default value: {}", pq_dim, this->pq_dim_);
            this->pq_dim_ = 1; 
            throw VsagException(ErrorType::INVALID_ARGUMENT, "Invalid pq_dim value in ProductQuantizerParameter");
        }
    }

    if (json.Contains(PRODUCT_QUANTIZATION_BITS) &&
        json[PRODUCT_QUANTIZATION_BITS].IsNumberInteger()) {
        int pq_bits = json[PRODUCT_QUANTIZATION_BITS].GetInt();
        if (pq_bits > 0 && pq_bits <= 32) {
            this->pq_bits_ = pq_bits;
        } else {
            logger::warn("Invalid pq_bits value: {}, using default value: {}", pq_bits, this->pq_bits_);
            this->pq_bits_ = 8; // Explicitly set to default value
        }
    }
    
    if (json.Contains(PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE) &&
        json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].IsNumberInteger()) {
        int train_sample_size = json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].GetInt();
        if (train_sample_size > 0 && train_sample_size <= 65536) {
            this->train_sample_size_ = train_sample_size;
        } else {
            logger::warn("Invalid train_sample_size value: {}, using default value: {}", train_sample_size, this->train_sample_size_);
            this->train_sample_size_ = 65536; // Explicitly set to default value
        }
    }
}

JsonType
ProductQuantizerParameter::ToJson() const {
    JsonType json;
    json[QUANTIZATION_TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_PQ);
    json[PRODUCT_QUANTIZATION_DIM].SetInt(this->pq_dim_);
    json[PRODUCT_QUANTIZATION_BITS].SetInt(this->pq_bits_);
    json[PRODUCT_QUANTIZATION_TRAIN_SAMPLE_SIZE].SetInt(this->train_sample_size_);
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
    if (this->train_sample_size_ != pq_other->train_sample_size_) {
        logger::error(
            "ProductQuantizerParameter::CheckCompatibility: "
            "train_sample_size mismatch: {} vs {}",
            this->train_sample_size_,
            pq_other->train_sample_size_);
        return false;
    }
    return true;
}
}  // namespace vsag
