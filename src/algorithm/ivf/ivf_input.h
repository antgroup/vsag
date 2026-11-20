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

#include <cstring>
#include <limits>

#include "container_types.h"
#include "data_type.h"
#include "simd/bf16_simd.h"
#include "utils/half_input.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

// IVF routing and stored-code consumers have a float ABI. Decode only the rows
// currently being consumed; never materialize a float copy of the input dataset.
class IVFInput {
public:
    IVFInput(const DatasetPtr& data, DataTypes type) : type_(type) {
        CHECK_ARGUMENT(data != nullptr, "query dataset cannot be null");
        CHECK_ARGUMENT(data->GetDim() > 0, "IVF input dimension must be positive");
        CHECK_ARGUMENT(data->GetNumElements() >= 0, "IVF input count cannot be negative");
        dim_ = data->GetDim();
        count_ = data->GetNumElements();
        CHECK_ARGUMENT(type == DataTypes::DATA_TYPE_FLOAT or type == DataTypes::DATA_TYPE_FP16 or
                           type == DataTypes::DATA_TYPE_BF16,
                       "unsupported IVF input datatype");
        const uint64_t scalar_size =
            type == DataTypes::DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);
        CHECK_ARGUMENT(dim_ <= std::numeric_limits<uint64_t>::max() / scalar_size,
                       "IVF input row size overflow");
        CHECK_ARGUMENT(count_ <= std::numeric_limits<uint64_t>::max() / (dim_ * scalar_size),
                       "IVF input size overflow");
        void* ptr = nullptr;
        get_vectors(type, dim_, data, &ptr, &stride_);
        data_ = static_cast<const uint8_t*>(ptr);
        if (data->GetNumElements() > 0 and data_ == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                type == DataTypes::DATA_TYPE_FLOAT
                                    ? "IVF requires float32 vectors for non-empty datasets"
                                    : "IVF requires float16 vectors for non-empty datasets");
        }
    }

    const float*
    Rows(uint64_t first, uint64_t count, Vector<float>& scratch) const {
        CHECK_ARGUMENT(first <= count_ and count <= count_ - first, "IVF input row range overflow");
        if (count == 0) {
            return nullptr;
        }
        const auto* ptr = data_ + first * stride_;
        if (type_ == DataTypes::DATA_TYPE_FLOAT) {
            return reinterpret_cast<const float*>(ptr);
        }
        scratch.resize(count * dim_);
        for (uint64_t i = 0; i < count * dim_; ++i) {
            uint16_t bits;
            std::memcpy(&bits, ptr + i * sizeof(bits), sizeof(bits));
            scratch[i] = type_ == DataTypes::DATA_TYPE_FP16 ? half_input::FP16ToFloat(bits)
                                                            : generic::BF16ToFloat(bits);
        }
        return scratch.data();
    }

    const void*
    Data() const {
        return data_;
    }

private:
    DataTypes type_;
    uint64_t dim_{0};
    uint64_t count_{0};
    uint64_t stride_{0};
    const uint8_t* data_{nullptr};
};

}  // namespace vsag
