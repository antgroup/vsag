
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

#include "fp16_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief FP16 Quantizer stores vectors in 16-bit half-precision floating-point format.
 *
 * code layout:
 * +------------------------+
 * | fp16-code              |
 * | [dim * 2B]             |
 * +------------------------+
 *
 * - fp16-code: IEEE 754 half-precision floats (required)
 * - 1 sign bit, 5 exponent bits, 10 mantissa bits
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP16Quantizer : public Quantizer<FP16Quantizer<metric>> {
public:
    explicit FP16Quantizer(int dim, Allocator* allocator);

    explicit FP16Quantizer(const FP16QuantizerParamPtr& param,
                           const IndexCommonParam& common_param);

    explicit FP16Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    ProcessQueryImpl(const DataType* query, Computer<FP16Quantizer>& computer) const;

    void
    ComputeDistImpl(Computer<FP16Quantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    SerializeImpl(StreamWriter& writer){};

    void
    DeserializeImpl(StreamReader& reader){};

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_FP16;
    }
};

}  // namespace vsag
