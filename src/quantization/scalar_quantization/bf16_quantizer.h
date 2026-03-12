
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

#include "bf16_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief BF16 Quantizer stores vectors in 16-bit bfloat16 floating-point format.
 *
 * code layout:
 * +------------------------+
 * | bf16-code              |
 * | [dim * 2B]             |
 * +------------------------+
 *
 * - bf16-code: bfloat16 (Brain Float) values (required)
 * - 1 sign bit, 8 exponent bits, 7 mantissa bits
 * - Same exponent range as FP32, less precision
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class BF16Quantizer : public Quantizer<BF16Quantizer<metric>> {
public:
    explicit BF16Quantizer(int dim, Allocator* allocator);

    explicit BF16Quantizer(const BF16QuantizerParamPtr& param,
                           const IndexCommonParam& common_param);

    explicit BF16Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    ProcessQueryImpl(const DataType* query, Computer<BF16Quantizer>& computer) const;

    void
    ComputeDistImpl(Computer<BF16Quantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    SerializeImpl(StreamWriter& writer){};

    void
    DeserializeImpl(StreamReader& reader){};

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_BF16;
    }
};

}  // namespace vsag
