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

#include "half_precision_quantizer_parameter.h"
#include "half_precision_traits.h"
#include "index_common_param.h"
#include "quantization/quantizer.h"

namespace vsag {

/**
 * @brief Half-precision quantizer template for FP16 and BF16 formats.
 *
 * This quantizer stores vectors in half-precision floating point format (2 bytes per dimension).
 * The code layout is: dim * 2 bytes, stored as uint16_t values representing half-precision floats.
 *
 * Supported formats:
 * - FP16Format: IEEE 754 half-precision (16-bit float)
 * - BF16Format: Brain floating point (16-bit with 8-bit exponent)
 *
 * For cosine metric, vectors are normalized before encoding.
 *
 * @tparam Format The half-precision format (FP16Format or BF16Format).
 * @tparam metric The distance metric type (L2SQR, IP, or COSINE).
 */
template <typename Format, MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class HalfPrecisionQuantizer : public Quantizer<HalfPrecisionQuantizer<Format, metric>> {
public:
    explicit HalfPrecisionQuantizer(int dim, Allocator* allocator);

    explicit HalfPrecisionQuantizer(const HalfPrecisionQuantizerParameter<Format>& param,
                                    const IndexCommonParam& common_param);

    explicit HalfPrecisionQuantizer(const QuantizerParamPtr& param,
                                    const IndexCommonParam& common_param);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const float* query,
                     Computer<HalfPrecisionQuantizer<Format, metric>>& computer) const;

    void
    ComputeDistImpl(Computer<HalfPrecisionQuantizer<Format, metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    SerializeImpl(StreamWriter& writer) {
    }

    void
    DeserializeImpl(StreamReader& reader) {
    }

    [[nodiscard]] std::string
    NameImpl() const {
        return Format::TYPE_NAME;
    }
};

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
using FP16Quantizer = HalfPrecisionQuantizer<FP16Format, metric>;

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
using BF16Quantizer = HalfPrecisionQuantizer<BF16Format, metric>;

}  // namespace vsag
