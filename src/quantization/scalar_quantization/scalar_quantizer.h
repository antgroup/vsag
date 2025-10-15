
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

#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "scalar_quantizer_parameter.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR, int bit = 8>
class ScalarQuantizer : public Quantizer<ScalarQuantizer<metric, bit>> {
public:
    explicit ScalarQuantizer(int dim, Allocator* allocator);

    explicit ScalarQuantizer(const std::shared_ptr<ScalarQuantizerParameter<bit>>& param,
                             const IndexCommonParam& common_param);

    explicit ScalarQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    ProcessQueryImpl(const float* query, Computer<ScalarQuantizer<metric, bit>>& computer) const;

    void
    ComputeDistImpl(Computer<ScalarQuantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ScanBatchDistImpl(Computer<ScalarQuantizer<metric, bit>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<ScalarQuantizer<metric, bit>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        static_assert(bit == 4 || bit == 8, "bit must be 4 or 8");
        if constexpr (bit == 8) {
            return QUANTIZATION_TYPE_VALUE_SQ8;
        } else if constexpr (bit == 4) {
            return QUANTIZATION_TYPE_VALUE_SQ4;
        }
    }

public:
    static constexpr int BIT_PER_DIM = bit;
    static constexpr int MAX_CODE_PER_DIM = 1 << BIT_PER_DIM;  // 2^Bit_PER_DIM

private:
    std::vector<float> lower_bound_{};
    std::vector<float> diff_{};
};

template <MetricType metric>
using SQ8Quantizer = ScalarQuantizer<metric, 8>;

template <MetricType metric>
using SQ4Quantizer = ScalarQuantizer<metric, 4>;

}  // namespace vsag
