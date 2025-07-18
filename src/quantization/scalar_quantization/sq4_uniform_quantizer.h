
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

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "sq4_uniform_quantizer_parameter.h"

namespace vsag {

using sum_type = float;

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4UniformQuantizer : public Quantizer<SQ4UniformQuantizer<metric>> {
public:
    explicit SQ4UniformQuantizer(int dim, Allocator* allocator, float trunc_rate = 0.05F);

    explicit SQ4UniformQuantizer(const SQ4UniformQuantizerParamPtr& param,
                                 const IndexCommonParam& common_param);

    explicit SQ4UniformQuantizer(const QuantizerParamPtr& param,
                                 const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const DataType* query, Computer<SQ4UniformQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<SQ4UniformQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<SQ4UniformQuantizer<metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM;
    }

public:
    [[nodiscard]] std::pair<DataType, DataType>
    GetLBandDiff() const {
        return {lower_bound_, diff_};
    }

    DataType
    GetCodesSum(const uint8_t* codes) const {
        return *(sum_type*)(codes + offset_codes_sum_);
    }

private:
    DataType lower_bound_{0};
    DataType diff_{0};

    /***
     * code layout: sq-code(fixed) + norm(opt) + sum(opt)
     * for L2 and COSINE, norm is needed for fast computation
     * for IP and COSINE, sum is needed for restoring original distance
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_sum_{0};
    uint64_t offset_codes_sum_{0};

    float scalar_rate_{0.0F};
    float trunc_rate_{0.05F};
};

}  // namespace vsag
