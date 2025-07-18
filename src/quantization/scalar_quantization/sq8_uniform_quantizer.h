
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
#include "sq8_uniform_quantizer_parameter.h"

namespace vsag {
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8UniformQuantizer : public Quantizer<SQ8UniformQuantizer<metric>> {
public:
    using norm_type = uint64_t;
    using sum_type = float;

    explicit SQ8UniformQuantizer(int dim, Allocator* allocator);

    SQ8UniformQuantizer(const SQ8UniformQuantizerParamPtr& param,
                        const IndexCommonParam& common_param);

    SQ8UniformQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~SQ8UniformQuantizer() = default;

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
    ProcessQueryImpl(const DataType* query, Computer<SQ8UniformQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<SQ8UniformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<SQ8UniformQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    void
    ReleaseComputerImpl(Computer<SQ8UniformQuantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM;
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
    float scalar_rate_{0.0F};
};

}  // namespace vsag
