
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

#include "fp32_quantizer_parameter.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantizer.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP32Quantizer : public Quantizer<FP32Quantizer<metric>> {
public:
    explicit FP32Quantizer(int dim, Allocator* allocator);

    FP32Quantizer(const FP32QuantizerParamPtr& param, const IndexCommonParam& common_param);

    FP32Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~FP32Quantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    SerializeImpl(StreamWriter& writer){};

    void
    DeserializeImpl(StreamReader& reader){};

    void
    ProcessQueryImpl(const DataType* query, Computer<FP32Quantizer<metric>>& computer) const;

    void
    ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<FP32Quantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;
    void
    ComputeDistsBatch4Impl(Computer<FP32Quantizer<metric>>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

    void
    ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_FP32;
    }
};

}  // namespace vsag
