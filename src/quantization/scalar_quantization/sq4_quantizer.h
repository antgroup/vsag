
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
#include "sq4_quantizer_parameter.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4Quantizer : public Quantizer<SQ4Quantizer<metric>> {
public:
    explicit SQ4Quantizer(int dim, Allocator* allocator);

    explicit SQ4Quantizer(const SQ4QuantizerParamPtr& param, const IndexCommonParam& common_param);

    explicit SQ4Quantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

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
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    ProcessQueryImpl(const DataType* query, Computer<SQ4Quantizer>& computer) const;

    void
    ComputeDistImpl(Computer<SQ4Quantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ScanBatchDistImpl(Computer<SQ4Quantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<SQ4Quantizer<metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ4;
    }

private:
    std::vector<DataType> lower_bound_{};
    std::vector<DataType> diff_{};
};

}  // namespace vsag
