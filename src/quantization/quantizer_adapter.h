
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

#include <memory>
#include <string>
#include <type_traits>

#include "data_type.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/computer.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

template <typename QuantT, typename DataT>
class QuantizerAdapter : public Quantizer<QuantizerAdapter<QuantT, DataT>> {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter currently only supports int8_t and uint16_t data types");

public:
    explicit QuantizerAdapter(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    virtual ~QuantizerAdapter() = default;

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    void
    ProcessQueryImpl(const float* query, Computer<QuantizerAdapter<QuantT, DataT>>& computer) const;

    void
    ComputeDistImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    bool
    ComputeDistWithThresholdImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                                 const uint8_t* codes,
                                 float threshold,
                                 float* dists) const;

    bool
    ComputeDistWithLowerBoundImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                                  const uint8_t* codes,
                                  float* dists,
                                  float* lower_bound) const;

    void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const override;

    void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const override;

    void
    ComputeDistsBatch4Impl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

    void
    ScanRowBatchDistImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                         uint64_t count,
                         const uint8_t* codes,
                         float* dists) const;

    void
    ScanBatchDistImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<QuantizerAdapter<QuantT, DataT>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return std::string(QUANTIZATION_ADAPTER_NAME_PREFIX) + inner_quantizer_->Name();
    }

private:
    // Own an actual inner Computer rather than aliasing its representation:
    // TransformQuantizer has a specialized Computer with additional state.
    Computer<QuantT>&
    get_inner_computer(Computer<QuantizerAdapter<QuantT, DataT>>& computer) const;
    std::shared_ptr<QuantT> inner_quantizer_{nullptr};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
};

#define TEMPLATE_QUANTIZER_ADAPTER(QuantType, DataT)                                  \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_L2SQR>, DataT>; \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_IP>, DataT>;    \
    template class QuantizerAdapter<QuantType<MetricType::METRIC_TYPE_COSINE>, DataT>;
}  // namespace vsag
