
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

#include <sstream>
#include <string>

#include "impl/transform/transformer_headers.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_interface.h"
#include "transform_quantizer_parameter.h"

namespace vsag {

using DataType = float;

template <MetricType metric>
class TransformQuantizer : public Quantizer<TransformQuantizer<metric>> {
public:
    explicit TransformQuantizer(const TransformQuantizerParamPtr& param,
                                const IndexCommonParam& common_param);

    explicit TransformQuantizer(const QuantizerParamPtr& param,
                                const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) const;

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data) {
        return false;
    }

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
        return false;
    }

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const DataType* query, Computer<TransformQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<TransformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<TransformQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<TransformQuantizer<metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer) {
        return;
    }

    void
    DeserializeImpl(StreamReader& reader) {
        return;
    };

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_TQ;
    }

public:
    VectorTransformerPtr
    MakeTransformerInstance(std::string transform_str,
                            const VectorTransformerParameter& param) const;

    void
    ExecuteChainTransform(DataType* prev_data, const uint32_t* meta_offsets, uint8_t* codes) const;

    float
    ExecuteChainDistanceRecovery(float quantize_dist,
                                 const uint32_t* meta_offsets_1,
                                 const uint32_t* meta_offsets_2,
                                 const uint8_t* codes_1,
                                 const uint8_t* codes_2) const;

public:
    Vector<uint32_t> base_meta_offsets_;   // note that code(quantizer) offset is always 0
    Vector<uint32_t> query_meta_offsets_;  // note that code(quantizer) offset is always 0

    uint32_t align_size_{0};

    QuantizerInterfacePtr quantizer_;

    std::vector<VectorTransformerPtr> transform_chain_;
};

}  // namespace vsag
