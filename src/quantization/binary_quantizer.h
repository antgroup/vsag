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

#include "binary_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantizer.h"

namespace vsag {

/***
 * @brief BinaryQuantizer stores packed binary vectors without converting them
 * into floating-point vectors.
 *
 * A logical dimension of `dim` bits is stored in `ceil(dim / 8)` contiguous
 * bytes. Bits are packed least-significant-bit first inside each byte, and the
 * unused high bits of the final byte are cleared. Hamming distance is computed
 * as `popcount(a XOR b)` over exactly those bytes.
 */
class BinaryQuantizer : public Quantizer<BinaryQuantizer> {
public:
    explicit BinaryQuantizer(int dim, Allocator* allocator);

    BinaryQuantizer(const BinaryQuantizerParamPtr& param, const IndexCommonParam& common_param);

    BinaryQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~BinaryQuantizer() override = default;

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
    SerializeImpl(StreamWriter& writer) {};

    void
    DeserializeImpl(StreamReader& reader) {
        this->query_code_size_ = this->code_size_;
    };

    void
    ProcessQueryImpl(const float* query, Computer<BinaryQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<BinaryQuantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ScanBatchDistImpl(Computer<BinaryQuantizer>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ComputeDistsBatch4Impl(Computer<BinaryQuantizer>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

    void
    ReleaseComputerImpl(Computer<BinaryQuantizer>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_BINARY;
    }

private:
    void
    MaskUnusedBits(uint8_t* codes) const;
};

}  // namespace vsag
