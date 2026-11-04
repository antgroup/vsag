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

#include <cstdint>
#include <string>

#include "binary_quantizer_parameter.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantizer.h"

namespace vsag {

/***
 * @brief Exact storage for bit-packed binary vectors compared with the Hamming distance.
 *
 * code layout:
 * +------------------------------+
 * | packed bits                  |
 * | [ceil(dim / 8) B]            |
 * +------------------------------+
 *
 * - dim counts bits. Bits are packed MSB-first: logical bit i is (code[i / 8] >> (7 - i % 8)) & 1.
 * - The input bytes are stored unchanged except for the padding bits of the last byte (the low
 *   8 - dim % 8 bits when dim % 8 != 0), which are cleared so equal vectors have equal codes.
 *   Queries are cleared the same way, and the distance kernels ignore padding bits anyway.
 * - The distance is popcount(a XOR b) over the first dim bits. It is an exact integer stored in
 *   a float, which is exact because dim never exceeds MAX_BINARY_DIM (2^24).
 *
 * The quantizer API passes vectors as `const float*`; for this quantizer such a pointer always
 * points at packed bytes, never at floats. No training is needed.
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
    DeserializeImpl(StreamReader& reader) {};

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
    // Copies one packed vector and clears its padding bits.
    void
    copy_canonical(const uint8_t* src, uint8_t* dst) const;
};

}  // namespace vsag
