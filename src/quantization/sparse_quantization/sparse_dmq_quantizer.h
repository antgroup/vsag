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

#include <array>

#include "hash_types.h"
#include "quantization/quantizer.h"
#include "vsag/dataset.h"

namespace vsag {

class SparseDmqQuantizer : public Quantizer<SparseDmqQuantizer> {
public:
    static constexpr uint32_t BITS = 8;
    static constexpr uint32_t CODEBOOK_SIZE = 1U << BITS;
    static constexpr uint32_t THRESHOLD_COUNT = CODEBOOK_SIZE - 1;

    struct VectorFactors {
        float mean{0.0F};
        float alpha{0.0F};
    };

    struct Codebook {
        std::array<float, THRESHOLD_COUNT> thresholds{};
        std::array<float, CODEBOOK_SIZE> values{};
    };

    struct EncodedHeader {
        uint32_t len{0};
        VectorFactors factors;
    };
    static_assert(sizeof(EncodedHeader) == 12, "DMQ encoded header layout must remain stable");

    SparseDmqQuantizer(uint32_t term_id_limit, Allocator* allocator);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    static bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, float* data) const;

    static bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const float* query, Computer<SparseDmqQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<SparseDmqQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ReleaseComputerImpl(Computer<SparseDmqQuantizer>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] static std::string
    NameImpl();

    [[nodiscard]] uint64_t
    GetEncodedSize(const SparseVector& vector) const;

    [[nodiscard]] static uint32_t
    GetEncodedLength(const uint8_t* codes);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

    void
    ExportModel(const SparseDmqQuantizer& other);

private:
    struct QueryData;

    [[nodiscard]] uint32_t
    GetCodebookIndex(uint32_t term_id) const;

    void
    RebuildCodebookLookup();

    void
    AddCodebookLookup(uint32_t term_id, uint32_t codebook_index);

    static void
    BuildCodebook(float* values, uint32_t length, Codebook* codebook);

    [[nodiscard]] static uint8_t
    EncodeResidual(float residual, const Codebook& codebook);

    [[nodiscard]] static float
    DecodeValue(const VectorFactors& factors, const Codebook& codebook, uint8_t code);

private:
    Vector<uint32_t> codebook_term_ids_;
    Vector<Codebook> codebooks_;
    UnorderedMap<uint32_t, uint32_t> codebook_index_by_term_id_;
    Vector<uint32_t> codebook_index_lookup_;
    uint32_t id_bits_{32};
};

}  // namespace vsag
