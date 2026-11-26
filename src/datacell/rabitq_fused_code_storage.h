// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>

#include "basic_types.h"

namespace vsag {

class ByteBuffer;
class StreamReader;
class StreamWriter;

// The split datacell owns the fixed-size supplement layout and its selected IO backend.
class FusedSupplementStorage {
public:
    virtual ~FusedSupplementStorage() = default;
    virtual void
    ResizeSupplement(InnerIdType capacity) = 0;
    virtual void
    WriteSupplement(InnerIdType id, const uint8_t* code) = 0;
    virtual void
    ReadSupplements(const InnerIdType* ids, uint64_t count, uint8_t* codes) const = 0;
    virtual void
    SerializeSupplement(StreamWriter& writer) = 0;
    virtual void
    DeserializeSupplement(StreamReader& reader, uint64_t expected_size) = 0;
    virtual bool
    SupplementInMemory() const = 0;
};

inline constexpr uint32_t K_FUSED_CLUSTER_COUNT = 16;

struct RaBitQFusedCodeView {
    const uint8_t* one_bit_code{nullptr};
    const uint8_t* supplement_code{nullptr};
    uint32_t cluster_id{0};
    std::shared_ptr<ByteBuffer> owned_supplement{};
};

/**
 * Internal non-owning bridge between the RaBitQ model/query processor and a fused node slab.
 *
 * Fused codes use cluster-residual semantics. The legacy split 1+7 format stores HNSW-compatible
 * BinData/ExData records; the x=1..4 native formats retain the ordinary split bit-plane encoding.
 * Callers must retain the cluster id and use the fused distance methods selected by the codec.
 */
class RabitQFusedInterface {
public:
    virtual ~RabitQFusedInterface() = default;

    virtual bool
    HasExternalSupplement() const {
        return false;
    }

    [[nodiscard]] virtual bool
    GetFusedCodeView(InnerIdType id, RaBitQFusedCodeView& view) const = 0;

    virtual void
    SetFusedCodes(InnerIdType id,
                  uint32_t cluster_id,
                  const uint8_t* one_bit_code,
                  const uint8_t* supplement_code) = 0;

    virtual void
    PrefetchFusedCodes(InnerIdType id, bool include_supplement) const = 0;

    [[nodiscard]] virtual uint64_t
    FusedOneBitCodeSize() const = 0;

    [[nodiscard]] virtual uint64_t
    FusedSupplementCodeSize() const = 0;
};

}  // namespace vsag
